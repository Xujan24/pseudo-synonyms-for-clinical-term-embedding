import torch
from transformers import AutoTokenizer, AutoModel
from pymedtermino.snomedct import *
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import yaml
from tqdm import tqdm
from utils.helpers import normalize_score, get_embeddings, cos_sim_score, get_terms, filter_terms
import argparse

VALID_SOURCES = ['snomedct', 'umls', 'snomedct+umls']
ALLOWED_METHODS = ['all', 'closest', 'distant', 'both', 'filtered']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--terms-only', help='Calculate scores using clinical terms only', action='store_true')
    parser.add_argument('--source', help='source of clinical terms', type=str, default=VALID_SOURCES[0])
    parser.add_argument('--model-id', help='pretrained model id from huggingface', default='sentence-transformers/all-mpnet-base-v1')
    parser.add_argument('--method', help='method to select similar terms [all, top, least, filtered]', type=str, default='filtered')
    parser.add_argument('--m', help='number of distant terms', type=int, default=3)
    parser.add_argument('--n', help='number of closest terms', type=int, default=20)
    parser.add_argument('--s', help='Threshold to filter similar terms using Levenshtein distance', type=int, default=20)
    parser.add_argument('--score', help='Pearson (per) or Spearman (spe) correlation coefficient score.', type=str, default='per')
    parser.add_argument('--exclude-null', help='Whether to exclude those terms with no similar terms.', action='store_true')
    parser.add_argument('--out', help='output filename', type=str)
    args = parser.parse_args()

    if not args.terms_only and args.method not in ALLOWED_METHODS:
        raise ValueError(f'Invalid option for --method. Valid options include {ALLOWED_METHODS}')
    
    if args.source not in VALID_SOURCES:
        raise ValueError(f'Invalid option for --source. Valid options include {VALID_SOURCES}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ## load pretrained tokenizer and model using huggingface transformers
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModel.from_pretrained(args.model_id, device_map=device)

    source = args.source

    ## load benchmark data config file
    with open('./benchmarks.yaml', 'r') as file:
        benchmarks = yaml.safe_load(file)

    ## initialize output file
    out = open(f'./results/{args.out}', 'w')
    out.write('Configurations: \n')
    out.write(f'\t model: {args.model_id} \n')
    if args.terms_only:
        out.write('\t method: terms only\n\n\n')
    else:
        out.write(f'\t method: using similar terms \n')
        out.write(f'\t approach: {args.method} \n')
        if not args.terms_only:
            out.write(f'\t source: {args.source} \n')
        if args.method not in ['all', 'distant']:
            out.write(f'\t n: {args.n} \n')
        if args.method not in ['all', 'closest']:
            out.write(f'\t m: {args.m} \n')
        if args.method == 'filtered':
            out.write(f'\t s: {args.s} \n')
        
        out.write(f'\t metric {'Pearson' if args.score == 'per' else 'Spearman'} \n')
        out.write(f'\t exclude null: {args.exclude_null} \n\n\n')
    out.write('Benchmark \t\t\t PCC \t\t\t p-value\n')

    for k, v in benchmarks.items():
        tqdm.write(f'Processing {k} benchmark.')
        data = pd.read_csv(v.get('data_dir'), sep='\t')

        ## get terms
        term1 = list(set(data['Term1'].tolist()))
        term2 = list(set(data['Term2'].tolist()))
        terms = list(set([*term1, *term2]))

        del term1, term2

        ## normalize scores
        scores = normalize_score(np.asarray(data['Score'].tolist()), min=v.get('min'), max=v.get('max'))

        ## generate term embeddings
        embs = {}
        if args.terms_only:
            tqdm.write('Generating embeddings using clinical terms only.')
        else:
            tqdm.write('Generating embeddings using similar terms.')
        
        for i in tqdm(range(len(terms))):
            term = terms[i]
            no_ps = False
            if args.terms_only:
                ## using terms only                
                mu = get_embeddings(model=model, tokenizer=tokenizer, terms=[term], device=device)
            
            else:
                ## using pseudo-synonyms           
                container = get_terms(term, source)

                if len(container) > 0:
                    ## if there's large number of psuedo-synonyms, then we apply filtering
                    ## use all psuedo-synonyms, otherwise.
                    if len(container) > args.n:
                        l = filter_terms(query=term, terms=container, method=args.method, n=args.n, m=args.m, s=args.s)
                    else:
                        l = [*container, term]
                        l = list(set(container))

                    mu = get_embeddings(model=model, tokenizer=tokenizer, terms=l, device=device)
                
                else:
                    no_ps = True
                    ## if no similar terms were found using SNOMED CT
                    if args.exclude_null:
                        ## option 1 exclude it from the finar score, if --exclude-null is True
                        mu = None
                    else:
                        mu = get_embeddings(model=model, tokenizer=tokenizer, terms=[term], device=device)
                    

            if mu is None:
                embs[term] = None
                continue

            embs[term] = {
                'mean': mu,
                'no_ps': no_ps
                }


        ## calculate the scores between the pairs
        pred = []
        no_ps_counter = 0
        tqdm.write('Calculating scores')
        for i in range(len(data)):
            term1, term2 = data.iloc[i]['Term1'], data.iloc[i]['Term2']

            emb1, emb2 = embs.get(term1), embs.get(term2)

            if emb1 is None or emb2 is None:
                pred.append(None)
                continue
            
            ## count number of pairs with either one or both terms with no pseudo-synonyms
            if emb1.get('no_ps') or emb2.get('no_ps'):
                no_ps_counter += 1

            d = cos_sim_score(emb1, emb2)
            pred.append(d)


        none_idxs = [i for i, item in enumerate(pred) if item is None]
        
        pred = [item for i, item in enumerate(pred) if i not in none_idxs]
        scores = [item for i, item in enumerate(scores) if i not in none_idxs]

        
        ## calculate the pearson's correlation coefficient
        if args.score == 'per':
            corr, p_value = pearsonr(scores, pred)
            out.write(f'{k} \t\t\t {round(corr, 4)}[{no_ps_counter}] \t\t\t {p_value}\n')
        else:
            rho, p_value = spearmanr(scores, pred)
            out.write(f'{k} \t\t\t {round(rho, 4)}[{no_ps_counter}] \t\t\t {p_value}\n')
        
    ## close output file
    out.close()