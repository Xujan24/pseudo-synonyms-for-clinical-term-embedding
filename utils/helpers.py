from typing import List, Dict, Tuple
import math
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import re
from strsimpy.levenshtein import Levenshtein
from pymedtermino.snomedct import *
from nltk.corpus import wordnet as wn
from itertools import chain, product
from ordered_set import OrderedSet
from decouple import config
import requests
import torch.nn.functional as F


def normalize_score(arr, min, max):
    if min == 0 and max == 1:
        norm_arr = arr
    else:
        norm_arr = (arr - min) / (max - min)
    return norm_arr


def get_embeddings(
        model: AutoModel, 
        tokenizer: AutoTokenizer, 
        terms: List[str], 
        device: torch.device
    ) -> Tuple[np.ndarray, np.ndarray]:
    """maps texts to vectors (embeddings)

    Keyword arguments:
    model -- instance of the pretrained language model \\
    tokenizer -- instance of the tokenizer \\
    terms -- list of terms \\
    device -- either cpu or gpu device \\

    Return: return_description
    """

    with torch.no_grad():
        tokenized_inputs = tokenizer(terms, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
        model_output = model(**tokenized_inputs)
    
    mu = __mean_pooling(model_output, tokenized_inputs['attention_mask'])

    return mu


def filter_terms(query: str, terms: List[str], method: str, m: int, n: int, s: int, p: str) -> List[str]:
    """returns a subset of (or whole) terms
    
    Keyword arguments:\\
    query -- query term \\
    terms -- list of all terms \\
    method -- method to create the sub list; 'all', 'top', 'least', 'both', 'filtered' \\
    m -- number of minority terms \\
    n -- number of majority terms \\
    s -- min levenshtein distance (LD) between the closest similar terms and the query term \\
    p -- preference for majority terms; p = 'top' - majority of the terms will be those with min. LD and p='least' - majority will be at max. LD. 
    
    For n = 20, f = 6, length of the returned list will be a maximum of 24 (20 + 3 + 1) 
    
    Return: filtered list of terms
    """

    levenshtein = Levenshtein()
    lev_dists = np.array([levenshtein.distance(s0=query, s1=x) for x in terms])
    
    if method == 'all':
            ## for using all terms
        l = [*terms, query]

    elif method == 'top':
        ## for using top n similar terms
        l = [terms[z] for z in np.argsort(lev_dists)[:n-1]]
        l = [*l, query]

    elif method == 'least':
        ## for using bottom n similar terms
        l = [terms[z] for z in np.flip(np.argsort(lev_dists))[:n-1]]
        l = [*l, query]

    elif method == 'both':
        l1 = [terms[z] for z in np.argsort(lev_dists)[:math.floor(n/2)]]
        l2 = [terms[z] for z in np.flip(np.argsort(lev_dists))[:math.floor(n/2)]]
        l = [*l1, *l2, query]
        del l1, l2

    elif method == 'filtered':
        # We select a fixed number of terms with both small and large levenshtein distances.
        # The closest terms must be at least at m LD from the query term.
        # For example: if the query term is finger and one of the similar terms is fingers then
        # we would want to discard such terms.

        lev_dists_cp = lev_dists.copy()
        mask = lev_dists_cp < s
        lev_dists_cp[mask] = np.max(lev_dists_cp)

        if p == 'top':
            idx_1 = n
            idx_2 = m
        
        elif p == 'least':
            idx_1 = m
            idx_2 = n

        ## take n terms with small lev. dist.
        l1 = [terms[z] for z in np.argsort(lev_dists_cp)[:idx_1]]        ## terms with least lev. dists
        l2 = [terms[z] for z in np.flip(np.argsort(lev_dists))[:idx_2]]  ## terms with large lev. dists

        ## concatinate terms with small and large lev. dist. to get the final list of similar terms
        l = [*l1, *l2, query]

        del l1, l2, lev_dists_cp

    del levenshtein, lev_dists

    return list(set(l))


def get_terms(query: str, src: str) -> List[str]:
    """return a list of associated terms based on SNOMED CT or wordnet
    
    Keyword arguments:
    query -- query term \\
    src -- knowledge base to generate pseudo-synonyms;
    
    Return: a list of associated terms
    """

    if src == 'snomedct':
        terms = __get_terms_using_snomedct(query)

    # elif src == 'wordnet':
    #     term_words = query.split(' ')
    #     synonyms = {}
    #     for i, term_word in enumerate(term_words):
    #         synsets = wn.synsets(term_word)
    #         l_names = list(set(chain.from_iterable([word.lemma_names() for word in synsets])))
    #         synonyms[i] = l_names

    #     items = list(set(list(product(*synonyms.values()))))

    #     del synonyms, term_words

    #     terms = []
    #     for item in items:
    #         syn = ' '.join([' '.join(x.split('_')) for x in item])
    #         terms.append(syn)
    
    elif src == 'umls':
        terms = __get_terms_using_umls(query)

    elif src == 'snomedct+umls':
        snomedct_terms = __get_terms_using_snomedct(query)
        umls_terms = __get_terms_using_umls(query)

        terms = list(OrderedSet([*snomedct_terms, *umls_terms]))

    else:
        raise ValueError(f'Unsupported knowledge base:{src}')

    return terms

    


def cos_sim_score(x: Dict, y: Dict) -> float:
    """
    Computes the cosine similarity between two vectors
    """
    
    return np.dot(x['mean'], y['mean']) / (np.linalg.norm(x['mean']) * np.linalg.norm(y['mean']))


def __mean_pooling(emb: torch.Tensor, mask: torch.Tensor) -> np.array:
    """Returns a fixed-length vector from variable-length word/token embeddings
    
    Keyword arguments:
    model
    emb -- word/token embeddings \\
    mask -- attention mask

    Return: a fixed length (numpy) array
    """
    ""
    token_embeddings = emb[0] #First element of model_output contains all token embeddings
    input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    emb = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    emb = F.normalize(emb, p=2, dim=1)

    return np.mean(emb.detach().cpu().numpy(), axis=0)


def __format_term(x: str) -> str:
    """removes any contents with in small brackets

    Example: "heart failure (disorder)" ➡ "heart failure"
    
    Keyword arguments:
    x -- term
    Return: string
    """
    
    return re.sub(r'\([^)]*\)', '', x).strip().lower()


def __get_terms_using_snomedct(query: str) -> List[str]:
    items = list(SNOMEDCT.search(query))

    terms = []
    for concept in items:
        terms = [*terms, *concept.terms, concept.term]
    
    ## filter out duplicates (if any) and format the terms
    ## compare to set OrderedSet preserves the initial ordering of the list
    ## good for reproducing the results
    terms = list(OrderedSet(terms))
    terms = [__format_term(x) for x in terms]

    del items

    return terms


def __get_terms_using_umls(query: str) -> List[str]:
    query_params = {
        'string':query, 
        'apiKey': __get_umls_api_key(), 
        'pageSize':1000
    }

    res = requests.get('https://uts-ws.nlm.nih.gov/search/current', params=query_params)
    res.encoding = 'utf-8'

    res_json = res.json()
    terms = [x['name'].lower() for x in res_json['result']['results']]
    terms = list(OrderedSet(terms))

    del query_params, res, res_json

    return terms


def __get_umls_api_key() -> str:
    return config('UMLS_API_KEY')