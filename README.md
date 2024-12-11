This is the source code for our work on "*Using Pseudo-Synonyms to Generate Embeddings for Clinical Terms*".

### Abstract
Existing approaches attempt to explicitly learn clinical term embedding from clinical datasets by training a model, such as word2vec and recurrent neural network or fine-tuning a pre-trained large language model (LLM). While the corpus-based methods require exposure to a rich vocabulary in the training corpus, insufficient contextual information, in clinical terms, makes LLMs prone to failure to generate meaningful embeddings. In this regard, we propose a novel method to generate embeddings for clinical terms using *pseudo-synonyms* - terms that might be associated with a clinical term but not the exact synonyms. The proposed method uses an LLM as a black-box tool and requires no training or fine-tuning. To demonstrate the effectiveness of the learned embeddings, we compared our approach with existing corpus-based embedding approaches on *semantic textual similarity* (STS) tasks on five benchmark datasets. Our proposed method outperformed all existing approaches.

## Setup
We used `pipenv` to create virtual environment. You can install all the required packages using:
```bash
pipenv install
```

### Additional requirements
1. Please follow the instruction to import SNOMED CT using `pymedtermino`. The documentation can be accessed from [this](https://pythonhosted.org/PyMedTermino/) link.
2. Download the pre-trained `UmlsBERT` model from [here](https://github.com/gmichalo/UmlsBERT)
3. To use UMLS get your api key from [here](https://documentation.uts.nlm.nih.gov/rest/authentication.html) and paste it in `.env` file (rename `.env.example` to `.env`)

## Usage
You can run `base.py` file to reproduce our results. For e.g., if you want to generate term embeddings using only the clinical terms itself then you can do so as follows:
```bash
python base.py --terms-only --model-id <model_id> --out <output_file>
```

You can use the proposed filtering method as follows:
```bash
python base.py \\
--model-id <model_id> \\
--m <no_of_distant_terms> \\
--n <no_of_closest_terms> \\
--s <min_dist> \\
--out <output_file>
```
You can specify the knowlsedge source with `--source`. At the moment the valid options are `snomedct`, `umls` and `snomedct+umls`.

You can set `--method` to `all` to use all terms, `closest` only the closest terms, `distant` only distant terms, `both` use both closest and distant terms and `filtered` to use filtered subset. In case of `both` we use `--n` as the number to determine the terms size, i.e. <= n+1 (includes the query terms itself)

```bash
python base.py \\
--method both \\
--n 20 \\
--out <output_file>
```