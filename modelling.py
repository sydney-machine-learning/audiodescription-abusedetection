import pandas as pd
import numpy as np

import torch
from transformers import TrainerCallback, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, roc_auc_score

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from collections import Counter

from typing import List, Callable

import re
import os

sem_rep_dir = os.path.join('data', 'semantic_representations')
sem_rep_filename = '{movie}_{model}_{rep_type}_{packing_type}_{pooling_strat}.pkl'

cat_cols = ['themes', 'violence', 'drug_use', 'sex']
full_cat_cols = ['themes', 'violence', 'language', 'drug_use', 'nudity', 'sex']
classifications = ['G', 'PG', 'M', 'MA 15+', 'R 18+']

# TODO: create dict of models with actual properties
pooling_models = [
    'cardiffnlp/twitter-roberta-large-sensitive-multilabel', # 0
    'cardiffnlp/twitter-roberta-base-offensive',
    'cardiffnlp/twitter-roberta-base-sentiment-latest',
    # 'FacebookAI/roberta-large',
    # 'nickmuchi/setfit-finetuned-movie-genre-prediction',
    # 'GroNLP/hateBERT',
    # 'microsoft/deberta-v3-large',
    'joeddav/distilbert-base-uncased-go-emotions-student', 
    'mrm8488/t5-base-finetuned-imdb-sentiment',
    # 'NemoraAi/modernbert-chat-moderation-X-V2',
    'sentence-transformers/all-MiniLM-L6-v2'  # 5
]

rep_types = ['dialogue', 'narration', 'transcript']
packing_types = ['chunks', 'utterances']
pooling_strategies = ['lhs2CLS', 'lhs1CLS'] #lhs2CLS_fp32


def convert_col_to_ordinal(series: pd.Series, compact: bool = True) -> pd.Series:
    
    low_cat = ['none', 'very mild']
    low_med_cat = ['mild']
    med_cat = ['moderate']
    high_cat = ['strong', 'high']
    
    if compact:
        new_series = np.select(
            [series.isin(low_cat), series.isin(low_med_cat), series.isin(med_cat), series.isin(high_cat)],
            [0, 1, 2, 3],
            default=-1
        )
    else:
        new_series = np.select(
            [series.eq(low_cat[0]), series.eq(low_cat[1]), series.eq(low_med_cat[0]), series.eq(med_cat[0]), series.eq(high_cat[0]), series.eq(high_cat[1])],
            [0, 1, 2, 3, 4, 5],
            default=-1
        )
    
    return new_series
            

def process_text(text: str, excl_stopwords: bool):
    
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word.isalpha()]
    if excl_stopwords:
        filtered_tokens = [word for word in filtered_tokens if word not in stop_words]
        
    return filtered_tokens


def get_ngram_counts(text, n, top_n=10, excl_stopwords: bool = True):
    
    tokens = process_text(text, excl_stopwords)
    ngram_list = list(ngrams(tokens, n))
    ngram_counts = Counter(ngram_list)
    ngram_df = pd.DataFrame(ngram_counts.most_common(top_n), columns=['Ngram', 'Frequency'])
    ngram_df['Ngram'] = ngram_df['Ngram'].apply(lambda x: ' '.join(x))
    
    return ngram_df


