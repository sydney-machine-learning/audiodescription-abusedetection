import pandas as pd
import numpy as np

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from collections import Counter

import re
import os

sem_rep_dir = os.path.join('data', 'semantic_representations')
sem_rep_filename = '{movie}_{model}_{rep_type}_{packing_type}_{pooling_strat}.pkl'

results_dir = os.path.join('data', 'results')
row_level_classification_fp = os.path.join(results_dir, 'row_level_classification-')
sem_rep_metrics_fp = os.path.join(results_dir, 'sem_rep_metrics-')

cat_cols = ['themes', 'violence', 'drug_use', 'sex']
full_cat_cols = ['themes', 'violence', 'language', 'drug_use', 'nudity', 'sex']
classifications = ['G', 'PG', 'M', 'MA 15+', 'R 18+']

# TODO: create dict of models with actual properties
pooling_models = [
    'cardiffnlp/twitter-roberta-large-sensitive-multilabel', # 0
    'cardiffnlp/twitter-roberta-large-emotion-latest',
    'cardiffnlp/twitter-roberta-base-offensive',
    'cardiffnlp/twitter-roberta-base-sentiment-latest',
    # 'FacebookAI/roberta-large',
    # 'nickmuchi/setfit-finetuned-movie-genre-prediction',
    # 'GroNLP/hateBERT',
    # 'microsoft/deberta-v3-large',
    'mrm8488/t5-base-finetuned-imdb-sentiment',
    'NemoraAi/modernbert-chat-moderation-X-V2',
    # 'sentence-transformers/all-MiniLM-L6-v2'  # 5
]

abb_pooling_model_names = [
    'Multilabel',
    'Emotion',
    'Offensive',
    'Twitter Sentiment',
    'IMDb Sentiment',
    'Chat Moderation'
]

rep_types = ['dialogue', 'narration', 'transcript']
packing_types = ['chunks', 'utterances']
pooling_strategies = ['lhsCLS', 'no-stride', 'lhsCLS-fp32', 'full-window']


def apply_cat_order(df: pd.DataFrame) -> pd.DataFrame:
    df.cat = df.cat.astype('category')
    df.cat = df.cat.cat.set_categories(full_cat_cols)

    return df.sort_values('cat')


def apply_model_order(df: pd.DataFrame) -> pd.DataFrame:
    df.model = df.model.astype('category')
    df.model = df.model.cat.set_categories(pooling_models)

    return df.sort_values('model')


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


def agg_and_sort_cv_results(hypothesis: str, final_groupby: str = 'cat', agg_groupby: str = None, top_n: int = 0) -> pd.DataFrame:

    metrics = {'f1_macro', 'acc', 'auroc'}
    if agg_groupby is None:
        agg_groupby = ['model', 'cat', 'rep_type', 'packing_type', 'pooling_strat', 'classifier']

    df = pd.read_parquet(f'{sem_rep_metrics_fp}{hypothesis}.parquet')
    metrics_agg = {x: 'mean' for x in metrics.intersection(df.columns)}

    df = pd.read_parquet(f'{sem_rep_metrics_fp}{hypothesis}.parquet') \
        .groupby(agg_groupby) \
        .agg(metrics_agg) \
        .reset_index() \
        .sort_values(['f1_macro'], ascending=False)
    
    if top_n > 0:
        df = df.groupby(final_groupby).head(top_n).reset_index(drop=True)

    return df
            
# TODO: reference properly (changed heavily) and/or improve efficiency (so slow)
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
    ngram_df = pd.DataFrame.from_dict(ngram_counts, orient='index', columns=['Frequency']).reset_index(names=['Ngram'])
    ngram_df['Ngram'] = ngram_df['Ngram'].apply(lambda x: ' '.join(x))
    ngram_df['Percentage'] = ngram_df['Frequency'] / ngram_df['Frequency'].sum() * 100
    ngram_df = ngram_df.sort_values('Percentage', ascending=False).iloc[:top_n]
    
    return ngram_df