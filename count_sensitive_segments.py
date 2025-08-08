import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(asctime)s - %(message)s')

import pandas as pd
import numpy as np

import torch
device = torch.device("cuda")

from torch.profiler import profile, record_function, ProfilerActivity

import transformers

from transformers import (
    AutoTokenizer,
    AutoModel,
    T5EncoderModel,
    DataCollatorWithPadding
)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA

import gc
import pickle
import os
from tqdm import tqdm

import data_extraction as da
import modelling as md
import utils

from typing import List, Tuple
import numpy.typing as npt

import extract_sem_representation as esr
import collect_model_results as cmr

import copy

n_jobs = -1


def train_seg_count_models(df: pd.DataFrame, ratings: npt.ArrayLike, max_iter: int, pca_params, rep_type):

    scaler = StandardScaler()
    basic_model = LogisticRegression(class_weight='balanced', max_iter=max_iter, n_jobs=n_jobs)

    if pca_params is None:
        pca_params = {'n_components': 32}
    use_pca = pca_params['n_components'] != 'original'

    pca = PCA(**pca_params)

    models = dict()
    cases = esr.get_cases([md.pooling_models[0]], ['dialogue', 'transcript'], [rep_type], [md.pooling_strategies[0]])

    for pooling_model_name, rep_type, packing_type, pooling_strat in tqdm(cases):

        logging.info(f'{pooling_model_name} {rep_type} {packing_type} {pooling_strat}')
        rep_list = esr.get_or_create_movie_sem_reps(df, pooling_model_name, rep_type, packing_type, pooling_strat, device, use_profiler=False)

        # Stack results, shift to numpy and reshape
        X = torch.stack([x.reshape(-1) for x in rep_list]).cpu().numpy().reshape(len(rep_list), -1)

        for cat in md.full_cat_cols:
            cat_idx = md.full_cat_cols.index(cat)

            y = np.array(ratings)[:, cat_idx]
            X_scaled = scaler.fit_transform(X)

            if use_pca:
                X_scaled_pca = pca.fit_transform(X_scaled)
            else:
                X_scaled_pca = X_scaled

            basic_model.fit(X_scaled_pca, y)
            models[rep_type, cat] = copy.deepcopy(scaler), copy.deepcopy(pca), copy.deepcopy(basic_model)

    transcript_models = {cat: model for (rep_type, cat), model in models.items() if rep_type == 'transcript'}
    dialogue_models = {cat: model for (rep_type, cat), model in models.items() if rep_type == 'dialogue'}

    return dialogue_models, transcript_models


def _extract_seg_counts_for_single_movie(all_segments, pooling_model, model_dict, data_collator, device, batch_size=64):
    
    loader = torch.utils.data.DataLoader(
        all_segments,
        batch_size=batch_size,
        collate_fn=data_collator,
        pin_memory=True,
        shuffle=False
    )
    seg_pred = []
    
    with torch.no_grad():
        for ii, batch in enumerate(loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            outputs = pooling_model(**batch, output_hidden_states=True)
            
            # Get the CLS rep
            last_layer = outputs.hidden_states[-1][:, 0, :]
            sec_last_layer = outputs.hidden_states[-2][:, 0, :]
            cls_rep = torch.cat([last_layer, last_layer, sec_last_layer, sec_last_layer], dim=1).cpu().numpy()

            # Iterate through models and segments in each batch
            for cat, (scaler, pca, model) in model_dict.items():
                scaled_cls_rep = scaler.transform(cls_rep)
                pca_cls_rep = pca.transform(scaled_cls_rep)
                # Apply model to cls rep
                preds = model.predict(pca_cls_rep)
                seg_pred.extend([{'pred': curr_pred, 'cat': cat} for curr_pred in preds])

    return seg_pred


def get_seg_counts(df, pooling_model_name, rep_type, packing_type, model_dict, device) -> pd.DataFrame:
    
    movie_list = list(df.movie.unique())

    # Use 2048 tokens with ModernBERT Model
    chunk_max_len = 512 if pooling_model_name != md.pooling_models[-1] else 2048
    utterances_max_len = 512 if pooling_model_name != md.pooling_models[-1] else 1024
    stride = 0
    
    # Initialise models, tokenizers and similar artifacts
    tokenizer = AutoTokenizer.from_pretrained(pooling_model_name)
    pooling_model = AutoModel.from_pretrained(pooling_model_name, add_pooling_layer=False)
    pooling_model.half()
    pooling_model.to(device)
    pooling_model.eval()

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt", padding='longest')
    
    batch_size = 32 if packing_type == 'chunks' else 64
    
    if rep_type != 'transcript':
        df = df[df.type.eq(rep_type)]
        
    if packing_type == 'chunks':
        all_enc, movie_indices = esr._get_chunked_encodings(df, stride, tokenizer, chunk_max_len)

    else:
        df = esr._agg_narrator_seg(df).sort_values(['movie', 'start_time'])
        all_enc, movie_indices = esr._get_utterance_encodings(df, tokenizer, utterances_max_len, label_speech_type=False)

    seg_counts = []

    for ii, movie in enumerate(tqdm(movie_list)):
        
        curr_idx = movie_indices[ii]
        if curr_idx[0] != movie:
            raise ValueError('Movie indices are out of sync')
        segments = all_enc[curr_idx[1]:curr_idx[2]]
            
        curr_movie_counts = _extract_seg_counts_for_single_movie(segments, pooling_model, model_dict, data_collator, device, batch_size)

        for row in curr_movie_counts:
            row['movie'] = movie
            row['rep_type'] = rep_type

        seg_counts.extend(curr_movie_counts)

        if ii % 10 == 9:
            torch.cuda.empty_cache()
            # torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()
            gc.collect()

    seg_counts_df = pd.DataFrame(seg_counts)

    return seg_counts_df

    
def main():

    torch.cuda.empty_cache()
    # torch.cuda.ipc_collect()
    gc.collect()

    rep_type = 'utterances'

    df, ratings = cmr.get_text_and_ratings()
    dialogue_models, transcript_models = train_seg_count_models(df, ratings, int(1e5), {'n_components': 32}, rep_type)

    # dialogue_counts_df = get_seg_counts(df, md.pooling_models[0], 'dialogue', rep_type, dialogue_models, device)
    # dialogue_counts_df.to_parquet(os.path.join(md.results_dir, f'dialogue_{rep_type}_film_segment_classification.parquet'))

    transcript_counts_df = get_seg_counts(df, md.pooling_models[0], 'transcript', rep_type, transcript_models, device)
    transcript_counts_df.to_parquet(os.path.join(md.results_dir, f'transcript_{rep_type}_film_segment_classification.parquet'))


if __name__ == "__main__":
    main()
