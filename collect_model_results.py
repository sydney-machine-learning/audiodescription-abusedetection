import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(asctime)s - %(message)s')

import pandas as pd
pd.options.mode.chained_assignment = None

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
device = torch.device('cuda')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA

from tqdm import tqdm

import utils
import data_extraction as da
import extract_sem_representation as esr

import modelling as md

import os

from typing import Tuple, Dict
import numpy.typing as npt

n_folds = 5
seed = 42

def get_text_and_ratings() -> Tuple[pd.DataFrame, npt.ArrayLike]:
    df = pd.read_parquet(da.cleaned_dataset_fp).sort_values(['movie', 'start_time'])

    for col in md.full_cat_cols:
        df[col] = md.convert_col_to_ordinal(df[col], compact=False)

    df['rating'] = df[md.full_cat_cols].max(axis=1)
        
    # Set none values to G, set R ratings to MA due to low sample size
    ratings = df[md.full_cat_cols + ['movie']].drop_duplicates().drop(columns=['movie']).values
    ratings[ratings == 5] = 4
    ratings[ratings == 0] = 1

    return df, ratings


def perform_baseline_log_reg(df: pd.DataFrame, ratings: npt.ArrayLike, output_dir: str, tfidf_params: Dict[str, str] = None):

    # TODO: experiment with hyperparams (unicode vs ascii)
    # TODO: preprocessing?
    if tfidf_params is None:
        tfidf_params = {'min_df': 0.01, 'max_df': 0.5, 'ngram_range': (1, 3)}

    results_fp = os.path.join(output_dir, f'tfidf-baseline_doc_freq_{tfidf_params["min_df"]}_{tfidf_params["max_df"]}_ngrams_{tfidf_params["ngram_range"]}_films_{df.movie.nunique()}.parquet')

    if os.path.exists(results_fp): return pd.read_parquet(results_fp)

    logging.info('Performing Baseline Logistic Regression Model - No files Found')
    vectorizer = TfidfVectorizer(**tfidf_params, strip_accents='unicode')
    k_fold = StratifiedKFold(n_splits=n_folds)
    log_reg = LogisticRegression(class_weight='balanced', n_jobs=1, max_iter=int(1e4))

    agg_cols = {col: 'first' for col in md.cat_cols + ['rating']}
    agg_cols['text'] = lambda x: ' '.join(x)

    log_reg_results = []

    for rep_type in md.rep_types:

        curr_df = df.copy()
        if rep_type != 'transcript':
            curr_df = curr_df[curr_df.type.eq(rep_type)].reset_index(drop=True)

        curr_df = curr_df.groupby('movie').agg(agg_cols)
        text_arr = curr_df.text.values

        for cat_idx, cat in enumerate(tqdm(md.full_cat_cols)):
            cat_ratings = ratings[:, cat_idx]
            for train_idx, test_idx in k_fold.split(text_arr, cat_ratings):
                X_train, y_train, X_test, y_test = text_arr[train_idx], cat_ratings[train_idx], text_arr[test_idx], cat_ratings[test_idx]
                X_train_vec = vectorizer.fit_transform(X_train)
                X_test_vec = vectorizer.transform(X_test)
                log_reg.fit(X_train_vec, y_train)

                pred_labels = log_reg.predict(X_test_vec)
                log_reg_results.append({
                    'f1_macro': f1_score(y_test, pred_labels, average='macro'),
                    'acc': accuracy_score(y_test, pred_labels),
                    'cat': cat,
                    'rep_type': rep_type
                })

    results_df = pd.DataFrame(log_reg_results)
    results_df.to_parquet(results_fp)

    return results_df


def calc_results(model_name: str, model, X_test, y_true: npt.ArrayLike, row: Dict[str, float], calc_y_prob=False, has_pca=False) -> Dict[str, float]:
    curr_row = {key: val for key, val in row.items()}
    curr_row['classifier'] = model_name 

    y_pred = model.predict(X_test)

    curr_row['pca'] = has_pca
    curr_row['acc'] = accuracy_score(y_true.reshape(-1), y_pred.reshape(-1))
    curr_row['f1_macro'] = f1_score(y_true.reshape(-1), y_pred.reshape(-1), average='macro')

    if calc_y_prob:
        y_prob = model.predict_proba(X_test)
        curr_row['auroc'] = roc_auc_score(y_true.reshape(-1), y_prob, average='macro', multi_class='ovo')
    
    return curr_row


def perform_sem_rep_modelling(df: pd.DataFrame, ratings: npt.ArrayLike, max_iter: int, hypothesis: str, cases: Tuple[str, str, str, str], fast_run: bool, strem_rep_type: bool):

    sem_rem_metrics_fp = f'{md.sem_rep_metrics_fp}{hypothesis.replace("/", "_")}.parquet'
    row_level_fp = f'{md.row_level_classification_fp}{hypothesis.replace("/", "_")}.parquet'
    row_level_df = None

    if os.path.exists(sem_rem_metrics_fp):
        if os.path.exists(row_level_fp):
            row_level_df = pd.read_parquet(row_level_fp)
        return pd.read_parquet(sem_rem_metrics_fp), row_level_df
    
    logging.info(f'{hypothesis.upper()} - Performing Semantic Representation Modelling - No Files found')

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    k_fold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scaler = MinMaxScaler()
    basic_svc_model = LinearSVC(class_weight='balanced', max_iter=max_iter)
    basic_log_reg_model = LogisticRegression(class_weight='balanced', max_iter=max_iter, n_jobs=1)
    pca = PCA(n_components=32 if fast_run else 64)
    results_metrics = []
    row_level_classification_list = []

    # TODO: ablative study on concat pooling

    for pooling_model_name, rep_type, packing_type, pooling_strat in tqdm(cases):

        logging.info(f'{pooling_model_name} {rep_type} {packing_type} {pooling_strat}')
        rep_list, movies_list = esr.get_or_create_movie_sem_reps(df, pooling_model_name, rep_type, packing_type, pooling_strat, device, use_profiler=False)

        if 'emotion' in hypothesis:
            emotion_rep_list, _ = esr.get_or_create_movie_sem_reps(df, hypothesis, rep_type, packing_type, pooling_strat, device, use_profiler=False)
            rep_list = [torch.cat([x.reshape(-1), y.reshape(-1)], dim=0) for x, y in zip(rep_list, emotion_rep_list)]

        # Normalise and stack results
        norm_rep_tensor = torch.nn.functional.normalize(torch.stack([x.reshape(-1) for x in rep_list]))
        X = norm_rep_tensor.cpu().numpy().reshape(len(rep_list), -1)
        y = np.array(ratings)

        for ii, cat in enumerate(md.full_cat_cols):
            # Chunks is faster and better for all categories except nudity
            if strem_rep_type and ((packing_type == 'utterances' and cat != 'nudity') or (packing_type == 'chunks' and cat == 'nudity')):
                continue
            
            y = np.array(ratings)[:, ii]
            for train_index, test_index in k_fold.split(X, y):
                X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]

                curr_data = {
                    'cat': cat, 'model': pooling_model_name, 'rep_type': rep_type,
                    'packing_type': packing_type, 'pooling_strat': pooling_strat
                }

                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                X_train_scaled_pca = pca.fit_transform(X_train_scaled)
                X_test_scaled_pca = pca.transform(X_test_scaled)

                basic_log_reg_model.fit(X_train_scaled_pca, y_train)
                curr_pca_log_data = calc_results('Log Reg', basic_log_reg_model, X_test_scaled_pca, y_test, curr_data, calc_y_prob=True, has_pca=True)
                results_metrics.append(curr_pca_log_data)

                if not fast_run:
                    basic_svc_model.fit(X_train_scaled, y_train)

                    curr_svm_data = calc_results('LSVM', basic_svc_model, X_test_scaled, y_test, curr_data)

                    # Store Log Reg row level probability errors and preds for confusion matrices and further analysis
                    new_classifications_dict = {
                        'movie': [movies_list[ii] for ii in test_index],
                        'model': pooling_model_name,
                        'rep_type': rep_type,
                        'packing_type': packing_type,
                        'pooling_strat': pooling_strat,
                        'cat': cat,
                        'true': y_test.reshape(-1),
                        'pred': basic_log_reg_model.predict(X_test_scaled_pca)
                    }
                    probs = basic_log_reg_model.predict_proba(X_test_scaled_pca)
                    for ii in range(probs.shape[1]):
                        new_classifications_dict[f'prob_{ii}'] = probs[:, ii].reshape(-1)

                    row_level_classification_list.append(new_classifications_dict)
                    results_metrics.append(curr_svm_data)

    results_metrics_df = pd.DataFrame(results_metrics)
    results_metrics_df.to_parquet(sem_rem_metrics_fp)

    if not fast_run:
        row_level_df = pd.concat([pd.DataFrame(x) for x in row_level_classification_list])
        row_level_df.to_parquet(row_level_fp)

    return results_metrics_df, row_level_df


def main():
    df, ratings = get_text_and_ratings()
    tfidf_params = {'min_df': 0.01, 'max_df': 0.5, 'ngram_range': (1, 3)}
    perform_baseline_log_reg(df, ratings, md.results_dir, tfidf_params)

    # Show model performance:
    model_cases = [x for x in esr.get_cases(md.pooling_models, md.rep_types, md.packing_types, md.pooling_strategies[:1])]
    perform_sem_rep_modelling(df, ratings, 1000, hypothesis='models', cases=model_cases, fast_run=True, strem_rep_type=True)

    # Show best performance:
    best_cases = [x for x in esr.get_cases(md.pooling_models[:2], md.rep_types, md.packing_types, md.pooling_strategies[:1])]
    perform_sem_rep_modelling(df, ratings, int(1e5), hypothesis='best', cases=best_cases, fast_run=False, strem_rep_type=False)

    # Combine Emotion
    emotion_models = [md.pooling_models[ii] for ii in [0, 2, 3]]
    emotion_cases = [x for x in esr.get_cases(emotion_models, md.rep_types, md.packing_types, md.pooling_strategies[:1])]
    perform_sem_rep_modelling(df, ratings, int(1e5), hypothesis=md.pooling_models[1], cases=emotion_cases, fast_run=False, strem_rep_type=True)

    # Potential Ablative Studies
    # - PCA vs none
    # - FP32


if __name__ == "__main__":
    main()