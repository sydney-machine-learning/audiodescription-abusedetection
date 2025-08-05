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
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_sample_weight

from scipy import stats

from tqdm import tqdm

import utils
import data_extraction as da
import extract_sem_representation as esr

import modelling as md

import os

from typing import Tuple, Dict, List
import numpy.typing as npt

n_folds = 5
seed = 42

# jobs to 1 for debugging and -1 for performance
n_jobs = -1


def get_text_and_ratings(compact: bool = True) -> Tuple[pd.DataFrame, npt.ArrayLike]:
    df = pd.read_parquet(da.cleaned_dataset_fp).sort_values(['movie', 'start_time'])

    for col in md.full_cat_cols:
        df[col] = md.convert_col_to_ordinal(df[col], compact=compact)

    df['rating'] = df[md.full_cat_cols].max(axis=1)
        
    ratings = df[md.full_cat_cols + ['movie']].drop_duplicates().drop(columns=['movie']).values

    if not compact:
        # None category doesn't correspond to classification, so remove it
        ratings[ratings == 0] = 1

    return df, ratings


def get_text_and_classifications() -> Tuple[pd.DataFrame, npt.ArrayLike]:
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

    if tfidf_params is None:
        tfidf_params = {'min_df': 0.01, 'max_df': 0.9, 'ngram_range': (1, 3), 'strip_accents': 'ascii'}

    results_fp = os.path.join(output_dir, f'tfidf-baseline_doc_freq_{tfidf_params["min_df"]}_{tfidf_params["max_df"]}_ngrams_{tfidf_params["ngram_range"]}_films_{df.movie.nunique()}.parquet')

    if os.path.exists(results_fp): return pd.read_parquet(results_fp)

    logging.info('Performing Baseline Logistic Regression Model - No files Found')
    vectorizer = TfidfVectorizer(**tfidf_params)
    k_fold = StratifiedKFold(n_splits=n_folds)
    log_reg = LogisticRegression(class_weight='balanced', n_jobs=n_jobs, max_iter=int(1e4))

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
                    'f1_macro': f1_score(y_test, pred_labels, average='macro') * 100,
                    'acc': accuracy_score(y_test, pred_labels) * 100,
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
    curr_row['acc'] = accuracy_score(y_true.reshape(-1), y_pred.reshape(-1)) * 100
    curr_row['f1_macro'] = f1_score(y_true.reshape(-1), y_pred.reshape(-1), average='macro') * 100
    curr_row['f1_weighted'] = f1_score(y_true.reshape(-1), y_pred.reshape(-1), average='weighted') * 100

    if calc_y_prob:
        y_prob = model.predict_proba(X_test)
        curr_row['auroc'] = roc_auc_score(y_true.reshape(-1), y_prob, average='macro', multi_class='ovo')
    
    return curr_row


def filter_CLS_rep(rep_list, concat_pooling: str):
    incl_idx = []

    # Depressing how ugly this is
    if 'max' in concat_pooling and '1' in concat_pooling:
        incl_idx.append(0)
    if 'mean' in concat_pooling and '1' in concat_pooling:
        incl_idx.append(1)
    if 'max' in concat_pooling and '2' in concat_pooling:
        incl_idx.append(2)
    if 'mean' in concat_pooling and '2' in concat_pooling:
        incl_idx.append(3)

    return [x[incl_idx, :] for x in rep_list]


def get_mode_filter_indices(y_train: npt.ArrayLike) -> npt.ArrayLike:
    mode, count = stats.mode(y_train)
    sec_mode, sec_count = stats.mode(y_train[y_train != mode])

    # First identify the indices associated with the mode
    mode_indices = np.arange(y_train.shape[0])[y_train == mode]
    non_mode_indices = np.arange(y_train.shape[0])[y_train != mode]

    # Then only choose as many observations as corresponding to the second mode, without replacement
    red_mode_indices = np.random.choice(mode_indices, sec_count, replace=False)

    indices_excl_excess_mode = list(set(non_mode_indices).union(red_mode_indices))

    return indices_excl_excess_mode


def perform_sem_rep_modelling(df: pd.DataFrame, ratings: npt.ArrayLike, max_iter: int, hypothesis: str, cases: Tuple[str, str, str, str], fast_run: bool, concat_pooling=None, pca_params=None, emote_cases: Dict[str, bool] = None, cats_lists: List[List[str]] = None):

    sem_rem_metrics_fp = f'{md.sem_rep_metrics_fp}{hypothesis}.parquet'
    row_level_fp = f'{md.row_level_classification_fp}{hypothesis}.parquet'
    row_level_df = None

    if os.path.exists(sem_rem_metrics_fp):
        if os.path.exists(row_level_fp):
            row_level_df = pd.read_parquet(row_level_fp)
        return pd.read_parquet(sem_rem_metrics_fp), row_level_df
    
    logging.info(f'{hypothesis.upper()} - Performing Semantic Representation Modelling - No Files found')

    k_fold = StratifiedKFold(n_splits=n_folds, shuffle=True if 'variance' in hypothesis else False) # , shuffle=True, random_state=seed
    scaler = StandardScaler()
    svc_model = LinearSVC(class_weight='balanced', max_iter=max_iter)

    basic_model = LogisticRegression(class_weight='balanced', max_iter=max_iter, n_jobs=n_jobs)

    movie_list = list(df.movie.unique())

    if pca_params is None:
        pca_params = {'n_components': 32}
    use_pca = pca_params['n_components'] != 'original'

    pca = PCA(**pca_params)

    if emote_cases is None:
        emote_cases = {cat: False for cat in md.full_cat_cols}

    results_metrics = []
    row_level_classification_list = []

    for ii, (pooling_model_name, rep_type, packing_type, pooling_strat) in enumerate(tqdm(cases)):

        logging.info(f'{pooling_model_name} {rep_type} {packing_type} {pooling_strat}')
        rep_list = esr.get_or_create_movie_sem_reps(df, pooling_model_name, rep_type, packing_type, pooling_strat, device, use_profiler=False)

        if concat_pooling is not None:
            rep_list = filter_CLS_rep(rep_list, concat_pooling)

        # Stack results, shift to numpy and reshape
        no_emote_X = torch.stack([x.reshape(-1) for x in rep_list]).cpu().numpy().reshape(len(rep_list), -1)

        if 'emotion' in hypothesis:
            emotion_model = next(x for x in md.pooling_models if 'emotion' in x)
            emotion_rep_list = esr.get_or_create_movie_sem_reps(df, emotion_model, rep_type, packing_type, pooling_strat, device, use_profiler=False)

            if concat_pooling is not None:
                emotion_rep_list = filter_CLS_rep(emotion_rep_list, concat_pooling)
            rep_list = [torch.cat([x.reshape(-1), y.reshape(-1)], dim=0) for x, y in zip(rep_list, emotion_rep_list)]
            emote_X = torch.stack([x.reshape(-1) for x in rep_list]).cpu().numpy().reshape(len(rep_list), -1)

        case_cats = md.full_cat_cols if cats_lists is None else cats_lists[ii]

        for cat in case_cats:
            cat_idx = md.full_cat_cols.index(cat)

            # Check whether category benefits from emotion or not
            X = emote_X.copy() if emote_cases[cat] else no_emote_X.copy()
            
            y = np.array(ratings)[:, cat_idx] if 'overall' not in hypothesis else np.array(ratings).max(axis=1)
            for train_index, test_index in k_fold.split(X, y):
                X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]

                if 'reduction' in hypothesis:
                    mode_filter_indices = get_mode_filter_indices(y_train)
                    X_train = X_train[mode_filter_indices, :]
                    y_train = y_train[mode_filter_indices]

                curr_data = {
                    'cat': cat, 'model': pooling_model_name, 'rep_type': rep_type, 'packing_type': packing_type,
                    'pooling_strat': pooling_strat, 'hypothesis': hypothesis
                }

                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                if use_pca:
                    X_train_scaled_pca = pca.fit_transform(X_train_scaled)
                    X_test_scaled_pca = pca.transform(X_test_scaled)
                else:
                    X_train_scaled_pca = X_train_scaled
                    X_test_scaled_pca = X_test_scaled

                basic_model.fit(X_train_scaled_pca, y_train)
                curr_pca_log_data = calc_results('Log Reg', basic_model, X_test_scaled_pca, y_test, curr_data, calc_y_prob=True, has_pca=use_pca)
                results_metrics.append(curr_pca_log_data)

                if not fast_run:
                    svc_model.fit(X_train_scaled, y_train)

                    curr_svm_data = calc_results('LSVM', svc_model, X_test_scaled, y_test, curr_data)

                    # Store Log Reg row level probability errors and preds for confusion matrices and further analysis
                    new_classifications_dict = {
                        'movie': [movie_list[kk] for kk in test_index],
                        'model': pooling_model_name,
                        'rep_type': rep_type,
                        'packing_type': packing_type,
                        'pooling_strat': pooling_strat,
                        'cat': cat,
                        'true': y_test.reshape(-1),
                        'pred': basic_model.predict(X_test_scaled_pca)
                    }
                    probs = basic_model.predict_proba(X_test_scaled_pca)
                    for kk in range(probs.shape[1]):
                        new_classifications_dict[f'prob_{kk}'] = probs[:, kk].reshape(-1)

                    row_level_classification_list.append(new_classifications_dict)
                    results_metrics.append(curr_svm_data)

    results_metrics_df = pd.DataFrame(results_metrics)
    results_metrics_df.to_parquet(sem_rem_metrics_fp)

    if not fast_run:
        row_level_df = pd.concat([pd.DataFrame(x) for x in row_level_classification_list])
        row_level_df.to_parquet(row_level_fp)

    return results_metrics_df, row_level_df


def main():

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    df, ratings = get_text_and_ratings()
    tfidf_params = {'min_df': 0.01, 'max_df': 0.9, 'ngram_range': (1, 3), 'strip_accents': 'ascii'}
    perform_baseline_log_reg(df, ratings, md.results_dir, tfidf_params)


    ### Reduction of Class Imbalance (Undersampling)
    # Try reduce most frequent class to frequency of second most frequent class in training set only
    # Balancing class weights in both the classifier and the Macro F1 metric means this has no impact
    # red_cases = [x for x in esr.get_cases(md.pooling_models[:2], md.rep_types, md.packing_types, md.pooling_strategies[:1])]
    # perform_sem_rep_modelling(df, ratings, int(1e5), hypothesis='reduction', cases=red_cases, fast_run=False)


    ### All Models Performance
    model_cases = [x for x in esr.get_cases(md.pooling_models, md.rep_types, md.packing_types, md.pooling_strategies[:2])]
    perform_sem_rep_modelling(df, ratings, int(1e5), hypothesis='models', cases=model_cases, fast_run=True)

    wide_cases = [x for x in esr.get_cases(md.pooling_models[-1:], md.rep_types, ['chunks'], md.pooling_strategies[-1:])]
    perform_sem_rep_modelling(df, ratings, int(1e5), hypothesis='wide', cases=wide_cases, fast_run=True)

    top_models = md.pooling_models[:3]
    first_pass_best_cases_df = md.agg_and_sort_cv_results('models', final_groupby='model', agg_groupby=['model', 'rep_type', 'packing_type', 'pooling_strat'], top_n=1)
    first_pass_best_cases = list(first_pass_best_cases_df[['model', 'rep_type', 'packing_type', 'pooling_strat']].itertuples(index=False, name=None))


    ### PCA testing
    pca_df_list = []
    for setting in [32, 64, 0.8, 0.95, 'original', None]:
        # All -> no PCA, None -> use all components in PCA, confusing but this is partly how the n_components setting works
        curr_df = perform_sem_rep_modelling(df, ratings, int(1e5), hypothesis=f'pca-{str(setting)}', cases=first_pass_best_cases, fast_run=True, pca_params={'n_components': setting})[0]
        curr_df['pca_setting'] = str(setting)
        pca_df_list.append(curr_df)

    pca_df = pd.concat(pca_df_list) \
        .groupby(['model', 'cat', 'rep_type', 'packing_type', 'pca_setting', 'pooling_strat']) \
        .agg({'f1_macro': 'mean', 'acc': 'mean'}) \
        .reset_index() 

    pca_df.to_parquet(f'{md.sem_rep_metrics_fp}pca.parquet')


    ### Ablative study on CLS Layers and min vs max concat pooling
    first_best_cases_best_models = [x for x in first_pass_best_cases if x[0] in md.pooling_models[:3]]

    perform_sem_rep_modelling(df, ratings, int(1e5), hypothesis='CLS-1', cases=first_best_cases_best_models, fast_run=True, concat_pooling='1-mean-max')
    perform_sem_rep_modelling(df, ratings, int(1e5), hypothesis='CLS-2', cases=first_best_cases_best_models, fast_run=True, concat_pooling='2-mean-max')
    perform_sem_rep_modelling(df, ratings, int(1e5), hypothesis='CLS-Mean', cases=first_best_cases_best_models, fast_run=True, concat_pooling='mean-1-2')
    perform_sem_rep_modelling(df, ratings, int(1e5), hypothesis='CLS-Max', cases=first_best_cases_best_models, fast_run=True, concat_pooling='max-1-2')


    ### Emotion:
    all_cases = [x for x in esr.get_cases(top_models, md.rep_types, md.packing_types, md.pooling_strategies[:2])]
    perform_sem_rep_modelling(df, ratings, int(1e5), hypothesis='emotion', cases=all_cases, fast_run=True, pca_params={'n_components': None})

    groupby_cols = ['model', 'cat', 'rep_type', 'packing_type', 'pooling_strat', 'classifier']
    best_cases_df = md.agg_and_sort_cv_results('models', final_groupby=['cat', 'rep_type'], agg_groupby=groupby_cols, top_n=3)
    best_cases_top_3 = list(best_cases_df[['model', 'rep_type', 'packing_type', 'pooling_strat']].itertuples(index=False, name=None))

    best_df_list = []
    for ii in range(5):
        best_df_list.append(perform_sem_rep_modelling(df, ratings, int(1e5), hypothesis=f'best-{ii}', cases=best_cases_top_3, fast_run=False, cats_lists=[[x] for x in best_cases_df.cat.values])[0])

    best_df = pd.concat(best_df_list)
    best_df.to_parquet(md.sem_rep_metrics_fp + 'best.parquet')


    ### Variance Testing
    groupby_cols = ['model', 'cat', 'rep_type', 'packing_type', 'pooling_strat', 'classifier']

    best_cases_df = md.agg_and_sort_cv_results('best', final_groupby='cat', agg_groupby=groupby_cols, top_n=1)
    best_cases = list(best_cases_df[['model', 'rep_type', 'packing_type', 'pooling_strat']].itertuples(index=False, name=None))
    cats_lists = [[x] for x in best_cases_df.cat.values]

    for ii in range(30):
        perform_sem_rep_modelling(df, ratings, int(1e5), hypothesis=f'variance-{ii}', cases=best_cases, fast_run=True, cats_lists=cats_lists)

    
    # Overall Rating Prediction
    # df, ratings = get_text_and_ratings(compact=False)
    overall_dialoge_cases = esr.get_cases(md.pooling_models[:1], ['dialogue'], md.packing_types, md.pooling_strategies[:1])
    perform_sem_rep_modelling(df, ratings, int(1e5), hypothesis='overall-rating', cases=[(md.pooling_models[0], 'transcript', 'chunks', 'no-stride')], fast_run=True)
    perform_sem_rep_modelling(df, ratings, int(1e5), hypothesis='overall-rating-dialogue', cases=overall_dialoge_cases, fast_run=True)


if __name__ == "__main__":
    main()