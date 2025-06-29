import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(asctime)s - %(message)s')

import pandas as pd
import numpy as np

import torch
device = torch.device("cuda")

from transformers import (
    AutoTokenizer,
    AutoModel,
    T5EncoderModel,
    DataCollatorWithPadding
)

from transformers.trainer_pt_utils import LengthGroupedSampler

from sklearn.metrics import RocCurveDisplay, accuracy_score, ConfusionMatrixDisplay, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split

# TODO: reinstate mord ordinal regression for minor performance improvement
import gc
import pickle
import os
import time
from tqdm import tqdm

import data_extraction as da
import modelling as md

seed = 42


def get_chunked_encodings(df: pd.DataFrame, stride: int, tokenizer, max_len: int):
    
    all_segments = []
    movie_indices = []
    overall_idx = 0

    for movie in df['movie'].unique():
        curr_df = df[df.movie.eq(movie)]
        full_text_token_ids = tokenizer(' '.join(curr_df['text']), add_special_tokens=False)
        full_text_tokens = tokenizer.batch_decode(full_text_token_ids['input_ids'])

        token_count = 0
        total_tokens = len(full_text_token_ids['input_ids'])
        start = 0

        while token_count < total_tokens:
            
            # Use small offset to ensure limits aren't exceeded
            end = min(len(full_text_tokens), start + tokenizer.model_max_length - 2)
            all_segments.append(''.join(full_text_tokens[start:end]))
            start = end - stride
            token_count = end
            
        movie_indices.append((movie, overall_idx, len(all_segments)))
        overall_idx = len(all_segments)

    all_seg_enc = [tokenizer(t, truncation=True, return_tensors=None, max_length=max_len) for t in all_segments]
    for ii, enc in enumerate(all_seg_enc):
        enc['idx'] = ii
    
    return all_seg_enc, movie_indices


def _agg_narrator_seg(df: pd.DataFrame):
    
    df = df.sort_values('start_time')
    
    agg_rows_list = []
    prev_row = dict(df.iloc[0])

    for row in df.iloc[1:].to_dict(orient="records"):
        
        if prev_row['type'] == 'narrator' and row['type'] == 'narrator' and prev_row['movie'] == row['movie']:
            prev_row['end_time'] = row['end_time']
            prev_row['text'] += (' ' + row['text'])
            
        else:
            agg_rows_list.append(prev_row)
            prev_row = row.copy()
        
    agg_seg_df = pd.DataFrame.from_records(agg_rows_list)

    return agg_seg_df 


def get_utterance_encodings(df: pd.DataFrame, tokenizer, max_len: int, label_speech_type: bool):
    
    all_segments = []
    movie_indices = []
    overall_idx = 0
    
    tokenizer_params = {
        'padding': False,
        'truncation': True,
        'return_tensors': None, #'pt',
        'max_length': max_len
    }

    for movie in df['movie'].unique():
        curr_df = _agg_narrator_seg(df[df.movie.eq(movie)])
        txt_list = list(curr_df.text)
        
        if label_speech_type:
            type_annot = list('Type: ' + curr_df.type)
            encodings = tokenizer(type_annot, txt_list, **tokenizer_params)
            
        else:
            encodings = tokenizer(txt_list, **tokenizer_params)
            
        for ii in range(len(txt_list)):
            sample = {k: v[ii] for k, v in encodings.items()}
            sample['idx'] = overall_idx + ii
            all_segments.append(sample)

        movie_indices.append((movie, overall_idx, len(all_segments)))
        overall_idx = len(all_segments)

    return all_segments, movie_indices


def process_with_advanced_optimizations(all_segments, movie_indices, pooling_model, data_collator, device, batch_size=128):
    
    torch.backends.cuda.matmul.allow_tf32 = True

    dataset = md.PlaceHolderDataset(all_segments)

    sampler = LengthGroupedSampler(
        dataset=dataset,
        batch_size=batch_size,
        lengths=[len(s['input_ids']) for s in all_segments]
    )
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        sampler=sampler,
        pin_memory=False,
        shuffle=False
    )
    pooling_model = torch.compile(pooling_model, mode="max-autotune")
    all_embeddings_list = []
    
    start_time = time.time()
    
    with torch.no_grad():
        with torch.amp.autocast(str(device)):
            for batch_idx, batch in enumerate(tqdm(loader, desc="Processing segments")):
                ordering_idx = batch['idx'].cpu()
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k != 'idx'}
                outputs = pooling_model(**batch, output_hidden_states=True)
                
                if pooling_model in md.pooling_models[:-1]:
                    # Get the CLS rep from the second last hidden layer since last one may be too finetuned on classification task
                    embeddings = outputs.hidden_states[-2][:, 0, :].float()
                else:
                    embeddings = outputs[0] # First element of model_output contains all token embeddings
                    input_mask_expanded = batch['attention_mask'].unsqueeze(-1).expand(embeddings.size()).float()
                    embeddings[input_mask_expanded == 0] = -1e9
                    
                for emb, idx in zip(embeddings, ordering_idx):
                    all_embeddings_list.append((idx.item(), emb.cpu()))
    del outputs
                
    all_embeddings_list.sort(key=lambda x: x[0])
    all_embeddings_arr = torch.stack([emb for idx, emb in all_embeddings_list], dim=0)
    del all_embeddings_list
    
    rep_list = []
    
    # Extract embeddings corresponding to each film, combine and add to list (store on CPU)
    for movie, start_idx, end_idx in movie_indices:
        curr_embedding_layers = all_embeddings_arr[start_idx:end_idx]
        # Get both max and mean pooling
        pooled_layers = torch.stack([curr_embedding_layers.max(dim=0)[0], curr_embedding_layers.mean(dim=0)], dim=-1)
        rep_list.append(pooled_layers)
    
    # Cleanup GPU memory and artifacts
    del all_embeddings_arr
    torch.cuda.empty_cache()
    gc.collect()
    
    logging.info(f"Total processing time: {time.time() - start_time:.1f}s")
    
    return rep_list


def build_sem_rep_for_single_model(model_name: str, stride: int, batch_size: int, use_utterances: bool):
    
    enc_max_len = 512 #512 if 'deberta' not in model_name else 1024
    
    df = pd.read_parquet(da.cleaned_dataset_fp).sort_values(['movie', 'start_time'])

    modelClass = AutoModel if 't5' not in model_name else T5EncoderModel
    pooling_model = modelClass.from_pretrained(model_name) #output_hidden_states=True
    pooling_model.to(device)
    pooling_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt", pad_to_multiple_of=16)
    
    # With transcriptions
    logging.info('Starting extraction of semantic representation with full dataset')
    if use_utterances:
        all_segments, movie_indices = get_utterance_encodings(df, tokenizer, enc_max_len, label_speech_type=False)
    else:    
        all_segments, movie_indices = get_chunked_encodings(df, stride, tokenizer, enc_max_len)
        
    rep_list = process_with_advanced_optimizations(
        all_segments, movie_indices, pooling_model, data_collator, device, batch_size
    )
    
    model_name_fp = model_name.replace('/', '_')
    rep_type = 'utterances' if use_utterances else 'chunks'
    with open(os.path.join(md.all_txt_sem_rep_dir, f'{md.all_txt_pickle_prefix}{model_name_fp}_{rep_type}.pkl'), 'wb') as fileobj:
        pickle.dump(rep_list, fileobj)
        
    # Repeat with only dialogue
    logging.info('Repeating extraction with only dialogue')
    df = df[df.type.eq('dialogue')].reset_index(drop=True)
    if use_utterances:
        all_segments, movie_indices = get_utterance_encodings(df, tokenizer, enc_max_len, label_speech_type=False)
    else:    
        all_segments, movie_indices = get_chunked_encodings(df, stride, tokenizer, enc_max_len)
        
    rep_list = process_with_advanced_optimizations(
        all_segments, movie_indices, pooling_model, data_collator, device, batch_size
    )
    
    with open(os.path.join(md.all_txt_sem_rep_dir, f'{md.dialogue_only_pickle_prefix}{model_name_fp}_{rep_type}.pkl'), 'wb') as fileobj:
        pickle.dump(rep_list, fileobj)
    
    
def get_dataset_split(model_name: str, only_dialogue: bool, rep_type: str, ratings, test_size: float):
    prefix = md.dialogue_only_pickle_prefix if only_dialogue else md.all_txt_pickle_prefix
    rep_pkl_fp = os.path.join(md.all_txt_sem_rep_dir, f'{prefix}{model_name.replace("/", "_")}_{rep_type}.pkl')
    batch_size = 64 if 'deberta' not in model_name else 16
    
    if not os.path.exists(rep_pkl_fp):
        logging.info(f'Building semantic representation pickle for {rep_pkl_fp}')
        build_sem_rep_for_single_model(model_name, stride=128, batch_size=batch_size, use_utterances=rep_type=='utterances')
        
    with open(rep_pkl_fp, 'rb') as fileobj:
        rep_list = pickle.load(fileobj)
        
    rep_list = [x.reshape(-1) for x in rep_list]
        
    return train_test_split(torch.stack(rep_list).numpy(), np.array(ratings), test_size=test_size, random_state=seed)
    
    
def main():
    
    test_size = 0.1
    
    # Get all transcript data
    df = pd.read_parquet(da.cleaned_dataset_fp).drop(columns=['nudity', 'language']).sort_values(['movie', 'start_time'])

    for col in md.cat_cols:
        df[col] = md.convert_col_to_ordinal(df[col])
        
    ratings = df[md.cat_cols + ['movie']].drop_duplicates().drop(columns=['movie']).values
    
    results = []

    # Iterate through all pooling models and chunking styles
    for pooling_model_name in md.pooling_models:
        logging.info(f'Curr Model: {pooling_model_name}')
        for rep_type in ('utterances', 'chunks'):
            logging.info(f'Semantic Representation Grouping strategy: {rep_type}')
            for only_dialogue in [False, True]:
                X_train, X_test, y_train, y_test = get_dataset_split(pooling_model_name, only_dialogue, rep_type, ratings, test_size)

                basic_log_model = MultiOutputClassifier(LogisticRegression(max_iter=100000))
                basic_log_model.fit(X_train, y_train)
                preds = basic_log_model.predict(X_test)
                
                logging.info(f'Dialogue Only {only_dialogue}, Acc: {accuracy_score(y_test.reshape(-1), preds.reshape(-1),) * 100}%')
                results.append({'model': pooling_model_name, 'rep_type': rep_type, 'only_dialogue': only_dialogue, 'preds': preds})
    
if __name__ == "__main__":
    main()