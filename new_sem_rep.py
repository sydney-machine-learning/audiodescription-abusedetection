import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(asctime)s - %(message)s')

import pandas as pd
import numpy as np

import torch
device = torch.device("cuda")

import transformers

from transformers import (
    AutoTokenizer,
    AutoModel,
    T5EncoderModel,
    DataCollatorWithPadding
)

# TODO: reinstate mord ordinal regression for minor performance improvement
import gc
import pickle
import os
from tqdm import tqdm
import time

import data_extraction as da
import modelling as md

seed = 42

def _extract_sem_rep_for_single_movie(all_segments, pooling_model, pooling_strat, data_collator, device, batch_size=64):
    
    torch.backends.cuda.matmul.allow_tf32 = True

    # dataset = md.PlaceHolderDataset(all_segments)
    loader = torch.utils.data.DataLoader(
        all_segments,
        batch_size=batch_size,
        collate_fn=data_collator,
        pin_memory=False,
        shuffle=False
    )
    # TODO: consider removing as autotuning every movie might slow it down
    # pooling_model = torch.compile(pooling_model, mode="max-autotune")
    embeddings_list = []
    
    with torch.no_grad():
        # with torch.autocast('cuda'):
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            outputs = pooling_model(**batch, output_hidden_states=True)
            
            # TODO: create functions for different pooling sizes
            if pooling_model.name_or_path in md.pooling_models[:-1]:
                # Get the CLS rep from the second last hidden layer since last one may be too finetuned on classification task
                embeddings = outputs.hidden_states[-2][:, 0, :].float()
            else:
                embeddings = outputs[0] # First element of model_output contains all token embeddings
                input_mask_expanded = batch['attention_mask'].unsqueeze(-1).expand(embeddings.size()).float()
                embeddings[input_mask_expanded == 0] = -1e9
                
            embeddings_list.append(embeddings)

    del outputs
    
    # Flatten and stack batches
    all_embeddings_arr = torch.stack([y for x in embeddings_list for y in x], dim=0)
    del embeddings_list
    
    # TODO: create functions for different pooling strategies
    pooled_layers = torch.stack([all_embeddings_arr.max(dim=0)[0], all_embeddings_arr.mean(dim=0)], dim=-1)
    
    # Cleanup GPU memory and artifacts
    del all_embeddings_arr
    torch.cuda.empty_cache()
    gc.collect()
    
    return pooled_layers


def _get_utterance_encodings(df: pd.DataFrame, tokenizer, max_len: int, label_speech_type: bool):
    
    tokenizer_params = {
        'padding': False,
        'truncation': True,
        'return_tensors': None, #'pt',
        'max_length': max_len
    }

    if label_speech_type:
        encodings = tokenizer(list('Type: ' + df.type), list(df.text), **tokenizer_params)
    else:
        encodings = tokenizer(list(df.text), **tokenizer_params)
        
    all_segments = [{k: v[ii] for k, v in encodings.items()} for ii in range(df.shape[0])]

    return all_segments

    # start_time = time.time()
    # logging.info(f"Total processing time: {time.time() - start_time:.1f}s")
    

def _old_agg_narrator_seg(df: pd.DataFrame):
    
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


def _agg_narrator_seg(df: pd.DataFrame):
    df = df.sort_values('start_time').reset_index(drop=True)
    
    # Vectorized approach
    narrator_mask = df['type'] == 'narrator'
    movie_change = df['movie'] != df['movie'].shift(1)
    type_change = df['type'] != df['type'].shift(1)
    
    # Group consecutive narrator segments
    group_ids = ((narrator_mask & type_change) | movie_change).cumsum()
    
    agg_df = df.groupby(group_ids).agg({
        'movie': 'first',
        'type': 'first', 
        'start_time': 'first',
        'end_time': 'last',
        'text': lambda x: ' '.join(x)
    }).reset_index(drop=True)
    
    return agg_df


def _old_get_chunked_encodings(df: pd.DataFrame, stride: int, tokenizer, max_len: int):
    
    all_segments = []

    # Briefly silence warnings as we intentionally want to tokenize more than the model context
    transformers.logging.set_verbosity_error()
    full_text_token_ids = tokenizer(' '.join(df['text']), add_special_tokens=False, truncation=False)
    transformers.logging.set_verbosity_warning()
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
            
    all_seg_enc = [tokenizer(t, truncation=True, return_tensors=None, max_length=max_len) for t in all_segments]
    
    return all_seg_enc


def _get_chunked_encodings(df: pd.DataFrame, stride: int, tokenizer, max_len: int):
    
    all_segments = []
    movie_indices = []
    overall_idx = 0
    
    # Briefly silence warnings as we intentionally want to tokenize more than the model context
    transformers.logging.set_verbosity_error()
    
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

    transformers.logging.set_verbosity_warning()
    all_seg_enc = [tokenizer(t, truncation=True, return_tensors=None, max_length=max_len) for t in all_segments]
    
    return all_seg_enc, movie_indices

    
def get_or_create_movie_sem_reps(df, pooling_model_name, rep_type, packing_type, pooling_strat, device, batch_size=64):
    
    enc_max_len = 512 #512 if 'deberta' not in model_name else 1024
    stride = 128
    
    # Identify which movies we need to build representations for
    file_group = md.sem_rep_filename.format(movie='', model=pooling_model_name.replace('/', '_'), rep_type=rep_type, packing_type=packing_type, pooling_strat=pooling_strat)
    stored_movies = [x.split('_')[0] for x in os.listdir(md.sem_rep_dir) if x.endswith(file_group)]
    filtered_df = df[~df.movie.isin(stored_movies)]
    missing_movies = filtered_df.movie.unique()
    
    # Initialise models, tokenizers and similar artifacts
    tokenizer = AutoTokenizer.from_pretrained(pooling_model_name)
    modelClass = AutoModel if 't5' not in pooling_model_name else T5EncoderModel
    pooling_model = modelClass.from_pretrained(pooling_model_name, add_pooling_layer=False)
    pooling_model.half()
    pooling_model.to(device)
    pooling_model.eval()
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt", pad_to_multiple_of=16)
    
    batch_size = 32 if packing_type == 'chunks' else 64
    
    if rep_type == 'dialogue':
        filtered_df = filtered_df[filtered_df.type.eq('dialogue')]
        
    if packing_type == 'chunks':
        all_enc, movie_indices = _get_chunked_encodings(filtered_df, stride, tokenizer, enc_max_len)
        logging.info('Finished chunking encodings')
    else:
        filtered_df = _agg_narrator_seg(filtered_df)
    
    for ii, movie in enumerate(tqdm(missing_movies)):
        
        movie_fp = os.path.join(md.sem_rep_dir, f'{movie}{file_group}')
        curr_df = filtered_df[filtered_df.movie.eq(movie)]
        
        if packing_type == 'utterances':
            segments = _get_utterance_encodings(curr_df, tokenizer, enc_max_len, label_speech_type=False)
        else:
            curr_idx = movie_indices[ii]
            if curr_idx[0] != movie:
                raise ValueError('Movie indices are out of sync')
            segments = all_enc[curr_idx[1]:curr_idx[2]]
            
        sem_rep = _extract_sem_rep_for_single_movie(segments, pooling_model, pooling_strat, data_collator, device, batch_size)
        
        with open(movie_fp, 'wb') as fileobj:
            pickle.dump(sem_rep, fileobj)
    
    rep_list = []
    
    for movie_fp in [os.path.join(md.sem_rep_dir, x) for x in os.listdir(md.sem_rep_dir) if x.endswith(file_group)]:
        with open(movie_fp, 'rb') as fileobj:
            rep_list.append(pickle.load(fileobj))
            
    return rep_list
        
    
def main():
    
    # Get all transcript data
    df = pd.read_parquet(da.cleaned_dataset_fp).sort_values(['movie', 'start_time'])

    for col in md.cat_cols:
        df[col] = md.convert_col_to_ordinal(df[col])
        
    batch_size = 64
    
    # Iterate through all pooling models and chunking styles
    for pooling_model_name in md.pooling_models:
        logging.info(f'Curr Model: {pooling_model_name}')
        for rep_type in ['dialogue', 'transcript']:
            logging.info(f'Text Content: {rep_type}')
            for packing_type in ('utterances', 'chunks'):
                logging.info(f'Packing Type: {packing_type}')
                for pooling_strat in ['lhs2 CLS']:
                    logging.info(f'Pooling Strategy: {pooling_strat}')
                    get_or_create_movie_sem_reps(df, pooling_model_name, rep_type, packing_type, pooling_strat, device, batch_size=batch_size)
                
                
if __name__ == "__main__":
    main()