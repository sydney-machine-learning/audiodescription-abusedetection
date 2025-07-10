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

# TODO: revert space removal for consistency (or apply everywhere?)
import gc
import pickle
import os
from tqdm import tqdm

import data_extraction as da
import modelling as md
import utils

from typing import List


def _extract_sem_rep_for_single_movie(all_segments, pooling_model, pooling_strat, data_collator, device, batch_size=64):
    
    if 'fp32' in pooling_strat:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    else:
        torch.backends.cuda.matmul.allow_tf32 = True

    loader = torch.utils.data.DataLoader(
        all_segments,
        batch_size=batch_size,
        collate_fn=data_collator,
        pin_memory=True,
        shuffle=False
    )
    embeddings_list = []
    
    with torch.no_grad():
        # with torch.autocast('cuda'):
        for ii, batch in enumerate(loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            outputs = pooling_model(**batch, output_hidden_states=True)
            
            # Get the CLS rep and do post processing during prediction
            if pooling_model.name_or_path in md.pooling_models[:-1] and 'CLS' in pooling_strat:
                pooled = torch.stack([outputs.hidden_states[-2][:, 0, :], outputs.hidden_states[-1][:, 0, :]], dim=2)
            else:
                embeddings = outputs.hidden_states[layer_idx] # First element of model_output contains all token embeddings
                mask = batch['attention_mask'].unsqueeze(-1).expand(embeddings.size())

                # Mean pooling
                token_embeddings_mean = embeddings.clone()
                token_embeddings_mean[mask == 0] = 0
                pooled_mean = token_embeddings_mean.sum(dim=1) / mask.sum(dim=1)

                # Max pooling
                token_embeddings_max = embeddings.clone()
                token_embeddings_max[mask == 0] = -1e4
                pooled_max = token_embeddings_max.max(dim=1)[0]
                
                pooled = torch.stack([pooled_mean, pooled_max], dim=1) 

            embeddings_list.append(pooled)

    # Flatten and stack batches
    all_embeddings_arr = torch.cat(embeddings_list, dim=0)

    # TODO: create functions for different pooling strategies
    if pooling_model.name_or_path in md.pooling_models[:-1] and 'CLS' in pooling_strat:
        pooled_layers = all_embeddings_arr.max(dim=0)[0]
        # torch.stack([all_embeddings_arr.mean(dim=0), , dim=0)
    else:
        max_of_max = all_embeddings_arr[:, 0, :].max(dim=0)[0]
        mean_of_mean = all_embeddings_arr[:, 1, :].mean(dim=0)
        pooled_layers = torch.cat([max_of_max, mean_of_mean])
    
    # Cleanup GPU memory and artifacts
    del all_embeddings_arr, embeddings_list, outputs
    
    return pooled_layers


def _get_utterance_encodings(df: pd.DataFrame, tokenizer, max_len: int, label_speech_type: bool):
    
    tokenizer_params = {
        'padding': False,
        'truncation': True,
        'return_tensors': None, #'pt',
        'max_length': max_len
    }

    # TODO: add back experimentation with labelling speech types

    encodings = tokenizer(list(df.text), **tokenizer_params)
    all_segments = [{k: v[ii] for k, v in encodings.items()} for ii in range(df.shape[0])]

    # Count the number of cumulative segments per film and then create the movie segment counts
    # so they can be easily slice out (tokenization only once)
    cum_seg_count = df.groupby('movie').text.count().cumsum().values
    movie_indices = [(movie, start, end) for movie, start, end in zip(df.movie.unique(), [0] + list(cum_seg_count[:-1]), cum_seg_count)]

    return all_segments, movie_indices


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


def clear_sem_reps_for_cat(pooling_model_name, rep_type, packing_type, pooling_strat, n=-1):
    file_group = md.sem_rep_filename.format(movie='', model=pooling_model_name.replace('/', '_'), rep_type=rep_type, packing_type=packing_type, pooling_strat=pooling_strat)
    files_to_clear = [os.path.join(md.sem_rep_dir, x) for x in os.listdir(md.sem_rep_dir) if x.endswith(file_group)]

    # Optionally only delete a certain amount (for profiling)
    if n > 0:
        files_to_clear = files_to_clear[:n]

    if len(files_to_clear) == 0:
        logging.warning('No files removed in semantic representation file clear')

    for path in files_to_clear:
        os.remove(path)


def _get_profile_table(key_avgs, col):

    rows = []
    for evt in key_avgs:
        rows.append({
            'name': evt.key,
            'cpu_time': evt.cpu_time_total,
            'device_time': evt.device_time_total,

        })
    df = pd.DataFrame(rows)

    df['cpu_time_s'] = df['cpu_time'] / 1e6
    df['cuda_time_s'] = df['device_time'] / 1e6
    df = df.drop(columns=['cpu_time', 'device_time'])

    return df.sort_values(by=col, ascending=False).head(10).reset_index(drop=True)


def _accumulate_profile_results(prof, cpu_df_list, gpu_df_list):
    prof.stop()
    key_avgs = prof.key_averages()
    cpu_df = _get_profile_table(key_avgs, 'cpu_time_s')
    gpu_df = _get_profile_table(key_avgs, 'cuda_time_s')

    cpu_df_list.append(cpu_df)
    gpu_df_list.append(gpu_df)

    return cpu_df_list, gpu_df_list


def get_or_create_movie_sem_reps(df, pooling_model_name, rep_type, packing_type, pooling_strat, device, use_profiler=False):
    
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
    no_pooling_layer = md.pooling_models[3:6]
    # model_kwargs = {} if pooling_model_name in no_pooling_layer else {'add_pooling_layer': False}
    pooling_model = modelClass.from_pretrained(pooling_model_name) #**model_kwargs

    if not 'fp32' in pooling_strat:
        pooling_model.half()
    pooling_model.to(device)
    pooling_model.eval()
    # padding_args = {'pad_to_multiple_of': 16} if pooling_model_name in md.pooling_models[:-1] else {'padding': 'max_length'}
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt", padding=True) # **padding_args
    
    batch_size = 32 if packing_type == 'chunks' else 64
    # batch_size = 64 if rep_type == 'dialogue' else 32
    
    if len(missing_movies) > 0:
        if rep_type != 'transcript':
            filtered_df = filtered_df[filtered_df.type.eq(rep_type)]
            
        if packing_type == 'chunks':
            all_enc, movie_indices = _get_chunked_encodings(filtered_df, stride, tokenizer, enc_max_len)

        else:
            # TODO: Consider removing narrator aggregation if minimal performance impact occurs
            filtered_df = _agg_narrator_seg(filtered_df).sort_values(['movie', 'start_time'])
            all_enc, movie_indices = _get_utterance_encodings(filtered_df, tokenizer, enc_max_len, label_speech_type=False)

        if use_profiler:
            gpu_df_list = []
            cpu_df_list = []
            prof = torch.profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False, with_stack=False)
            prof.start()
    
    for ii, movie in enumerate(tqdm(missing_movies)):
        
        movie_fp = os.path.join(md.sem_rep_dir, f'{movie}{file_group}')
        curr_idx = movie_indices[ii]
        if curr_idx[0] != movie:
            raise ValueError('Movie indices are out of sync')
        segments = all_enc[curr_idx[1]:curr_idx[2]]
            
        # Now save semantic representation based on segments
        # Profile first 20 movies
        sem_rep = _extract_sem_rep_for_single_movie(segments, pooling_model, pooling_strat, data_collator, device, batch_size)

        with open(movie_fp, 'wb') as fileobj:
            pickle.dump(sem_rep, fileobj)

        if ii % 10 == 9 or 'fp32' in pooling_strat:
            if use_profiler:
                cpu_df_list, gpu_df_list = _accumulate_profile_results(prof, cpu_df_list, gpu_df_list)
                prof = torch.profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False, with_stack=False)
                prof.start()

            torch.cuda.empty_cache()
            # torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()
            gc.collect()


    if len(missing_movies) > 0 and use_profiler:
        if prof._stats().number_of_events > 0:
            prof.stop()
            cpu_df_list, gpu_df_list = _accumulate_profile_results(prof, cpu_df_list, gpu_df_list)
        cpu_df = pd.concat(cpu_df_list).groupby('name').sum().reset_index().sort_values('cpu_time_s', ascending=False)
        gpu_df = pd.concat(gpu_df_list).groupby('name').sum().reset_index().sort_values('gpu_time_s', ascending=False)
        cpu_df.to_csv(os.path.join('logs', f'{utils.remove_ext(file_group)}_cpu.csv'))
        gpu_df.to_csv(os.path.join('logs', f'{utils.remove_ext(file_group)}_gpu.csv'))

    rep_list = []
    
    sorted_films = os.listdir(md.sem_rep_dir)
    sorted_films.sort()
    for movie_filename in sorted_films:
        if movie_filename.endswith(file_group):
            try:
                with open(os.path.join(md.sem_rep_dir, movie_filename), 'rb') as fileobj:
                    rep_list.append(pickle.load(fileobj))
            except:
                logging.info(f'Failed to open pickle of {movie_filename}')
            
    return rep_list


def get_cases(models: List[str], rep_types: List[str], packing_types: List[str], pooling_strats: List[str]):
    
    cases = []
    # Iterate through all pooling models and chunking styles
    for pooling_model_name in models:
        for rep_type in rep_types:
            for packing_type in packing_types:
                for pooling_strat in pooling_strats: 
                    cases.append((pooling_model_name, rep_type, packing_type, pooling_strat))
                                 
    return cases

    
def main(models: List[str], rep_types: List[str], packing_types: List[str], pooling_strats: List[str]):

    torch.cuda.empty_cache()
    # torch.cuda.ipc_collect()
    gc.collect()

    # Get all transcript data
    df = pd.read_parquet(da.cleaned_dataset_fp).sort_values(['movie', 'start_time'])

    use_profiler = False
    
    # Iterate through all pooling models and chunking styles
    cases = get_cases(models, rep_types, packing_types, pooling_strats)
    for pooling_model_name, rep_type, packing_type, pooling_strat in cases:
        # clear_sem_reps_for_cat(pooling_model_name, rep_type, packing_type, pooling_strat, n=20)
        logging.info(f'Curr Model: {pooling_model_name}, {rep_type}, {packing_type}, {pooling_strat}')
        get_or_create_movie_sem_reps(df, pooling_model_name, rep_type, packing_type, pooling_strat, device, use_profiler=use_profiler)

        # TODO: ensure logging directory exists
        torch.cuda.empty_cache()
        # torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
                
                
if __name__ == "__main__":
    models = [
        'cardiffnlp/twitter-roberta-large-sensitive-multilabel',
        'cardiffnlp/twitter-roberta-base-sentiment-latest',
        'mrm8488/t5-base-finetuned-imdb-sentiment'
    ]
    main(md.pooling_models[:1], ['dialogue'], ['chunks'], [md.pooling_strategies[-1]])
    # main(md.pooling_models, md.rep_types, md.packing_types, md.pooling_strategies)