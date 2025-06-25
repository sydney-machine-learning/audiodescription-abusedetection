import torch
import time
import logging
from tqdm import tqdm

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(asctime)s - %(message)s')

import pandas as pd

import torch
device = torch.device("cuda")


from transformers import (
    AutoTokenizer,
    AutoModel,
    T5EncoderModel,
    DataCollatorWithPadding
)


import gc
import pickle
import os

import data_extraction as da
import modelling as md

model_name = md.pooling_models[1]
stride = 128
batch_size = 32
enc_max_len = 512 #512 if 'deberta' not in model_name else 1024


def get_chunked_encodings(df: pd.DataFrame, stride: int, tokenizer, max_len: int):
    
    # TODO: convert to utterance level (compare with chunked)
    
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
    
    return all_seg_enc, movie_indices


# Concatenation Methods:
# - BERT Original paper (last 4 layers): "best performing method concatenates the token representations from the top four hidden layers"

def process_with_advanced_optimizations(all_segments, movie_indices, pooling_model, data_collator, device, batch_size=128):
    
    torch.backends.cuda.matmul.allow_tf32 = True
    
    loader = torch.utils.data.DataLoader(
        all_segments,
        batch_size=batch_size,
        collate_fn=data_collator,
        pin_memory=False,
        shuffle=False
    )
    pooling_model = torch.compile(pooling_model, mode="max-autotune")
    all_embeddings = []
    
    start_time = time.time()
    
    with torch.no_grad():
        with torch.amp.autocast(str(device)):
            for batch_idx, batch in enumerate(tqdm(loader, desc="Processing segments")):
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                outputs = pooling_model(**batch) # , output_hidden_states=True
                cls_embeddings = outputs.last_hidden_state[:, 0, :].float()
                all_embeddings.append(cls_embeddings.clone())
                
    all_embeddings = torch.cat(all_embeddings, dim=0)
    
    rep_list = []
    
    # Extract embeddings corresponding to each film, combine and add to list (store on CPU)
    for movie, start_idx, end_idx in movie_indices:
        pooled_layers = all_embeddings[start_idx:end_idx].max(dim=0)[0]
        rep_list.append(pooled_layers.cpu())
    
    # Cleanup GPU memory and artifacts
    del all_embeddings
    torch.cuda.empty_cache()
    gc.collect()
    
    logging.info(f"Total processing time: {time.time() - start_time:.1f}s")
    
    return rep_list


def main():
    
    logging.info(f'Pooling Model: {model_name}')

    df = pd.read_parquet(da.cleaned_dataset_fp).sort_values(['movie', 'start_time'])

    modelClass = AutoModel if 't5' not in model_name else T5EncoderModel
    pooling_model = modelClass.from_pretrained(model_name) #output_hidden_states=True
    pooling_model.to(device)
    pooling_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt", padding=True)
    
    # With transcriptions
    logging.info('Starting extraction of semantic representation with full dataset')
    all_segments, movie_indices = get_chunked_encodings(df, stride, tokenizer, enc_max_len)
    rep_list = process_with_advanced_optimizations(
        all_segments, movie_indices, pooling_model, data_collator, device, batch_size
    )
    
    model_name_fp = model_name.replace('/', '_')
    
    with open(os.path.join(md.all_txt_sem_rep_dir, f'{md.all_txt_pickle_prefix}{model_name_fp}.pkl'), 'wb') as fileobj:
        pickle.dump(rep_list, fileobj)
        
    # Repeat with only dialogue
    logging.info('Repeating extraction with only dialogue')
    df = df[df.type.eq('dialogue')].reset_index(drop=True)
    all_segments, movie_indices = get_chunked_encodings(df, stride, tokenizer, enc_max_len)
    rep_list = process_with_advanced_optimizations(
        all_segments, movie_indices, pooling_model, data_collator, device, batch_size
    )
    
    with open(os.path.join(md.all_txt_sem_rep_dir, f'{md.dialogue_only_pickle_prefix}{model_name_fp}.pkl'), 'wb') as fileobj:
        pickle.dump(rep_list, fileobj)
    
if __name__ == "__main__":
    main()