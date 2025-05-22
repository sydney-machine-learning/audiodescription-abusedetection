import pandas as pd
import numpy as np

import os
import logging
import re
from typing import List, Tuple

from transformers import (
    AutoTokenizer
)

import utils


sub_dir = os.path.join('data', 'subtitles')
time_sep = '-->'


def clean_dialogue(dialogue: pd.Series) -> pd.Series:
    return dialogue.str.lower().str.replace('.', '').str.replace('"', '').str.replace(',', '').str.replace('-', '')


def convert_time_to_readable_txt(df: pd.DataFrame, cols: List[Tuple[str, str]]):
    
    for old_col, new_col in cols:
        hours = (df[old_col] // 3600).astype(int).astype(str)
        mins = ((df[old_col] // 60) % 60).astype(int).astype(str)
        secs = (df[old_col] % 60).astype(int).astype(str)
        df[new_col] = hours + 'hr ' + mins + 'min ' + secs + 's'
    
    return df


def extract_subs():
    
    subs_file_list = [os.path.join(sub_dir, x) for x in os.listdir(sub_dir) if not x.endswith('.zip')]
    movie_subs_dict = {}

    for sub_file in subs_file_list:
        with open(sub_file, 'r', encoding='utf-8-sig') as fileobj:
            raw_timestamps = []
            raw_dialogue = []
            counter = -1
            
            # Separate lines of dialogue from timestamps into raw lists
            for raw_line in fileobj.readlines():
                line = raw_line.strip()
                
                if re.match('^\d{1,}$', line):
                    counter += 1
                    raw_dialogue.append('')
                elif time_sep in line:
                    raw_timestamps.append(line)
                elif line != '':
                    raw_dialogue[counter] += (' ' + line)
        
        # Create dataframe
        if 'subtitle' in raw_dialogue[0].lower():
            raw_dialogue = raw_dialogue[1:]
            raw_timestamps = raw_timestamps[1:]
        
        subs_df = pd.DataFrame({'raw_dialogue': raw_dialogue, 'raw_time_str': raw_timestamps})
        
        # Replace commas with decimals to ensure conversion goes smoothly
        subs_df['raw_time_str'] = subs_df['raw_time_str'].str.replace(',', '.')
        
        # Remove new line characters on ends of dialogue (current method still keeps 'internal' new line characters)
        subs_df['raw_dialogue'] = subs_df['raw_dialogue'].str.strip().str.replace('</i>|<i>', '', regex=True)
        
        # Split start and end times apart using separator and store in two separate columns
        subs_df[['raw_start_time', 'raw_end_time']] = subs_df['raw_time_str'].str.split(time_sep, n=1, expand=True)
        
        # Convert str -> timedelta
        subs_df['start_time'] = pd.to_timedelta(subs_df['raw_start_time'])
        subs_df['end_time'] = pd.to_timedelta(subs_df['raw_end_time'])
        
        movie_name = utils.remove_ext(os.path.basename(sub_file))
        movie_subs_dict[movie_name] = subs_df
        
    return movie_subs_dict


def get_segments(path: str, movies: List[str], file_format: str, tokenizer) -> pd.DataFrame:
    
    segment_fps = [x for x in os.listdir(path) if x.endswith(file_format.split('-')[-1])]
    
    # Optional filtering
    if len(movies) > 0:
        segment_fps = [file_format.format(movie_name=movie) for movie in movies if file_format.format(movie_name=movie) in segment_fps]
    
    df_list = []
    for movie in segment_fps:
        curr_df = pd.read_parquet(os.path.join(path, movie))
        curr_df['movie'] = movie.split('-')[0]
        df_list.append(curr_df)
        
    seg_df = pd.concat(df_list)

    # Whisper adds whitespace which affects tokens
    seg_df['cleaned_text'] = clean_dialogue(seg_df['text'].str.strip())
    seg_df['cleaned_tokens'] = seg_df['cleaned_text'].apply(lambda x: tokenizer.encode(x))

    # TODO: add this earlier in data cleaning process
    seg_df['start'] = seg_df['start'].round(1)
    seg_df['end'] = seg_df['end'].round(1)

    seg_df = seg_df.drop(columns=['movie_name', 'start_frame', 'end_frame', 'tokens', 'cleaned_tokens', 'cleaned_text'])
    
    return seg_df 


def get_vsd_movie_annotations(path: str, movies: List[str]) -> pd.DataFrame:
    
    df_list = []
    annotation_fps = os.listdir(path)
    
    movies = movies if len(movies) > 0 else set(x.split('_')[0] for x in os.listdir(path))

    for movie in movies:
        for annot_filepath in [x for x in annotation_fps if movie in x]:
            cat = utils.remove_ext(annot_filepath.split('_')[1])
            raw_annot_df = extract_vsd_annotations_file(os.path.join(path, annot_filepath))
            raw_annot_df['movie'] = movie
            raw_annot_df['annotation_cat'] = cat
            df_list.append(raw_annot_df)
            
    return pd.concat(df_list)
    

def extract_vsd_annotations_file(path):
    
    with open(path, 'r') as fileobj:
        raw_lines = [x.strip() for x in fileobj.readlines() if x != '\n']
        
    df_rows = []
    for line in raw_lines:
        vals = line.split(' ', maxsplit=2)
        row = {'start': float(vals[0]), 'finish': float(vals[1])}
        
        if len(vals) > 2:
            row['desc'] = vals[2]
        df_rows.append(row)
        
    return pd.DataFrame(df_rows)


def aggregate_segments(df: pd.DataFrame, tokenizer, enc_len: int, max_seg_duration: int, max_break: int, excl_dialogue: bool):
    
    df['num_tokens'] = [len(x) for x in tokenizer(list(df['text']))['input_ids']]
    df = df.sort_values(['movie', 'start'])
    
    agg_rows_list = []
    prev_row = dict(df.iloc[0])

    for row in df.iloc[1:].to_dict(orient="records"):
        
        is_under_token_limit = (prev_row['num_tokens'] + row['num_tokens']) < enc_len
        is_under_duration_limit = (row['end'] - prev_row['start']) < max_seg_duration
        is_under_max_break = (row['start'] - prev_row['end']) < max_break
        is_same_speaker = prev_row['movie'] == row['movie'] and prev_row['speaker'] == row['speaker'] 
        
        if all((is_same_speaker, is_under_max_break, is_under_duration_limit, is_under_token_limit)):
            prev_row['end_txt'] = row['end_txt']
            prev_row['duration'] += row['duration']
            prev_row['text'] += (' ' + row['text'])
            prev_row['end'] = row['end']
            prev_row['num_tokens'] += row['num_tokens']
            
        else:
            agg_rows_list.append(prev_row)
            prev_row = row.copy()
        
    agg_seg_df = pd.DataFrame.from_records(agg_rows_list).sort_values(['movie', 'start']).reset_index(drop=True)

    if excl_dialogue:
        agg_seg_df = agg_seg_df[agg_seg_df.is_dialogue.eq(False)].reset_index(drop=True)
        
    return agg_seg_df


def annotate_segments(agg_seg_df: pd.DataFrame, gore_df: pd.DataFrame, tokenizer):
    
    # Mark each segment as containing gore by looping through all the gore annotations
    agg_seg_df['has_gore'] = False
    agg_seg_df['has_blood'] = False
    agg_seg_df['blood_cat'] = pd.NA

    agg_seg_df['gore_start'] = -1 
    agg_seg_df['gore_end'] = -1 

    agg_seg_df['blood_start'] = -1 
    agg_seg_df['blood_end'] = -1 

    seg_df_list = []

    for movie in gore_df.movie.unique():
        
        curr_gore_df = gore_df[gore_df.movie.eq(movie)].copy()
        curr_seg_df = agg_seg_df[agg_seg_df.movie.eq(movie)].copy()
        
        for ii in range(curr_gore_df.shape[0]):
            
            cat = 'gore' if 'gore' in curr_gore_df['annotation_cat'].iloc[ii] else 'blood'
            
            # Identify any segments that overlap with gore segment
            anot_start_after_seg = curr_seg_df.start > curr_gore_df['end_sec'].iloc[ii]
            seg_end_before_anot = curr_seg_df.end < curr_gore_df['start_sec'].iloc[ii]
            curr_anot_mask = np.logical_not(anot_start_after_seg | seg_end_before_anot)
            
            if sum(curr_anot_mask) > 0:
                curr_seg_df[f'{cat}_start'] = np.where(curr_anot_mask, curr_gore_df['start_sec'].iloc[ii], curr_seg_df[f'{cat}_start'])
                curr_seg_df[f'{cat}_end'] = np.where(curr_anot_mask, curr_gore_df['end_sec'].iloc[ii], curr_seg_df[f'{cat}_end'])
                
                curr_seg_df[f'has_{cat}'] = curr_seg_df[f'has_{cat}'] | curr_anot_mask
                
                if cat == 'blood':
                    curr_seg_df['blood_cat'] = np.where(curr_anot_mask, curr_gore_df['full_annotation_cat'].iloc[ii], curr_seg_df['blood_cat'])
                    
            # No overlap, so find closest segment   
            else:
                annot_midpoint = (curr_gore_df['start_sec'].iloc[ii] + curr_gore_df['end_sec'].iloc[ii]) / 2
                closest_start_idx = next(iter(curr_seg_df.start[curr_seg_df.start > annot_midpoint].index), None)
                closest_start = abs(curr_seg_df.start[closest_start_idx] - annot_midpoint) if closest_start_idx else curr_seg_df.end.iloc[-1]
                closest_end_idx = max(0, closest_start_idx - 1) if closest_start_idx else curr_seg_df.index[-1]
                closest_end = abs(curr_seg_df.end[closest_end_idx] - annot_midpoint) if closest_end_idx else curr_seg_df.end.iloc[-1]
                
                closest_seg_idx = closest_end_idx
                
                if closest_start is not None and closest_end is not None and closest_start < closest_end:
                    closest_seg_idx = closest_start_idx
                    
                curr_seg_df.loc[closest_seg_idx, f'{cat}_start'] = curr_gore_df['start_sec'].iloc[ii].astype(int)
                curr_seg_df.loc[closest_seg_idx, f'{cat}_end'] = curr_gore_df['end_sec'].iloc[ii].astype(int)
                
                curr_seg_df.loc[closest_seg_idx, f'has_{cat}'] = True
                
                if cat == 'blood':
                    curr_seg_df.loc[closest_seg_idx, 'blood_cat'] = curr_gore_df['full_annotation_cat'].iloc[ii]
                    
        seg_df_list.append(curr_seg_df)
        
    cat_seg_df = pd.concat(seg_df_list).reset_index(drop=True)
    
    cat_seg_df['cat'] = np.select(
        [cat_seg_df['has_gore'] | cat_seg_df['blood_cat'].isin(['blood-high', 'blood-medium']), cat_seg_df.has_blood],
        ['gore', 'blood'],
        default='neutral'
    )
    
    cat_seg_df['tokens'] = tokenizer(list(cat_seg_df['text']), add_special_tokens=False)['input_ids']

    return cat_seg_df


def add_context_and_tokenize(cat_seg_df: pd.DataFrame, tokenizer, enc_len: int):
    
    tokens_w_context = []

    for ii in range(cat_seg_df.shape[0]):
        context_len = (enc_len - cat_seg_df['num_tokens'][ii]) // 2
        prev_seg = tokenizer.decode(cat_seg_df['tokens'][ii - 1][-context_len:]) if ii > 0 else ''
        next_seg = tokenizer.decode(cat_seg_df['tokens'][ii + 1][:context_len]) if ii < cat_seg_df.shape[0] - 1 else ''
        tokens_w_context.append('[CLS]' + prev_seg + '[SEP]' + tokenizer.decode(cat_seg_df['tokens'][ii]) + '[SEP]' + next_seg)
        
    return tokens_w_context