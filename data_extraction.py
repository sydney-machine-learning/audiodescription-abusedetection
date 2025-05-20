import pandas as pd

import os
import logging
import re
from typing import List, Tuple

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