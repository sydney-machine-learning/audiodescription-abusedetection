import pandas as pd
import numpy as np

import os
import logging
import re
from typing import List, Tuple

from transformers import (
    AutoTokenizer
)

import kaggle
from nltk import download
download('stopwords')

import utils


kaggle_subtitles_slug = 'mlopssss/subtitles'

sub_dir = os.path.join('data', 'subtitles')
sub_by_year_dir = os.path.join(sub_dir, 'Subtitlesforoscarandblockbusters')
sub_df_dir = os.path.join('data', 'subtitles', 'all_subtitles.parquet')

audio_vault_dir = os.path.join('data', 'audio-vault')

voice_activity_dir = os.path.join(audio_vault_dir, 'voice_activity')

transcription_dir = os.path.join(audio_vault_dir, 'transcriptions')
transcript_df_fp = '{movie_name}-transcript.parquet'
all_transcripts_df_dir = os.path.join(transcription_dir, 'all_transcripts.parquet')

diarization_dir = os.path.join(audio_vault_dir, 'diarization_segments')
trans_mp3_dir = os.path.join(audio_vault_dir, 'longitudinal_movies')

credits_ts_fp = os.path.join(transcription_dir, 'manual', 'credit_removal_timestamps.csv')

time_sep = '-->'
sample_rate = 16000

os.makedirs(sub_dir, exist_ok=True)
os.makedirs(diarization_dir, exist_ok=True)
os.makedirs(trans_mp3_dir, exist_ok=True)
os.makedirs(transcription_dir, exist_ok=True)
os.makedirs(voice_activity_dir, exist_ok=True)


def get_or_create_subtitles_data(parquet_path: str, download_dir: str):
    
    if os.path.exists(parquet_path) and parquet_path.endswith('.parquet'):
        movie_list_df = pd.read_parquet(parquet_path)
    
    else:
        kaggle.api.authenticate()
        os.makedirs(download_dir, exist_ok=True)
        kaggle.api.dataset_download_files(kaggle_subtitles_slug, path=download_dir, unzip=True)

        with open(os.path.join(download_dir, 'titles with awards and categories.txt')) as file_obj:
            movie_list_lines = file_obj.readlines()

        df_datatypes = {
            'movie': 'str', 'year': 'int', 'fame_category': 'category', 'genre': 'category'
        }

        movie_list_df =  pd.DataFrame([re.match(r'([^\()]*)\s\((\d{4}),\s(\w*)\),\s(\w*)', x).groups() for x in movie_list_lines],
                                    columns = ['movie', 'year', 'fame_category', 'genre']).astype(df_datatypes)
        
        movie_list_df.to_parquet(parquet_path)

    return movie_list_df


def get_credits_timestamps():
    return pd.read_csv(credits_ts_fp)


def clean_dialogue(dialogue: pd.Series) -> pd.Series:
    return dialogue.str.lower().str.replace('.', '').str.replace('"', '').str.replace(',', '').str.replace('-', '')


def wipe_movie_files(movie_name: str):
    if os.path.exists(os.path.join(transcription_dir, transcript_df_fp.format(movie_name=movie_name))):
        os.remove(os.path.join(transcription_dir, transcript_df_fp.format(movie_name=movie_name)))
    if os.path.exists(os.path.join(diarization_dir, f'{movie_name}-diarization.parquet')):
        os.remove(os.path.join(diarization_dir, f'{movie_name}-diarization.parquet'))
    if os.path.exists(os.path.join(voice_activity_dir, f'{movie_name}-vad.parquet')):
        os.remove(os.path.join(voice_activity_dir, f'{movie_name}-vad.parquet'))


def convert_time_to_readable_txt(df: pd.DataFrame, cols: List[Tuple[str, str]]):
    
    for old_col, new_col in cols:
        hours = (df[old_col] // 3600).astype(int).astype(str)
        mins = ((df[old_col] // 60) % 60).astype(int).astype(str)
        secs = (df[old_col] % 60).astype(int).astype(str)
        df[new_col] = hours + 'hr ' + mins + 'min ' + secs + 's'
    
    return df


def extract_single_subs_file(filepath: str):
    
    with open(filepath, 'r', encoding='utf-8-sig') as fileobj:
        file_lines = fileobj.readlines()
        
    raw_timestamps = []
    raw_dialogue = []
    counter = -1
    
    # Separate lines of dialogue from timestamps into raw lists
    for raw_line in file_lines:
        line = raw_line.strip()
        
        if re.match('^\d{1,}$', line):
            continue
        elif time_sep in line:
            counter += 1
            raw_dialogue.append('')
            raw_timestamps.append(line)
        elif line != '':
            raw_dialogue[counter] += (' ' + line)
    
    # Create dataframe # TODO: remove downloaded tag at end
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
    
    movie_name = utils.remove_ext(os.path.basename(filepath))
    subs_df['movie'] = movie_name
        
    return subs_df


def aggregate_segments(df: pd.DataFrame):
    
    df = df.sort_values('start')
    
    agg_rows_list = []
    prev_row = dict(df.iloc[0])

    for row in df.iloc[1:].to_dict(orient="records"):
        
        if prev_row['speaker'] == row['speaker']:
            prev_row['end'] = row['end']
            
        else:
            agg_rows_list.append(prev_row)
            prev_row = row.copy()
        
    agg_seg_df = pd.DataFrame.from_records(agg_rows_list)

    return agg_seg_df 
