import pandas as pd
import numpy as np

import os
import logging
import re
from typing import List, Tuple

from bs4 import BeautifulSoup

from transformers import (
    AutoTokenizer
)

from evaluate import load

import kaggle
from nltk import download
download('stopwords')
download('punkt_tab')

import utils


kaggle_subtitles_slug = 'mlopssss/subtitles'

sub_dir = os.path.join('data', 'subtitles')
sub_by_year_dir = os.path.join(sub_dir, 'Subtitlesforoscarandblockbusters')
sub_df_dir = os.path.join('data', 'subtitles', 'all_subtitles.parquet')
add_sub_dir = os.path.join(sub_dir, 'additional_subtitles')

audio_vault_dir = os.path.join('data', 'audio-vault')

voice_activity_dir = os.path.join(audio_vault_dir, 'voice_activity')

transcription_dir = os.path.join(audio_vault_dir, 'transcriptions')
transcript_df_fp = '{movie_name}-transcript.parquet'
all_transcripts_df_dir = os.path.join(transcription_dir, 'all_transcripts.parquet')

diarization_dir = os.path.join(audio_vault_dir, 'diarization_segments')
trans_mp3_dir = os.path.join(audio_vault_dir, 'longitudinal_movies')

credits_ts_fp = os.path.join(transcription_dir, 'manual', 'credit_removal_timestamps.csv')

acb_ratings_fp = os.path.join('data', 'acb_film_ratings.csv')

cleaned_dataset_fp = os.path.join('data', 'cleaned_dataset.parquet')

time_sep = '-->'
sample_rate = 16000

os.makedirs(sub_dir, exist_ok=True)
os.makedirs(diarization_dir, exist_ok=True)
os.makedirs(trans_mp3_dir, exist_ok=True)
os.makedirs(transcription_dir, exist_ok=True)
os.makedirs(voice_activity_dir, exist_ok=True)


def clean_movie_name_series(series: pd.Series) -> pd.Series:
    return series.str.strip().str.replace('-', ' ').str.replace('.', '').str.title()


def get_transcript_list():
    return [x for x in os.listdir(transcription_dir) if x.endswith(transcript_df_fp.format(movie_name=''))]


def get_sorted_mp3_list(rerun_all: bool = False):
    
    # First identify movies with no transcripts, these should be done first
    mp3_files = [x for x in os.listdir(trans_mp3_dir) if os.path.splitext(x)[-1].lower() == '.mp3']
    all_movies = [utils.remove_ext(x) for x in mp3_files]
    transcripts = [x.removesuffix(transcript_df_fp.format(movie_name='')) for x in get_transcript_list()]
    filtered_transcripts = [x for x in transcripts if x in all_movies]

    # Add all missing transcripts to outputlist
    output_list = [x + '.mp3' for x in set(all_movies).difference(set(transcripts))]
    
    if rerun_all:
        # Next sort all the transcriptions by last modified time and add them
        file_times = [(movie + '.mp3', os.stat(os.path.join(transcription_dir, transcript_df_fp.format(movie_name=movie))).st_mtime) for movie in filtered_transcripts]
        remaining_mp3s = [x for x, _ in sorted(file_times, key=lambda x: x[1])]
        output_list += remaining_mp3s

    return output_list


def get_or_create_subtitles_data(parquet_path: str, download_dir: str):
    
    if os.path.exists(parquet_path) and parquet_path.endswith('.parquet'):
        movie_list_df = pd.read_parquet(parquet_path)
    
    else:
        kaggle.api.authenticate()
        os.makedirs(download_dir, exist_ok=True)
        kaggle.api.dataset_download_files(kaggle_subtitles_slug, path=download_dir, unzip=True)
        
        corrupted_set = set(['Grease.srt', 'X-Men.srt', 'Mr. Mrs. Smith.srt', 'The Hangover Part II.srt', 'Finding Neverland.srt', 'The Social Network.srt'])

        for fp in os.walk(os.path.join(sub_dir, 'Subtitlesforoscarandblockbusters')):
            overlap = set(fp[2]).intersection(corrupted_set)
            if len(overlap) > 0:
                for file in overlap:
                    logging.info(f'Deleting {file}')
                    os.remove(os.path.join(fp[0], file))
                    
        # Add 01 to line 3877 of The Super Mario Bros. Movie 2023

        with open(os.path.join(download_dir, 'titles with awards and categories.txt')) as file_obj:
            movie_list_lines = file_obj.readlines()

        df_datatypes = {
            'movie': 'str', 'year': 'int', 'fame_category': 'category', 'genre': 'category'
        }

        movie_list_df =  pd.DataFrame([re.match(r'([^\()]*)\s\((\d{4}),\s(\w*)\),\s(\w*)', x).groups() for x in movie_list_lines],
                                    columns = ['movie', 'year', 'fame_category', 'genre']).astype(df_datatypes)
        
        movie_list_df['movie'] = movie_list_df['movie'].str.strip().str.title()
        
        movie_list_df.to_parquet(parquet_path)

    return movie_list_df


def remove_html(text):
    if pd.isna(text):
        return text
    return BeautifulSoup(text, 'html.parser').get_text(separator=' ')


def get_credits_timestamps():
    return pd.read_csv(credits_ts_fp)


def get_acb_film_ratings():
    return pd.read_csv(acb_ratings_fp)


def clean_dialogue(dialogue: pd.Series) -> pd.Series:
    return dialogue.str.lower().str.replace('.', '').str.replace('"', '').str.replace(',', '').str.replace('-', '')


def wipe_movie_files(movie_name: str):
    
    removed_nothing = True
    if os.path.exists(os.path.join(transcription_dir, transcript_df_fp.format(movie_name=movie_name))):
        os.remove(os.path.join(transcription_dir, transcript_df_fp.format(movie_name=movie_name)))
        removed_nothing = False
    if os.path.exists(os.path.join(diarization_dir, f'{movie_name}-diarization.parquet')):
        os.remove(os.path.join(diarization_dir, f'{movie_name}-diarization.parquet'))
        removed_nothing = False
    if os.path.exists(os.path.join(voice_activity_dir, f'{movie_name}-vad.parquet')):
        os.remove(os.path.join(voice_activity_dir, f'{movie_name}-vad.parquet'))
        removed_nothing = False
        
    if removed_nothing:
        logging.info(f'Removed no artifacts for {movie_name}')
    else:
        logging.info(f'Success')
        

def _fix_specific_file(old_fp: str, new_fp: str, new_name: str):
    curr_df = pd.read_parquet(old_fp)
    curr_df['movie_name'] = new_name
    curr_df.to_parquet(new_fp)
    os.remove(old_fp)
    
        
def rename_movie_files(movie_name: str, new_name: str):
    
    rename_nothing = True
    old_mp3_fp = os.path.join(trans_mp3_dir, movie_name + '.mp3')
    old_trans_fp = os.path.join(transcription_dir, transcript_df_fp.format(movie_name=movie_name))
    old_diaz_fp = os.path.join(diarization_dir, f'{movie_name}-diarization.parquet')
    old_vad_fp = os.path.join(voice_activity_dir, f'{movie_name}-vad.parquet')
    
    if os.path.exists(old_mp3_fp):
        os.rename(old_mp3_fp, os.path.join(trans_mp3_dir, new_name + '.mp3'))
        rename_nothing = False
    
    if os.path.exists(old_trans_fp):
        new_trans_fp = os.path.join(transcription_dir, transcript_df_fp.format(movie_name=new_name))
        _fix_specific_file(old_trans_fp, new_trans_fp, new_name)
        rename_nothing = False
        
    if os.path.exists(old_diaz_fp):
        new_diaz_fp = os.path.join(diarization_dir, f'{new_name}-diarization.parquet')
        _fix_specific_file(old_diaz_fp, new_diaz_fp, new_name)
        rename_nothing = False
        
    if os.path.exists(old_vad_fp):
        new_vad_fp = os.path.join(voice_activity_dir, f'{new_name}-vad.parquet')
        _fix_specific_file(old_vad_fp, new_vad_fp, new_name)
        rename_nothing = False
        
    if rename_nothing:
        logging.info(f'Renamed no artifacts for {movie_name}')
    else:
        logging.info(f'Success')


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


def check_existing_subs():
    films_list_df = get_or_create_subtitles_data(os.path.join(sub_dir, 'movie_index.parquet'), sub_dir)
    test = [x.removesuffix('.srt') for x in os.listdir(os.path.join(sub_dir, 'additional_subtitles'))]
    return any(x in films_list_df.movie.values for x in test)