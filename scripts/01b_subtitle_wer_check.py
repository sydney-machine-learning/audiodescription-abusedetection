import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(asctime)s - %(message)s')

import pandas as pd

import os
import re
import sys

from evaluate import load

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import audiodescription_abusedetection.data_extraction as da
import audiodescription_abusedetection.stt as stt
import audiodescription_abusedetection.utils as utils

import warnings
warnings.filterwarnings("ignore")

sub_wer_scores_df = pd.DataFrame(columns=['movie', 'cer', 'wer'])
if os.path.exists(da.sub_wer_scores_fp):
    sub_wer_scores_df = pd.read_parquet(da.sub_wer_scores_fp).drop_duplicates(['movie'])

movie_names = [utils.remove_ext(x) for x in da.get_sorted_mp3_list(rerun_all=True)]
name_conv_dict = {key: val for key, val in zip(da.clean_movie_name_series(pd.Series(movie_names)), movie_names)}
df = pd.read_parquet(da.cleaned_dataset_fp)
sub_df = df[df.type.eq('dialogue')].copy().reset_index(drop=True).copy()
filtered_sub_df = sub_df[~sub_df.movie.isin(sub_wer_scores_df.movie.unique())]

cer, wer = load('cer'), load('wer')
results = []
word_counts = []

movie_list = list(filtered_sub_df.movie.unique())

# Reverse movie list if negative commandline argument is passed in (so independent scripts can be ran)
if len(sys.argv) > 1 and int(sys.argv[1]) < 0:
    movie_list = movie_list[::-1]

for ii, cleaned_movie_name in enumerate(movie_list):
    logging.info(f'{ii} / {filtered_sub_df.movie.nunique()}')
    curr_sub_df = sub_df[sub_df.movie.eq(cleaned_movie_name)]
    if curr_sub_df.shape[0] == 0: raise ValueError('Subtitle Dataframe is empty')

    sub_txt = ' '.join(curr_sub_df.text.str.replace('[\.,"\?!♪♪<>]', '', regex=True)).lower().replace('-', ' ')

    # Work out original file name
    movie_name = name_conv_dict[cleaned_movie_name]
    mp3_filename = movie_name + '.mp3'
    vad_df_path = os.path.join(da.voice_activity_dir, f'{movie_name}-vad.parquet')
    seg_df_path = os.path.join(da.diarization_dir, f'{movie_name}-diarization.parquet')
    wav_filepath = os.path.join(da.trans_mp3_dir, f'{movie_name}_speech_only.wav')

    # Reapply VAD to get shortened WAV file
    stt.apply_silero_vad_to_wav(mp3_filename, wav_filepath, vad_df_path, silero_threshold=stt.silero_threshold)
    curr_stt_df = stt.transcribe_segments(os.path.join('data', 'interim', 'temp.parquet'), seg_df_path, wav_filepath, stt.whisper_model, stt.whisper_config, stt.narr_cosine_sim_lim, device, convert_dialogue=True)
    curr_stt_df = curr_stt_df[curr_stt_df['text'].ne(' Thank you.')]

    cleaned_sub_series = curr_sub_df.text.str.replace('[\.,"\?!]', '', regex=True) \
        .str.replace('([\[\(]/?[\w\s]+[\]\)])|(\w+:\s)', '', regex=True) \
        .str.replace('-', ' ') \
        .str.replace('[‘’]', "'", regex=True) \
        .str.replace('[\.“”;…♪\?]', '', regex=True) \
        .str.replace('\s+', ' ', regex=True).str.strip()
    sub_txt = ' '.join(cleaned_sub_series[cleaned_sub_series.ne('')]).lower()
    sub_txt = re.sub('\s+', ' ', sub_txt)

    trans_txt = ''.join(curr_stt_df.text.str.replace('[\.,"\?!]', '', regex=True)).lower().replace('-', ' ')

    if len(sub_txt) == 0 or len(trans_txt) == 0:
        raise ValueError('Scripts are too short')

    wer_score = wer.compute(predictions=[trans_txt], references=[sub_txt])
    cer_score = cer.compute(predictions=[trans_txt], references=[sub_txt])
    new_row = [{'cer': cer_score, 'wer': wer_score, 'movie': cleaned_movie_name}]

    os.remove(wav_filepath)

    # Add new row and save progress (unperformant but means script can be interrupted)
    sub_wer_scores_df = pd.concat([sub_wer_scores_df, pd.DataFrame(new_row)], axis=0).reset_index(drop=True)
    sub_wer_scores_df.to_parquet(da.sub_wer_scores_fp)