import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(asctime)s - %(message)s')

import pandas as pd
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import audiodescription_abusedetection.data_extraction as da
from audiodescription_abusedetection import stt, utils

import sys
import argparse
import os
import gc
import warnings
warnings.filterwarnings("ignore")

logging.getLogger('pyannote.audio').setLevel(logging.ERROR)
logging.getLogger('speechbrain').setLevel(logging.ERROR)
logging.getLogger('pytorch').setLevel(logging.ERROR)
logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)


# RUN PARAMS
parser = argparse.ArgumentParser(description='STT Pipeline')
parser.add_argument('--overwrite', action='store_true', help='Flag to enable overwriting transcription artifacts based on oldest creation dates')
args = parser.parse_args()

logging.info(args.overwrite)

# Torch (pyannote) isn't familiar with MP3 files, so convert to wav for effective performance
# Perform diarization to help separate narration in audio description from dialogue in original movie
# Finally use OpenAI's Whisper to convert to a transcript

mp3_files = da.get_sorted_mp3_list(rerun_all=args.overwrite)

for ii, mp3_filename in enumerate(mp3_files):
    
    movie_name = utils.remove_ext(mp3_filename)
    
    vad_df_path = os.path.join(da.voice_activity_dir, f'{movie_name}-vad.parquet')
    seg_df_path = os.path.join(da.diarization_dir, f'{movie_name}-diarization.parquet')
    curr_transcript_fp = os.path.join(da.transcript_dir, da.transcript_df_fp.format(movie_name=movie_name))
    wav_filepath = os.path.join(da.trans_mp3_dir, f'{movie_name}_speech_only.wav')

    # If either diarization or transcript is missing, we'll need to generate the wav file
    if not os.path.exists(curr_transcript_fp) or not os.path.exists(seg_df_path) or args.overwrite:
        logging.info(f'{ii} \t/ {len(mp3_files)} \t {movie_name}')
        stt.apply_silero_vad_to_wav(mp3_filename, wav_filepath, vad_df_path, stt.silero_threshold)
            
    # Only perform diarization if parquet doesn't exist
    if not os.path.exists(seg_df_path) or args.overwrite:
        stt.apply_diarization(movie_name, wav_filepath, stt.pyannote_model_name, seg_df_path, vad_df_path, device)

    # Only assess cosine similarity if it is missing
    seg_df = pd.read_parquet(seg_df_path)
    if not 'cosine_sim' in seg_df.columns:
        stt.add_pyannote_cosine_sim(seg_df_path, wav_filepath, min_seg_sec=stt.min_seg_sec, device=device)

    # Only perform transcription if parquet doesn't exist
    if not os.path.exists(curr_transcript_fp) or args.overwrite:
        stt.transcribe_segments(curr_transcript_fp, seg_df_path, wav_filepath, stt.whisper_model, stt.whisper_config, stt.narr_cosine_sim_lim, device)
        
    # Delete Wav File afterwards as they are quick to generate and consume too much space
    if os.path.exists(wav_filepath):
        os.remove(wav_filepath)
        
    gc.collect()
    torch.cuda.empty_cache()
    
logging.info('STT Pipeline Complete')
utils.clean_up_missed_wav_files(da.trans_mp3_dir)

