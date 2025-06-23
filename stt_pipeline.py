import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(asctime)s - %(message)s')

import pandas as pd

import os

pyannote_model = 'pyannote/speaker-diarization-3.1'
embedding_model = "pyannote/embedding" # speechbrain/spkrec-ecapa-voxceleb

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import data_extraction as da
import stt
import utils

import gc
import psutil
import warnings
warnings.filterwarnings("ignore")

logging.getLogger("speechbrain").setLevel(logging.WARNING)
logging.getLogger("pyannote").setLevel(logging.WARNING)
logging.getLogger("pytorch").setLevel(logging.WARNING)

# CONFIG PARAMS
use_vad = True
narr_cosine_sim_lim = 0.14
min_seg_sec = 0

whisper_model = 'turbo'
silero_threshold = 0.5

whisper_config = {
    'beam_size': 7,
    'no_speech_threshold': 0.1,
    'condition_on_previous_text': True
}

# RUN PARAMS
overwrite_files = True

# Torch (pyannote) isn't familiar with MP3 files, so convert to wav for effective performance
# Perform diarization to help separate narration in audio description from dialogue in original movie
# Finally use OpenAI's Whisper to convert to a transcript

mp3_files = da.get_sorted_mp3_list()

for ii, mp3_filename in enumerate(mp3_files):
    
    if mp3_filename == 'Drive My Car.mp3':
        continue
    
    movie_name = utils.remove_ext(mp3_filename)
    
    vad_df_path = os.path.join(da.voice_activity_dir, f'{movie_name}-vad.parquet')
    seg_df_path = os.path.join(da.diarization_dir, f'{movie_name}-diarization.parquet')
    curr_transcript_fp = os.path.join(da.transcription_dir, da.transcript_df_fp.format(movie_name=movie_name))
    wav_filepath = os.path.join(da.trans_mp3_dir, f'{movie_name}_speech_only.wav')

    # If either diarization or transcript is missing, we'll need to generate the wav file
    if not os.path.exists(curr_transcript_fp) or not os.path.exists(seg_df_path) or overwrite_files:
        logging.info(f'{ii} \t/ {len(mp3_files)} \t {movie_name}')
        stt.apply_silero_vad_to_wav(mp3_filename, wav_filepath, vad_df_path, silero_threshold)
            
    # Only perform diarization if parquet doesn't exist
    if not os.path.exists(seg_df_path) or overwrite_files:
        stt.apply_diarization(movie_name, wav_filepath, pyannote_model, seg_df_path, vad_df_path, device)
        stt.add_pyannote_cosine_sim(seg_df_path, wav_filepath, min_seg_sec=min_seg_sec, device=device)

    # Only perform transcription if parquet doesn't exist
    if not os.path.exists(curr_transcript_fp) or overwrite_files:
        stt.transcribe_segments(curr_transcript_fp, seg_df_path, wav_filepath, whisper_model, whisper_config, narr_cosine_sim_lim, device)
        
    # Delete Wav File afterwards as they are quick to generate and consume too much space
    if os.path.exists(wav_filepath):
        os.remove(wav_filepath)
        
    gc.collect()
    torch.cuda.empty_cache()
    
utils.clean_up_missed_wav_files(da.trans_mp3_dir)

