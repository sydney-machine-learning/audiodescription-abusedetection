import pandas as pd

import logging
import os

import utils
import data_extraction as da

import silero_vad

from pyannote.audio import Pipeline, Inference
from pyannote.core import Segment
# from pyannote.audio.pipelines.utils.hook import ProgressHook

import whisper

from scipy.spatial.distance import cdist

import difflib
from termcolor import colored

from typing import Dict
   

def apply_silero_vad_to_wav(mp3_filename: str, wav_filepath: str, vad_out_fp: str, silero_threshold: float, credits_df: pd.DataFrame = None):
    movie_name = utils.remove_ext(mp3_filename)

    logging.info(f'Applying Silero VAD to {movie_name}')
    silero_model = silero_vad.load_silero_vad()

    full_silero_audio = silero_vad.read_audio(os.path.join(da.trans_mp3_dir, mp3_filename))

    if credits_df is not None and movie_name in credits_df.movie.values:
        credits_ts = credits_df[credits_df.movie.eq(movie_name)]['credits_start_sec'].iloc[0]
        full_silero_audio = full_silero_audio[:int(credits_ts*da.sample_rate)]

    speech_timestamps = silero_vad.get_speech_timestamps(full_silero_audio, silero_model, threshold=silero_threshold, speech_pad_ms=200)
    vad_df = pd.DataFrame(speech_timestamps).rename(columns={'start': 'start_frames', 'end': 'end_frames'})
    vad_df[['start', 'end']] = vad_df[['start_frames', 'end_frames']] / da.sample_rate
    vad_df.to_parquet(vad_out_fp)

    # Now cut audio down to just dialogue
    logging.info(f'Slicing up audio from {movie_name} to speech only')
    silero_vad.save_audio(wav_filepath, silero_vad.collect_chunks(speech_timestamps, full_silero_audio), sampling_rate=da.sample_rate) 
    utils.cleanup_model(silero_model)
    del full_silero_audio
    
    
def apply_diarization(movie_name: str, wav_filepath: str, pyannote_model: str, seg_df_path: str, device):
    
    logging.info(f'Started pyannote pipeline for {movie_name}')
    pyannote_pipeline = Pipeline.from_pretrained(pyannote_model, use_auth_token=utils.get_hf_token())
    pyannote_pipeline.to(device)

    # with ProgressHook() as hook:
    dz = pyannote_pipeline(wav_filepath) # , hook=hook

    # Extract start and end times from segments object and split integer out from 'SPEAKER_x' labels
    records = [(x[0].start, x[0].end, int(x[2].split('_')[-1])) for x in dz.itertracks(yield_label = True)]
    segments_df = pd.DataFrame(records, columns=['start', 'end', 'speaker'])

    # Assume narrator speaks first (describing opening logos etc)
    narrator_id = segments_df['speaker'].iloc[0]
    segments_df['is_dialogue'] = segments_df['speaker'].ne(narrator_id)
    segments_df['movie_name'] = movie_name

    segments_df['start_frame'] = (da.sample_rate * segments_df['start']).astype(int)
    segments_df['end_frame'] = (da.sample_rate * segments_df['end']).astype(int)
    segments_df['duration'] = segments_df['end'] - segments_df['start']

    segments_df.to_parquet(seg_df_path)

    utils.cleanup_model(pyannote_pipeline)
    del dz
    
    
def transcribe_segments(transcript_fp: str, seg_df_path: str, wav_filepath: str, whisper_model: str, whisper_config: Dict, narr_cs_limit: float, device):
    
    segments_df = pd.read_parquet(seg_df_path)
    narrator_true_pos_mask = segments_df.is_dialogue.eq(False) & segments_df.cosine_sim.gt(narr_cs_limit)
    narrator_df = segments_df[narrator_true_pos_mask].copy().reset_index(drop=True)
    
    model = whisper.load_model(whisper_model, device=device)
    audio = whisper.load_audio(wav_filepath, sr=da.sample_rate)
    seg_start_arr, seg_end_arr = narrator_df['start_frame'].values, narrator_df['end_frame'].values

    segment_list = []
    
    for ii in range(len(seg_start_arr)):
        if ii % 50 == 0:
            logging.info(f'Segment: {ii + 1} / {len(seg_start_arr)}')
        
        result = {'text': ''}
        segment = audio[seg_start_arr[ii]: seg_end_arr[ii]]
        raw_transcription = model.transcribe(segment, language='en', **whisper_config)
    
        if raw_transcription['language'] == 'en':
            result = raw_transcription
                
        segment_list.append(result)
            
    narrator_df['text'] = [x['text'] for x in segment_list]
    narrator_df.to_parquet(transcript_fp)
    
    utils.cleanup_model(model)
    del audio
    
    
def add_pyannote_cosine_sim(seg_df_path: str, wav_filepath: str, min_seg_sec: float, device):
    
    segments_df = pd.read_parquet(seg_df_path)
    segments_df['cosine_sim'] = 1 - segments_df.is_dialogue.astype(int)
    
    agg_seg_df = da.aggregate_segments(segments_df)
    embedding_model = Inference('pyannote/embedding', device=device, use_auth_token=utils.get_hf_token())
    narrator_segment = Segment(agg_seg_df.start.iloc[0], agg_seg_df.end.iloc[0])
    narrator_embedding = embedding_model.crop(wav_filepath, narrator_segment)
    
    # Small offset at end due to tiny file length discrepancies
    max_end = segments_df['end'].iloc[-1] - 0.03

    for ii in range(1, segments_df.shape[0]):
        if segments_df['duration'].iloc[ii] > min_seg_sec:
            end = min(segments_df['end'].iloc[ii], max_end)
            segments_df['cosine_sim'].iloc[ii] = _calc_pyannote_cosine_sim(narrator_embedding, embedding_model, segments_df['start'].iloc[ii], end, wav_filepath)
        
    segments_df.to_parquet(seg_df_path)
    
    utils.cleanup_model(embedding_model)
    
    
def _calc_pyannote_cosine_sim(narrator_embed, embedding_model, start, end, wav_filepath: str):
    
    test_seg = Segment(start, end)
    test_embedding = embedding_model.crop(wav_filepath, test_seg)
    c_dist = cdist(narrator_embed.data.mean(axis=0, keepdims=True), test_embedding.data.mean(axis=0, keepdims=True), metric="cosine")
    
    return 1 - c_dist.item()
    
    
# TODO: reference appropriately
def visualise_wer_differences(candidate_txt: str, reference_txt: str):
    ref_words = reference_txt.split()
    hyp_words = candidate_txt.split()

    matcher = difflib.SequenceMatcher(None, ref_words, hyp_words)

    for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
        if opcode == 'equal':
            print(" ".join(ref_words[i1:i2]), end=' ')
        elif opcode == 'insert':
            print(colored(" ".join(hyp_words[j1:j2]), 'green', attrs=['bold']), end=' ')
        elif opcode == 'delete':
            print(colored(" ".join(ref_words[i1:i2]), 'red', attrs=['bold']), end=' ')
        elif opcode == 'replace':
            print(colored(" ".join(ref_words[i1:i2]), 'red', attrs=['bold']), end='|')
            print(colored(" ".join(hyp_words[j1:j2]), 'yellow', attrs=['bold']), end=' ')
    print()
    
    
def count_wer_difference_types(candidate_txt: str, reference_txt: str):
    ref_words = reference_txt.split()
    hyp_words = candidate_txt.split()

    matcher = difflib.SequenceMatcher(None, ref_words, hyp_words)
    counts = {'equal': 0, 'insert': 0, 'delete': 0, 'replace': 0}

    for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
        counts[opcode] += max(j2-j1, i2-i1)
        
    return counts