import pandas as pd
import numpy as np

import logging
import os
import re

import utils
import data_extraction as da

import silero_vad

from pyannote.audio import Pipeline, Inference
from pyannote.core import Segment
# from pyannote.audio.pipelines.utils.hook import ProgressHook

import whisper

import torchaudio

from scipy.spatial.distance import cdist

from evaluate import load

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
    
    # Calculate time removed, so we can re-establish original timing after diarization
    vad_df['duration'] = (vad_df['end'] - vad_df['start'])
    vad_df['running_time'] = vad_df['duration'].cumsum().shift(1).fillna(0)
    vad_df['removed_time'] = vad_df['start'] - vad_df['running_time']
    
    vad_df.to_parquet(vad_out_fp)

    # Now cut audio down to just dialogue
    logging.info(f'Slicing up audio from {movie_name} to speech only')
    silero_vad.save_audio(wav_filepath, silero_vad.collect_chunks(speech_timestamps, full_silero_audio), sampling_rate=da.sample_rate) 
    utils.cleanup_model(silero_model)
    del full_silero_audio
    
    
def apply_diarization(movie_name: str, wav_filepath: str, pyannote_model: str, seg_df_path: str, vad_df_path: str, device):
    
    logging.info(f'Started pyannote pipeline for {movie_name}')
    pyannote_pipeline = Pipeline.from_pretrained(pyannote_model, use_auth_token=utils.get_hf_token())
    pyannote_pipeline.to(device)

    # with ProgressHook() as hook:
    dz = pyannote_pipeline(wav_filepath) # , hook=hook

    # Extract start and end times from segments object and split integer out from 'SPEAKER_x' labels
    records = [(x[0].start, x[0].end, int(x[2].split('_')[-1])) for x in dz.itertracks(yield_label = True)]
    segments_df = pd.DataFrame(records, columns=['start', 'end', 'speaker'])

    segments_df['start_frame'] = (da.sample_rate * segments_df['start']).astype(int)
    segments_df['end_frame'] = (da.sample_rate * segments_df['end']).astype(int)
    segments_df['duration'] = segments_df['end'] - segments_df['start']

    # Check duration of first speaker
    first_speaker_dur = segments_df[segments_df.speaker.eq(segments_df.speaker.iloc[0])].duration.sum()
    # Assume narrator speaks first (describing opening logos etc), but if duration is quite low, check second speaker
    narrator_id = segments_df['speaker'].iloc[0] if first_speaker_dur > 500 else segments_df.speaker.unique()[1]

    segments_df['is_dialogue'] = segments_df['speaker'].ne(narrator_id)
    segments_df['movie_name'] = movie_name
    
    # Use Voice Activity Detection dataframe to restablish correct timing in film (since we filtered out speechless sections)
    vad_df = pd.read_parquet(vad_df_path)
    
    # Iterate through each segment and find it's corresponding segment in the VAD df, then add the time removed before that segment
    for ii in range(segments_df.shape[0]):
        seg_start = segments_df['start'].iloc[ii]
        segments_df.loc[ii, 'uncut_start'] = seg_start + vad_df.removed_time.iloc[vad_df.running_time.gt(seg_start).idxmax() - 1]
        
    segments_df['uncut_end'] = segments_df['uncut_start'] + segments_df['duration']

    segments_df.to_parquet(seg_df_path)

    utils.cleanup_model(pyannote_pipeline)
    del dz
    
    
def clean_text(txt: str) -> str:
    return txt.removesuffix(' Thank you.').replace('.', '').strip()
    
    
def transcribe_segments(transcript_fp: str, seg_df_path: str, wav_filepath: str, whisper_model: str, whisper_config: Dict, narr_cs_limit: float, device, convert_dialogue=False):
    
    segments_df = pd.read_parquet(seg_df_path)
    narrator_true_pos_mask = segments_df.is_dialogue.eq(convert_dialogue) & segments_df.cosine_sim.gt(narr_cs_limit)
    narrator_df = segments_df[narrator_true_pos_mask].copy().reset_index(drop=True)
    
    model = whisper.load_model(whisper_model, device=device)
    audio = whisper.load_audio(wav_filepath, sr=da.sample_rate)
    seg_start_arr, seg_end_arr = narrator_df['start_frame'].values, narrator_df['end_frame'].values

    segment_list = []
    
    for ii in range(len(seg_start_arr)):
        # TODO: replace with tdqm
        if ii % 50 == 0:
            logging.info(f'Segment: {ii + 1} / {len(seg_start_arr)}')
        
        result = {'text': ''}
        segment = audio[seg_start_arr[ii]: seg_end_arr[ii]]
        raw_transcription = model.transcribe(segment, language='en', **whisper_config)
    
        if raw_transcription['language'] == 'en':
            result = raw_transcription
                
        segment_list.append(result)
            
    narrator_df['text'] = [x['text'] for x in segment_list]
    narrator_df['transcription_start_offset'] = [next((y['start'] for y in x['segments'] if clean_text(y['text']) != ''), 0) for x in segment_list]
    narrator_df.to_parquet(transcript_fp)
    
    utils.cleanup_model(model)
    del audio

    return narrator_df
    
    
def add_pyannote_cosine_sim(seg_df_path: str, wav_filepath: str, min_seg_sec: float, device):
    
    segments_df = pd.read_parquet(seg_df_path)
    segments_df['cosine_sim'] = 1.0 - segments_df.is_dialogue.astype(int)
    
    agg_seg_df = da.aggregate_segments(segments_df)
    agg_seg_df = agg_seg_df[agg_seg_df.is_dialogue.eq(False)]
    embedding_model = Inference('pyannote/embedding', device=device, use_auth_token=utils.get_hf_token())
    narrator_segment = Segment(agg_seg_df.start.iloc[0], agg_seg_df.end.iloc[0])
    narrator_embedding = embedding_model.crop(wav_filepath, narrator_segment)
    
    # Pyannote estimates of segment length can go outside file, so prevent this by changing any segments where it is greater
    file_info = torchaudio.info(wav_filepath)
    file_end_time = file_info.num_frames / file_info.sample_rate
    segments_df.loc[segments_df.end > file_end_time, 'end'] = file_end_time

    for ii in range(segments_df.shape[0]):
        if segments_df['duration'].iloc[ii] > min_seg_sec:
            start = segments_df['start'].iloc[ii]
            end = segments_df['end'].iloc[ii]
            segments_df.loc[ii, 'cosine_sim'] = _calc_pyannote_cosine_sim(narrator_embedding, embedding_model, start, end, wav_filepath)
        
    segments_df.to_parquet(seg_df_path)
    
    utils.cleanup_model(embedding_model)
    del narrator_segment, narrator_embedding
    
    
def _calc_pyannote_cosine_sim(narrator_embed, embedding_model, start, end, wav_filepath: str):
    
    test_seg = Segment(start, end)
    test_embedding = embedding_model.crop(wav_filepath, test_seg)
    c_dist = cdist(narrator_embed.data.mean(axis=0, keepdims=True), test_embedding.data.mean(axis=0, keepdims=True), metric="cosine")
    
    return 1 - c_dist.item()
    

def calc_wer(movie_name: str):

    with open(os.path.join(da.transcription_dir, 'manual', f'{movie_name}.txt')) as fileobj:
        raw_txt = fileobj.read()
    ref_txt = re.sub('[\.,"\?!:]', '', raw_txt).lower().replace('-', ' ').replace('\n', ' ')

    trans_df = pd.read_parquet(os.path.join(da.transcription_dir, da.transcript_df_fp.format(movie_name=movie_name)))
    trans_df = trans_df[trans_df['text'].ne(' Thank you.')]
    trans_txt = ''.join(trans_df.text.str.replace('[\.,"\?!]', '', regex=True)).lower().replace('-', ' ')
    
    wer, cer = load('wer'), load('cer')
    wer_score = wer.compute(predictions=[trans_txt], references=[ref_txt])
    cer_score = cer.compute(predictions=[trans_txt], references=[ref_txt])

    return wer_score, cer_score      


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