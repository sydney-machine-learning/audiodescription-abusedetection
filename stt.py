import pandas as pd

import logging
import os

import utils
import data_extraction as da

from pydub import AudioSegment
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

from pyannote.audio import Pipeline, Inference
from pyannote.core import Segment
# from pyannote.audio.pipelines.utils.hook import ProgressHook

import whisper

from scipy.spatial.distance import cdist

import difflib
from termcolor import colored

from typing import Dict


def apply_silero_vad_to_wav(mp3_filename: str, wav_filepath: str, vat_out_fp: str, silero_threshold: float, credits_df: pd.DataFrame):
    movie_name = utils.remove_ext(mp3_filename)

    logging.info(f'Applying Silero VAD to {movie_name}')
    silero_model = load_silero_vad()

    full_silero_audio = read_audio(os.path.join(da.trans_mp3_dir, mp3_filename))

    if movie_name in credits_df.movie.values:
        credits_ts = credits_df[credits_df.movie.eq(movie_name)]['credits_start_sec'].iloc[0]
        full_silero_audio = full_silero_audio[:int(credits_ts*da.sample_rate)]

    speech_timestamps = get_speech_timestamps(full_silero_audio, silero_model, threshold=silero_threshold, speech_pad_ms=200)
    vad_df = pd.DataFrame(speech_timestamps).rename(columns={'start': 'start_frames', 'end': 'end_frames'})
    vad_df[['start', 'end']] = vad_df[['start_frames', 'end_frames']] / da.sample_rate
    vad_df.to_parquet(vat_out_fp)

    utils.cleanup_model(silero_model)
    del full_silero_audio

    # Now cut audio down to just dialogue
    logging.info(f'Slicing up Audio from {movie_name} to speech only')
    full_audio = AudioSegment.from_mp3(os.path.join(da.trans_mp3_dir, mp3_filename))
    dialogue_only_audio = AudioSegment.empty()

    if movie_name in credits_df.movie.values:
        full_audio = full_audio[:int(credits_ts*1000)]

    for start, end in zip(vad_df['start'].values, vad_df['end'].values):
        dialogue_only_audio += full_audio[start * 1000: end * 1000]
        
    dialogue_only_audio.export(wav_filepath, format="wav")
    del full_audio, dialogue_only_audio
    
    
def apply_diarization(movie_name: str, wav_filepath: str, pyannote_model: str, seg_df_path: str, device):
    
    logging.info(f'Started pyannote pipeline for {movie_name}')
    pyannote_pipeline = Pipeline.from_pretrained(pyannote_model, use_auth_token=utils.get_hf_token())
    pyannote_pipeline.to(device)

    # with ProgressHook() as hook:
    dz = pyannote_pipeline(wav_filepath) # , hook=hook

    # Extract start and end times from segments object and split integer out from 'SPEAKER_x' labels
    records = [(x[0].start, x[0].end, int(x[2].split('_')[-1])) for x in dz.itertracks(yield_label = True)]
    segments_df = pd.DataFrame(records, columns=['start', 'end', 'speaker'])

    agg_seg_df = da.aggregate_segments(segments_df)

    # Assume narrator speaks first (describing opening logos etc)
    narrator_id = agg_seg_df['speaker'].iloc[0]
    agg_seg_df['is_dialogue'] = agg_seg_df['speaker'].ne(narrator_id)
    agg_seg_df['movie_name'] = movie_name

    agg_seg_df['start_frame'] = (da.sample_rate * agg_seg_df['start']).astype(int)
    agg_seg_df['end_frame'] = (da.sample_rate * agg_seg_df['end']).astype(int)

    agg_seg_df.to_parquet(seg_df_path)

    utils.cleanup_model(pyannote_pipeline)
    del dz
    
    
def transcribe_segments(transcript_fp: str, seg_df_path: str, wav_filepath: str, whisper_model: str, whisper_config: Dict, embed_model: str, cosine_sim_lim: float, device):
    
    segments_df = pd.read_parquet(seg_df_path)
    narrator_df = segments_df[~segments_df.is_dialogue].copy().reset_index(drop=True)
    
    model = whisper.load_model(whisper_model, device=device)
    audio = whisper.load_audio(wav_filepath, sr=da.sample_rate)
    seg_start_arr, seg_end_arr = narrator_df['start_frame'].values, narrator_df['end_frame'].values

    embedding_model = Inference(embed_model, device=device, use_auth_token=utils.get_hf_token())
    narrator_segment = Segment(segments_df.start.iloc[0], segments_df.end.iloc[0])
    narrator_embedding = embedding_model.crop(wav_filepath, narrator_segment)

    segment_list = []
    max_end = seg_end_arr[-1] / da.sample_rate - 0.03

    for ii in range(len(seg_start_arr)):
        if ii % 50 == 0:
            logging.info(f'Segment: {ii + 1} / {len(seg_start_arr)}')
            
        end_ts = min(seg_end_arr[ii] / da.sample_rate, max_end)
        test_seg = Segment(seg_start_arr[ii] / da.sample_rate, end_ts)
        test_embedding = embedding_model.crop(wav_filepath, test_seg)
        sim = 1 - cdist(narrator_embedding.data.mean(axis=0, keepdims=True), test_embedding.data.mean(axis=0, keepdims=True), metric="cosine")
        
        result = {'text': ''}
        if sim > cosine_sim_lim:
            segment = audio[seg_start_arr[ii]: seg_end_arr[ii]]
            raw_transcription = model.transcribe(segment, language='en', **whisper_config)
        
            if raw_transcription['language'] == 'en':
                result = raw_transcription
                
        segment_list.append(result)
            
    narrator_df['text'] = [x['text'] for x in segment_list]
    narrator_df.to_parquet(transcript_fp)

    utils.cleanup_model(model)
    del audio
    
    
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