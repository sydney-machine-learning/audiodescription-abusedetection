import os
import json
import gc

import torch

       
def remove_ext(filename):
    return os.path.splitext(filename)[0]


def cleanup_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    
def clean_up_missed_wav_files(dir: str):
    # Cleanup Missed wav files
    wav_files = [x for x in os.listdir(dir) if os.path.splitext(x)[-1].lower() == '.wav']

    for path in wav_files:
        os.remove(os.path.join(dir, path))
    

def get_hf_token():
    with open('config.json') as fileobj:
        hf_token = json.load(fileobj)['hugging_face_token']
        
    return hf_token
