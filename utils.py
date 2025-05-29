import os

import gc
import torch

       
def remove_ext(filename):
    return os.path.splitext(filename)[0]


def cleanup_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()
