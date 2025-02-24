import os

def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
      
        
def remove_ext(filename):
    return os.path.splitext(filename)[0]
