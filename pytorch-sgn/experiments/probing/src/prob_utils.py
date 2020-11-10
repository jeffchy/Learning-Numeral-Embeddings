import os

def create_dir(path):
    try:
        os.mkdir(path)
    except:
        pass