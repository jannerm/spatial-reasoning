import os, subprocess

def mkdir(path):
    if not os.path.exists(path):
        subprocess.call(['mkdir', path])