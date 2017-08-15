# coding: utf-8
import pickle
import sys
import pandas as pd
import requests
import sox
import numpy as np
import os
from subprocess import call, PIPE
from PIL import Image
import tempfile
import re
import time
import string
import glob
from helper import *

class_labels = {
  0: "Drum And Bass",
  1: "Minimal House",
  2: "Electro House",
  3: "Funky House",
  4: "Deep House",
  5: "Dubstep/Grime",
  6: "Disco",
  7: "Breakbeat",
  8: "Techno",
}

DEFAULT_IMG_SIZE = 256
DATA_DIR = 'images/'

df = pd.read_pickle('final_data.pkl')

def create_file_names(id):
    genre_list = list(df['parent_genre'])
    genre_name = str(genre_list[id]).lower()
    genre_name = genre_name.replace('/','_')
    genre_name = genre_name.replace(' ','_')
    genre_name = genre_name.replace('&', 'n')
    name_list = list(df['name'])
    track_name_raw = (name_list[id])
    track_name_raw = track_name_raw.replace(' ', '-')
    track_name_raw = track_name_raw.replace('(', '-')
    track_name_raw = track_name_raw.replace(')', '-')
    track_name_raw = track_name_raw.replace('&', 'n')
    track_name = '{}__{}.mp3'.format(genre_name, track_name_raw)
    spect_name = track_name.replace('.mp3','')
    spect_name = '{}.png'.format(spect_name)
    return track_name, spect_name, genre_name


url_list = list(df['link'])
for track_id in range(0, 3):
  url = url_list[track_id]
  track_name, spect_name, genre_name = create_file_names(track_id)
  print('Track: {}, Spect: {}, Genre: {}'
        .format(track_name, spect_name, genre_name))
  
  try:
    download(url, track_name)
    set_to_mono(track_name)
    audio_to_spect(track_name, spect_name)
    slice_spect(spect_name)

  except KeyboardInterrupt:
        sys.exit()
