import pandas as pd
import pickle
import numpy as np
import json
import os
import requests
import matplotlib
from PIL import Image
import glob
from subprocess import PIPE, call
import string
import glob
import tempfile

DEFAULT_IMG_SIZE = 256
DATA_DIR = 'images/'

def load_data(path):
  with open(path) as f:
    return json.load(f)

def delete_file(file_path):
  os.remove(file_path)


def prep_data(data):
  song_array = []
  for song in data:
    genre = song['genre']
    link = song['tracks'][0][1]
    name = song['tracks'][0][0]
    name = name.split(' - ')[0]
    song_array.append([name, link, genre])
  df = pd.DataFrame(song_array, columns = ['name', 'link', 'genre'])
  return df

def download(url, file_name):
  with open(file_name, "wb") as file:
    response = requests.get(url)
    file.write(response.content)

def set_to_mono(input_file):
  tmp_name = 'tmp.mp3'
  command = "sox {} {} remix 1,2".format(input_file, tmp_name)
  call(command, shell=True)
  delete_file(input_file)
  os.rename(tmp_name, input_file)

def audio_to_spect(input_file, output_file):
  command = "sox {} -n spectrogram -Y 300 -X 50 -m -r -o {}".format(input_file, output_file)
  call(command, shell=True, stdin=PIPE, stdout=PIPE)
  delete_file(input_file)

def get_spect_dims(input_img):
  img_width, img_height = input_img.size
  return img_width, img_height

def get_num_slices(img_width):
  n_slices = img_width // DEFAULT_IMG_SIZE
  return n_slices

def get_slice_dims(input_img):
  img_width, img_height = get_spect_dims(input_img)
  num_slices = get_num_slices(img_width)
  unused_size = img_width - (num_slices * DEFAULT_IMG_SIZE)
  start_px = 0 + unused_size
  image_dims = []
  for i in range(num_slices):
    img_width = DEFAULT_IMG_SIZE
    image_dims.append((start_px, start_px + DEFAULT_IMG_SIZE))
    start_px += DEFAULT_IMG_SIZE
  return image_dims

def slice_spect(input_file):
  input_file_cleaned = input_file.replace('.png', '')
  img = Image.open(input_file)
  dims = get_slice_dims(img)
  counter = 0

  if not os.path.exists(DATA_DIR):
      os.makedirs(DATA_DIR)

  for dim in dims:
    counter_formatted = str(counter).zfill(3)
    img_name = '{}__{}.png'.format(input_file_cleaned, counter_formatted)
    start_width = dim[0]
    end_width = dim[1]
    sliced_img = img.crop((start_width, 0, end_width, DEFAULT_IMG_SIZE))
    sliced_img.save(DATA_DIR + img_name)
    counter += 1
  delete_file(input_file)

def create_parent_genre(s):
  parent_genres = {
    'Minimal/Tech House':'Minimal House',
    'Progressive House':'Not Needed',
    'Funky/Club House':'Funky House',
    'Deep House':'Deep House',
    'Techno':'Techno',
    'Uplifting Trance':'Not Needed',
    'Electro House':'Electro House',
    'Drum And Bass':'Drum And Bass',
    'Dirty Dubstep/Trap/Grime':'Dubstep/Grime',
    'Breakbeat':'Breakbeat',
    'Disco/Nu-Disco':'Disco',
    'Balearic/Downtempo':'Not Needed',
    'Euro Dance/Pop Dance':'Not Needed',
    'Hip Hop/R&B':'Not Needed',
    'Hardstyle':'Not Needed',
    'Psy/Goa Trance':'Not Needed',
    'Dancehall/Ragga':'Not Needed',
    'Hard Trance':'Not Needed',
    'Indie':'Not Needed',
    'UK Hardcore':'Not Needed',
    'Hard House':'Not Needed',
    'Experimental/Electronic':'Not Needed',
    'Pop Trance':'Not Needed',
    'Bass':'Not Needed',
    'Broken Beat/Nu Jazz':'Not Needed',
    'Rock':'Not Needed',
    'Gabba':'Not Needed',
    'Pop':'Pop',
    'UK Garage':'Not Needed',
    'Electro':'Not Needed',
    'Deep Dubstep':'Dubstep/Grime',
    'Roots/Lovers Rock':'Not Needed',
    'Hard Techno':'Not Needed',
    'Ambient/Drone':'Not Needed',
    'Funk':'Not Needed',
    'Scouse House':'Not Needed',
    'Dub':'Not Needed',
    'Coldwave/Synth':'Not Needed',
    'Jazz':'Not Needed',
    'DJ Tools':'Not Needed',
    'Industrial/Noise':'Not Needed',
    'Footwork/Juke':'Not Needed',
    'Classics/Ska':'Not Needed',
    'International':'Not Needed',
    'Soul':'Not Needed',
    'Soundtracks':'Not Needed',
    'Leftfield':'Not Needed',
    '50s/60s':'Not Needed',
    'Rock (All)':'Not Needed',
  }
  parent = parent_genres[s]
  return parent

def create_equal_sized_groups(data_frame):
  not_needed = data_frame[data_frame['parent_genre'] == 'Not Needed']
  df_trimmed = data_frame.drop(not_needed.index)
  grouped = df_trimmed.groupby('parent_genre', as_index=False)
  equal_sample_df = grouped.apply(lambda x: x.sample(200)).reset_index()
  equal_sample_df.drop(['level_0','level_1'], axis=1, inplace=True)
  return equal_sample_df

def save_data_frame_as_pickle(data_frame):
  data_frame.to_pickle('final_data.pkl')