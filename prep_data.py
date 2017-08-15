import pandas as pd
import numpy as np
from helper import *

songs = load_data('songs.json')
df = prep_data(songs)
df['parent_genre'] = df['genre'].apply(create_parent_genre)
df = create_equal_sized_groups(df)

save_data_frame_as_pickle(df)