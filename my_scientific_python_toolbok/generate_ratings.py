import os

import pandas as pd
from utils import data_path

ratings_dest = data_path('ratings.pkl')

all_ratings = None

for i, filename in enumerate(os.listdir(data_path('training_set'))):
    print("\rParsing {:4} of 1000".format(i), end=' ')
    with open(data_path('training_set', filename)) as f:
        movie_id = int(f.readline().replace(':', ''))
        ratings = pd.read_csv(f, header=0, names=['cust_id', 'stars', 'date'], parse_dates=['date'])
        ratings['movie_id'] = movie_id

        if all_ratings is None:
            all_ratings = ratings
        else:
            all_ratings = pd.concat([all_ratings, ratings])
    if i == 1000:
        break

all_ratings.to_pickle(ratings_dest)
