"""Script to separate data into train-test-validate sets taking loops with the same sequence into account
"""
import pandas as pd
import numpy as np
import os
from ABDB import database as db

# Using the same test set as in the deepH3 paper to fairly compare in the future.
test_set = ["1x9q", "2d7t", "3hc4", "1mlb", "2e27", "3g5y", "3m8o", "1jpt", "3e8u", "1mqk", "1nlb", "2adf", "2fbj",
            "2w60", "3gnm", "3hnt", "3v0w", "1mfa", "3mxw", "2xwt", "1dlf", "2ypv", "3ifl", "3liz", "3oz9", "3umt",
            "4h0h", "4h20", "4hpy", "2v17", "3t65", "1oaq", "2vxv", "3eo9", "3p0y", "1jfq", "2r8s", "3i9g", "3giz",
            "3go1", "1fns", "1seq", "1gig", "3mlr", "4nzu", "3lmj", "4f57", "2fb4", "3nps"]

# Load file containing data (Assumes it is in the same directory)
current_directory = os.path.dirname(os.path.realpath(__file__))
data = pd.read_csv(os.path.join(current_directory, "data.csv"))

# Separating the test data from the train and validate data.
new_test = []
for v in test_set:
    new_test += [x[0] for x in db.fetch(v).fabs[0].get_identicals(0.9999)] + [v]

test_data = data[[x[:-1] in test_set for x in data.ID]]
test_data.to_csv(os.path.join(current_directory, "test_data.csv"), index=False)

# Removing test data from train/validate dataset.
# All structures proceeding from antibodies with the same Fv are removed too.
test_set = np.unique(new_test)
new_data = data[[x[:-1] not in test_set for x in data.ID]]
new_data = new_data.sample(frac=1).reset_index(drop=True)

# I randomly selected a 100 and generated decoys, so I will keep them
val_set = ['6tcqH', '5tfwH', '43caD', '3l7fE', '6cw3C', '5kvgH', '4gxvH', '5ezlA', '1ct8B', '5t6lH', '6i07A', '4hwbH',
           '4xvuE', '6avnH', '4n9gH', '1ngwB', '5ea0H', '3bkcH', '6higH', '1fskC', '4kmtH', '2fjfR', '1jglH', '4f37H',
           '4jn1H', '1d6vH', '4dvbA', '6u1tH', '6k0yA', '6iguH', '6nmrJ', '1himM', '5mhgH', '4gw5D', '5dtfA', '3r1gH',
           '4d9lK', '5kn5A', '3esvF', '5w06H', '1ngyB', '4rzcH', '5x8mB', '6mi2D', '6uteE', '6dwiJ', '6k4zA', '2qqkH',
           '4xc1H', '6g8rA', '5y9jH', '4ffyH', '4s1dO', '6ubiD', '1wt5B', '6pa0A', '4jn2H', '2w9eH', '1ncbH', '3phoB',
           '1nd0F', '5mykB', '4yhpE', '6db5I', '5mv4I', '4xgzE', '5iczD', '4gmtH', '5j13C', '4k7pH', '5ugyJ', '6mg7H',
           '6o9hH', '3kymD', '4ffvD', '6ejmI', '2vxqH', '1hzhH', '1bfoB', '6dc9I', '4xvuC', '1xgtB', '3nfpA', '6vbpA',
           '2mpaH', '2bfvH', '6oz2H', '2h9gB', '5xcqA', '5i16B', '3mlyH', '6osvH', '5xcrD', '4k9eH', '6b0sH', '3skjI',
           '6bzvC']

# All structures proceeding from antibodies with the same Fv are removed too.
new_vals = []
for v in val_set:
    new_vals += [x[0] for x in db.fetch(v[:-1]).fabs[0].get_identicals(0.9999)] + [v]

val_data = new_data[new_data.ID.isin(np.unique(new_vals))]
val_data.to_csv(os.path.join(current_directory, "validation_data.csv"), index=False)

# Saving the remaining data as training data

train_data = new_data[~new_data.ID.isin(np.unique(new_vals))]
train_data.to_csv(os.path.join(current_directory, "train_data.csv"), index=False)
