# -*- coding: utf-8 -*-
"""Script to separate data into train-test-validate sets taking loops with the same sequence into account
"""
import pandas as pd
import numpy as np
import os
from ABDB import database as db

test_set = ["1x9q", "2d7t", "3hc4", "1mlb", "2e27", "3g5y", "3m8o", "1jpt", "3e8u", "1mqk", "1nlb", "2adf", "2fbj", "2w60", "3gnm", "3hnt", "3v0w", "1mfa", "3mxw", "2xwt", "1dlf", "2ypv", "3ifl", "3liz", "3oz9", "3umt", "4h0h", "4h20", "4hpy", "2v17", "3t65", "1oaq", "2vxv", "3eo9", "3p0y", "1jfq", "2r8s", "3i9g", "3giz", "3go1", "1fns", "1seq", "1gig", "3mlr", "4nzu", "3lmj", "4f57", "2fb4", "3nps"]

current_directory = os.path.dirname(os.path.realpath(__file__))
data = pd.read_csv(os.path.join(current_directory,"data.csv"))

# Separating the test data from the train and validate data. 

new_test = []
for v in test_set:
    new_test += [x[0] for x in db.fetch(v).fabs[0].get_identicals(0.9999)] + [v]
    
test_data = data[[x[:-1] in test_set for x in data.ID]]
test_data.to_csv(os.path.join(current_directory,"test_data.csv"), index = False)

# Removing test data from train/validate dataset.
# All structures proceeding from antibodies with the same Fv are removed too.

test_set = np.unique(new_test)
new_data = data[[x[:-1] not in test_set for x in data.ID]]
new_data = new_data.sample(frac=1).reset_index(drop=True)

# Randomly select 20 distinct structures for validation
val_set = [x for x in new_data.loc[:100].ID]

# All structures proceeding from antibodies with the same Fv are removed too.
new_vals = []
for v in val_set:
    new_vals += [x[0] for x in db.fetch(v[:-1]).fabs[0].get_identicals(0.9999)] + [v]
    
val_data = new_data[new_data.ID.isin(np.unique(new_vals))]
val_data.to_csv(os.path.join(current_directory,"validation_data.csv"), index = False)

# Saving the remaining data as training data

train_data = new_data[~new_data.ID.isin(np.unique(new_vals))]
train_data.to_csv(os.path.join(current_directory,"train_data.csv"), index = False)
