import io
import os
import pickle
import random
import numpy as np
import pandas as pd
from time import time

DATA_DIR = "/cryosat3/jalafate/bathymetry-data"
INVENTORY_PATH = os.path.join(DATA_DIR, "inventory.tsv")

MODEL_DIR = "runtime_models"
SCORES_DIR = "runtime_scores"


def init_setup(base_dir):
    for dirname in [MODEL_DIR, SCORES_DIR]:
        dir_path = os.path.join(base_dir, dirname)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)


def read_data(filepaths):
    _features, _labels = [], []
    for filename in filepaths:
        with open(filename, 'rb') as f:
            features, labels = pickle.load(f)
        _features += features.tolist()
        _labels += labels.tolist()
    return np.array(_features), np.array(_labels).astype(np.int)


def get_region_parts(region):
    inv = pd.read_csv(INVENTORY_PATH, sep="\t", names=["region", "cruise", "total", "bad", "parts"])
    region_inv = inv[inv["region"] == region]
    partfiles = []
    for basename, parts in region_inv[["cruise", "parts"]].values:
        for k in range(parts):
            partfiles.append(
                os.path.join(DATA_DIR, region, "{}_part{:06d}.pkl".format(basename, k)))
    return partfiles


def get_all_parts(all_regions):
    partfiles = []
    for region in all_regions:
        partfiles += get_region_parts(region)
    return partfiles


def train_test_split(files, rtrain, rvalidate, rtest):
    assert(rtrain + rvalidate + rtest == 1.0)
    filenames = files[::]
    random.shuffle(filenames)
    r0, r1 = int(len(filenames) * rtrain), int(len(filenames) * rvalidate)
    r1 = r0 + r1
    trains, validates, tests = filenames[:r0], filenames[r0:r1], filenames[r1:]
    return (trains, validates, tests)



def get_model_path(base_dir, region):
    dir_path = os.path.join(base_dir, MODEL_DIR)
    return os.path.join(dir_path, '{}_model.pkl'.format(region))


def get_prediction_path(base_dir, model_region, test_region):
    dir_path = os.path.join(base_dir, SCORES_DIR)
    return os.path.join(dir_path, 'model_{}_test_{}_scores.pkl'.format(model_region, test_region))


def persist_predictions(base_dir, model_region, test_region, features, label, scores, weights):
    with open(get_prediction_path(base_dir, model_region, test_region), 'wb') as fout:
        pickle.dump((features[:, :4], label, scores, weights), fout)


def persist_model(base_dir, region, gbm):
    pkl_model_path = get_model_path(base_dir, region)
    txt_model_path = pkl_model_path.rsplit('.', 1)[0] + ".txt"
    with open(pkl_model_path, 'wb') as fout:
        pickle.dump(gbm, fout)
    gbm.save_model(txt_model_path)
