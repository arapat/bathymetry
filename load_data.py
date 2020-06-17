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

            
######### General #########

def init_setup(base_dir):
    for dirname in [MODEL_DIR, SCORES_DIR]:
        dir_path = os.path.join(base_dir, dirname)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)


def train_test_split(array, rtrain, rvalidate, rtest):
    assert(rtrain + rvalidate + rtest == 1.0)
    random.shuffle(array)
    r0, r1 = int(len(array) * rtrain), int(len(array) * rvalidate)
    r1 = r0 + r1
    trains, validates, tests = array[:r0], array[r0:r1], array[r1:]
    return (trains, validates, tests)

            
######### Process filenames #########

def get_region_parts(region):
    inv = pd.read_csv(INVENTORY_PATH, sep="\t", names=["region", "cruise", "total", "bad", "parts"])
    inv = inv[inv["parts"] > 0]
    region_inv = inv[inv["region"] == region]
    partfiles = []
    for basename, parts in region_inv[["cruise", "parts"]].values:
        partfiles.append([
            os.path.join(DATA_DIR, region, "{}_part{:06d}.pkl".format(basename, k)) for k in range(parts)
        ])
    return partfiles


def get_parts(all_regions):
    partfiles = []
    for region in all_regions:
        partfiles.append(get_region_parts(region))
    return partfiles


def remove_region_info_in_parts(all_parts):
    # return [["part1", "part2", ...], ["part1", "part2", ...]]
    def merge_array(array):
        ret = []
        for t in array:
            ret += t
        return ret
    return merge_array(all_parts)


######### Process loaded data #########


def read_data(filepaths):
    assert(type(filepaths) is list and type(filepaths[0]) is list and type(filepaths[0][0]) is str)
    _ret = []
    for cruise in filepaths:
        ret = []
        for parts in cruise:
            with open(parts, "rb") as f:
                features, labels = pickle.load(f)
                ret.append((features, labels.astype(np.int)))
        _ret.append(ret)
    return _ret


def transform_data_get_parts(data):
    # return list of numpy
    ret = []
    for t in data:
        ret += t
    return ret


def transform_data_get_cruises(data):
    # return list of numpy
    ret = []
    for cruise in data:
        _features = []
        _labels = []
        for a, b in cruise:
            _features.append(a)
            _labels.append(b)
        ret.append(
            (np.concatenate(_features, axis=0), np.concatenate(_labels, axis=0))
        )
    return ret


def split_features_labels(dataset):
    return (
        np.concatenate([a for a, b in dataset]),
        np.concatenate([b for a, b in dataset]),
    )


######### Persist models and scores #########


def get_model_path(base_dir, region, model_id):
    dir_path = os.path.join(base_dir, MODEL_DIR)
    return os.path.join(dir_path, '{}_model_{}'.format(region, model_id))


def persist_predictions(base_dir, model_region, test_region, features, label, scores, model_id):
    def get_prediction_path(base_dir, model_region, test_region, model_id):
        dir_path = os.path.join(base_dir, SCORES_DIR)
        return os.path.join(dir_path, 'model_{}_test_{}_scores_{}.pkl'.format(
            model_region, test_region, model_id))

    filepath = get_prediction_path(base_dir, model_region, test_region, model_id)
    with open(filepath, 'wb') as fout:
        pickle.dump((features[:, :4], label, scores), fout)


def persist_model(base_dir, region, gbm, model_id):
    model_path = get_model_path(base_dir, region, model_id)
    pkl_model_path = model_path + ".pkl"
    txt_model_path = model_path + ".txt"
    with open(pkl_model_path, 'wb') as fout:
        pickle.dump(gbm, fout)
    gbm.save_model(txt_model_path)
