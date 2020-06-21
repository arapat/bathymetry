import io
import os
import pickle
import random
import numpy as np
import pandas as pd
from time import time

DATA_DIR = "/cryosat3/jalafate/bathymetry-data"
INVENTORY_PATH = os.path.join(DATA_DIR, "inventory.tsv")
CHUNK_SIZE = 100000

MODEL_DIR = "runtime_models"
SCORES_DIR = "runtime_scores"

            
######### General #########

def init_setup(base_dir):
    for dirname in [MODEL_DIR, SCORES_DIR]:
        dir_path = os.path.join(base_dir, dirname)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)


def train_test_split(array, rtrain, rvalidate, rtest, shuffle):
    assert(rtrain + rvalidate + rtest == 1.0)
    if shuffle:
        shuffle(array)
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
    for basename, parts, counts, bad in region_inv[["cruise", "parts", "total", "bad"]].values:
        cruise = [
            [os.path.join(DATA_DIR, region, "{}_part{:06d}.pkl".format(basename, k)), CHUNK_SIZE]
            for k in range(parts)
        ]
        if counts % CHUNK_SIZE > 0:
            cruise[-1][1] = counts % CHUNK_SIZE
        partfiles.append((cruise, counts, bad))
    return partfiles


def get_parts(all_regions):
    partfiles = []
    for region in all_regions:
        partfiles.append(get_region_parts(region))
    return partfiles


def remove_region_info_in_parts(all_parts):
    # return [["part1", "part2", ...], ["part1", "part2", ...]]
    return merge_array(all_parts)


def remove_cruise_info_in_parts(all_parts):
    # return ["part11", "part12", ..., "part21", "part22", ...]
    return merge_array(merge_array(all_parts))


def merge_array(array):
    ret = []
    for t in array:
        if type(t) is tuple:
            ret += t[0]
        else:
            ret += t
    return ret

######### Process loaded data #########


def read_data(filepaths):
    assert(type(filepaths) is list and len(filepaths[0]) == 2 and type(filepaths[0][0]) is str
           and type(filepaths[0][1]) is int)
    total = sum([count for _, count in filepaths])
    with open(filepaths[0][0], "rb") as f:
        _f, _ = pickle.load(f)
    _ret = np.empty((total, _f.shape[1] + 1))
    index = 0
    for cruise_part, count in filepaths:
        assert(index + count <= _ret.shape[0])
        with open(cruise_part, "rb") as f:
            features, labels = pickle.load(f)
            try:
                _ret[index:index + count] = np.concatenate(
                    (labels.reshape(-1, 1), features), axis=1)
            except:
                print("Failed to read, {}, {}, {}, {}", cruise_part, count, index, features.shape)
            index += count
    return _ret


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
