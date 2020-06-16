import io
import os
import pickle
import ray
import numpy as np
from time import time


REGIONS = ['AGSO', 'JAMSTEC', 'JAMSTEC2', 'NGA', 'NGA2', 'NGDC', 'NOAA_geodas', 'SIO', 'US_multi',
           'US_multi2']
NUM_COLS = 36
# Data columns to not include in the feature representation
REMOVED_FEATURES = [3, 4, 5, 7]

TYPE_INDEX = 35
DATA_TYPE = {
    "M": 1,  # - multibeam
    "G": 2,  # - grid
    "S": 3,  # - single beam
    "P": 4,  # - point measurement
    "X": np.nan,
    'nan': np.nan,
}

CHUNK_SIZE = 100000
INPUT_ROOT = "/cryosat3/btozer/CREATE_ML_FEATURES/tsv_all"
OUTPUT_ROOT = "/cryosat3/jalafate/bathymetry-data"


def init():
    for region in REGIONS:
        dir_path = os.path.join(OUTPUT_ROOT, region)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)


def write_data_to_binary(regionname, basename, features, labels):
    data_len = features.shape[0]
    parts = 0
    for k, i in enumerate(range(0, data_len, CHUNK_SIZE)):
        filename = os.path.join(OUTPUT_ROOT, regionname, basename + "_part{:06d}.pkl".format(k))
        with open(filename, 'wb') as f:
            pickle.dump((features[i:i + CHUNK_SIZE], labels[i:i + CHUNK_SIZE]), f, protocol=4)
        parts += 1
    return parts


# cols[4] == 9999, the instance is corrupted, set label to 0
# otherwise, the instance is good, set the label to 1
def read_data_from_text(filename, get_label=lambda cols: cols[4] != '9999'):
    features = []
    labels = []
    filename = filename.strip()
    format_error = 0
    with io.open(filename, 'r', newline='\n') as fread:
        for line in fread:
            cols = line.strip().split()
            if len(cols) not in [NUM_COLS, NUM_COLS - 2]:
                format_error += 1
                continue
            if len(cols) == NUM_COLS - 2:
                cols = ["X"] * 2
            cols[TYPE_INDEX] = DATA_TYPE[cols[TYPE_INDEX]]
            labels.append(get_label(cols))
            features.append(np.array(
                [float(cols[i]) for i in range(len(cols)) if i not in REMOVED_FEATURES]
            ))
    assert(len(features) == len(labels))
    return (np.array(features), np.array(labels), format_error)


@ray.remote
def parse_region(region, thread_id):
    log_name = "log_tsv2pkl_{:02d}.tsv".format(thread_id)
    if os.path.exists(log_name):
        os.remove(log_name)

    dirname = os.path.join(INPUT_ROOT, region)
    # AGSO   => .tsv_ky 
    # others => .tsv
    ext = ".tsv"
    if region == "AGSO":
        ext = ".tsv_ky"
    for filename in [filename for filename in os.listdir(dirname) if filename.endswith(ext)]:
        basename = os.path.splitext(filename)[0]
        filepath = os.path.join(dirname, filename)
        (features, labels, format_error) = read_data_from_text(filepath)
        parts = write_data_to_binary(region, basename, features, labels)
        log_info = "{}\t{}\t{}\t{}\t{}".format(
            region, basename, features.shape[0], int(np.sum(1 - labels)), parts)
        print(log_info)
        with open(log_name, "a") as f:
            f.write(log_info + "\n")


if __name__ == '__main__':
    init()
    ray.init(num_cpus=10)
    result_ids = []
    for i, region in enumerate(REGIONS):
        result_ids.append(parse_region.remote(region, i))
    ray.get(result_ids)