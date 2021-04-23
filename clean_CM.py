import os
import sys
import pickle
import json

import numpy as np
from load_data import get_prediction_path, load_predictions, init_setup
from common import Logger

usage_msg = "Usage: python clean_CM.py <MODEL> <TESTED> <config_path>"

def edit_one_cm(cm_filename, scores, logger):
    """
    Append model scores to CM, reformat for Py-CMeditor
    """
    logger.log("Cleaning CM file, {}".format(cm_filename))
    cm_edit_filename = cm_filename + '.edit.' + model_source
    logger.log("Writing new file, {}".format(cm_edit_filename))
    fwrite = open(cm_edit_filename,'w+')

    with open(cm_filename, 'r', newline='\n') as fread:
        for line in fread:
            fields = line.strip().split()
            fields = fields[0:6]
            fields.append(scores[0])

            scores = scores[1:]
            fwrite.write("{}\n".format(" ".join(fields)))

    fwrite.close()
    return scores

def get_cm_filename(filelist):
    CM_filelist = []
    with open(filelist) as f:
        for line in f:
            dir_name = os.path.dirname(line).replace("tsv_all","cm_data/public")
            cm_name = os.path.basename(line).replace(".tsv",".cm").strip()
            CM_filelist.append(dir_name + '/' + cm_name)
    return CM_filelist

if __name__ == '__main__':
    # args: Model-source Test-source config.json
    # e.g., python clean_CM.py NGDC US_multi2 config.json
    if len(sys.argv) != 4:
        print(usage_msg)
        sys.exit(1)
    model_source = sys.argv[1]
    test_source = sys.argv[2]

    with open(sys.argv[3]) as f:
        config = json.load(f)
    config["base_dir"] = os.path.expanduser(config["base_dir"])
    init_setup(config["base_dir"])

    logger = Logger()
    logfile = os.path.join(config["base_dir"], "CM-cleaning.log")
    logger.set_file_handle(logfile)
    logger.log("begin CM cleaning for {} data trained on {} model".format(test_source, model_source))

    prediction_file = get_prediction_path(config["base_dir"], model_source, test_source)
    _, _, scores, _ = load_predictions(prediction_file,logger)

    CM_filelist = get_cm_filename(config["testing_files"])

    for CM_file in CM_filelist:
        scores = clean_one_cm(CM_file, scores, logger)
