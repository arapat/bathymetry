import json
import os
import sys
import numpy as np
import pickle
import random
from random import shuffle

from . import booster
from .common import Logger
from .load_data import init_setup
from .load_data import get_parts
from .load_data import remove_region_info_in_parts
from .load_data import merge_array
from .load_data import read_data
from .load_data import persist_model
from .load_data import persist_predictions

REGIONS = ['AGSO', 'JAMSTEC', 'JAMSTEC2', 'NGA', 'NGA2', 'NGDC', 'NOAA_geodas', 'SIO', 'US_multi']


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        config = json.load(f)
    work_dir = config["work_dir"]
    init_setup(work_dir)
    region = "all"

    logger = Logger()
    logfile = os.path.join(work_dir, "training_{}.log".format(region))
    logger.set_file_handle(logfile)

    region_level = get_parts(REGIONS)
    cruise_level = remove_region_info_in_parts(region_level)
    shuffle(cruise_level)

    logger.log("total number of cruises: {}".format(len(cruise_level)))
    cruise_level = [sample for sample in cruise_level if random.random() <= 0.34]
    logger.log("sampled number of cruises: {}".format(len(cruise_level)))
    parts = merge_array(cruise_level)
    logger.log("total number of parts: {}".format(len(parts)))
    logger.log("start loading data")
    data = read_data(parts)

    logger.log("start cross validation")
    cv_results = booster.cv(config, data, 10, shuffle=False, logger=logger)
    logger.log("{}".format(cv_results))
    logger.log("finished cross validation")
    with open(os.path.join(work_dir, "cv_results.pkl"), "wb") as f:
        pickle.dump((parts, cv_results), f)


# if __name__ == "__main__":
#     with open(sys.argv[1]) as f:
#         config = json.load(f)
#     work_dir = config["work_dir"]
#     init_setup(work_dir)
#     region = "all"
# 
#     logger = Logger()
#     logfile = os.path.join(work_dir, "training_{}.log".format(region))
#     logger.set_file_handle(logfile)
# 
#     for i in range(10):
#         logger.log("start iteration, {}".format(i))
#         parts = get_parts(REGIONS)
#         parts = remove_region_info_in_parts(parts)
#         logger.log("start splitting")
#         trains, validates, tests = train_test_split(parts, 0.7, 0.15, 0.15)
# 
#         logger.log("start loading training data")
#         data = read_data(trains)
#         data = transform_data_get_cruises(data)
#         train_features, train_labels       = split_features_labels(data)
#         logger.log("train, {}, {}".format(train_features.shape[0], np.sum(1.0 - train_labels)))
#         data = read_data(validates)
#         data = transform_data_get_cruises(data)
#         validate_features, validate_labels = split_features_labels(data)
#         logger.log("validate, {}, {}".format(
#             validate_features.shape[0], np.sum(1.0 - validate_labels)))
# 
#         # Training
#         logger.log("start training")
#         gbm = booster.train(
#             config, train_features, train_labels, validate_features, validate_labels, logger
#         )
#         logger.log("finish training")
#         persist_model(config["work_dir"], region, gbm, i)
# 
#         # Testing
#         logger.log("start loading test data")
#         data = read_data(tests)
#         data = transform_data_get_cruises(data)
#         test_features, test_labels         = split_features_labels(data)
#         logger.log("test, {}, {}".format(test_features.shape[0], np.sum(1.0 - test_labels)))
#         scores = booster.test(
#             gbm, "all", "all", test_features, test_labels, logger
#         )
#         persist_predictions(work_dir, "all", "all", test_features, test_labels, scores, i)
#         logger.log("finish testing".format(region))
