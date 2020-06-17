import json
import os
import sys
import numpy as np

from . import booster
from .common import Logger
from .load_data import init_setup
from .load_data import get_parts
from .load_data import remove_region_info_in_parts
from .load_data import read_data
from .load_data import transform_data_get_cruises
from .load_data import train_test_split
from .load_data import split_features_labels
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

    logger.log("start loading data")
    parts = get_parts(REGIONS)
    parts = remove_region_info_in_parts(parts)
    data = read_data(parts)
    logger.log("finished loading data")

    cruises = transform_data_get_cruises(data)
    for i in range(10):
        # Train/test split
        logger.log("start splitting")
        trains, validates, tests = train_test_split(cruises, 0.7, 0.15, 0.15)
        train_features, train_labels       = split_features_labels(trains)
        logger.log("train, {}, {}".format(train_features.shape[0], np.sum(1.0 - train_labels)))
        validate_features, validate_labels = split_features_labels(validates)
        logger.log("validate, {}, {}".format(
            validate_features.shape[0], np.sum(1.0 - validate_labels)))
        test_features, test_labels         = split_features_labels(tests)
        logger.log("test, {}, {}".format(test_features.shape[0], np.sum(1.0 - test_labels)))

        # Training
        logger.log("start training")
        gbm = booster.train(
            config, train_features, train_labels, validate_features, validate_labels, logger
        )
        logger.log("finish training")
        persist_model(config["work_dir"], region, gbm, i)

        # Testing
        logger.log("start testing".format(region))
        scores = booster.test(
            gbm, "all", "all", test_features, test_labels, logger
        )
        persist_predictions(work_dir, "all", "all", test_features, test_labels, scores, i)
        logger.log("finish testing".format(region))
