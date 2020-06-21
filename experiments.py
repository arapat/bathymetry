import os
import random
import pickle
from random import shuffle

from . import booster
from .common import Logger
from .booster import get_lgb_dataset
from .load_data import get_parts
from .load_data import remove_region_info_in_parts
from .load_data import merge_array
from .load_data import read_data
from .load_data import persist_model
from .load_data import persist_predictions


REGIONS = ['AGSO', 'JAMSTEC', 'JAMSTEC2', 'NGA', 'NGA2', 'NGDC', 'NOAA_geodas', 'SIO', 'US_multi']


def run_cv(config):
    work_dir = config["work_dir"]
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


def run_cv_custom(config):
    def make_k_folds(cruise_level, logger):
        fold_size = int(len(cruise_level) / 2)#10)
        logger.log("fold size, {}".format(fold_size))
        total        = sum([total for _, total, _ in cruise_level])
        total_bads   = sum([bad for _, _, bad in cruise_level])
        ret = []
        num_measures = 0
        for start_index in range(0, len(cruise_level), fold_size):
            seg = cruise_level[start_index:start_index + fold_size]
            counts = sum([total for _, total, _ in seg])
            bads   = sum([bad for _, _, bad in seg])
            ret.append((num_measures, num_measures + counts, counts, bads))
        return (ret, total, total_bads)

    work_dir = config["work_dir"]
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

    valid_seg, total_counts, total_bads = make_k_folds(cruise_level, logger)
    parts = merge_array(cruise_level)
    logger.log("total number of parts: {}".format(len(parts)))
    logger.log("start loading data")
    data = read_data(parts)
    train_dataset = get_lgb_dataset(data, config["max_bin"], logger)

    for i, (start_index, end_index, valid_counts, valid_bads) in enumerate(valid_seg):
        logger.log("start training, {}".format(i))
        logger.log("stats, {}, {}, {}, {}".format(
            total_counts - valid_counts, total_bads - valid_bads, valid_counts, valid_bads))
        subset_index = list(range(start_index)) + list(range(end_index, total_counts))
        gbm = booster.train(config, train_dataset.subset(subset_index), None, True, logger)
        persist_model(work_dir, region, gbm, i)
    
        logger.log("start testing, {}, {}, {}".format(i, start_index, end_index))
        scores = booster.test(gbm, region, region, data[start_index:end_index], logger)
        persist_predictions(work_dir, region, region, data[start_index:end_index, 1:5],
                            data[start_index:end_index, 0], scores, i)
