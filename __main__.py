import json
import os
import ray
import sys

from .common import Logger
from .load_data import init_setup
from .train import run_training
from .train import run_training_all
from .train import run_training_specific_file
from .train import run_training_n_times
from .test import get_all_data
from .test import run_testing
from .test import run_testing_specific_file


regions = ['AGSO', 'JAMSTEC', 'JAMSTEC2', 'NGA', 'NGA2', 'NGDC', 'NOAA_geodas', 'SIO', 'US_multi']
#regions = ['NGDC','US_multi']
#regions = ['TEST-ATL','TEST-PAC']
param1 = ["tsv", "pickle"]
param2 = ["train", "train-all", "test-self", "test-cross", "test-all",
          "train-instances", "test-instances", "train-random"]
usage_msg = "Usage: python -m bathymetry <{}> <{}> <config_path>".format("|".join(param1), "|".join(param2))


@ray.remote
def run_training_one_region(region):
    logger = Logger()
    logfile = os.path.join(config["base_dir"], "training_log_{}.log".format(region))
    logger.set_file_handle(logfile)
    run_training(config, [region], is_read_text, logger)


def run_training_all_regions(regions):
    logger = Logger()
    logfile = os.path.join(config["base_dir"], "training_log_all.log")
    logger.set_file_handle(logfile)
    run_training_all(config, regions, is_read_text, logger)


@ray.remote
def run_test(model_name, test_regions, task, data=None):
    logger = Logger()
    logfile = os.path.join(config["base_dir"], "testing_log_{}.log".format(model_name))
    logger.set_file_handle(logfile)
    logger.log("eval, model_region, data_region, model_size, loss, auprc, auroc, accuracy")
    task = task.split('-')[1]
    run_testing(config, [model_name], test_regions, is_read_text, task, logger, all_data=data)


@ray.remote
def run_training_instances(regions, region_name):
    DATA_DIR = "data_pkl"
    logger = Logger()
    logfile = os.path.join(config["base_dir"], "training_inst_log_{}.log".format(region_name))
    logger.set_file_handle(logfile)
    dirname = os.path.join(config["base_dir"], DATA_DIR)
    filenames = [
        os.path.join(dirname, "training-instances_{}.pkl".format(region)) for region in regions]
    run_training_specific_file(filenames, region_name, config, logger)


@ray.remote
def run_testing_instances(model_name, regions):
    DATA_DIR = "data_pkl"
    logger = Logger()
    logfile = os.path.join(config["base_dir"], "testing_inst_log_{}.log".format(model_name))
    logger.set_file_handle(logfile)
    dirname = os.path.join(config["base_dir"], DATA_DIR)
    filenames = [
        os.path.join(dirname, "testing-instances_{}.pkl".format(region)) for region in regions]
    for test_region_name, filename in zip(regions, filenames):
        run_testing_specific_file(model_name, [filename], test_region_name, config, logger)
    run_testing_specific_file(model_name, filenames, "all", config, logger)

@ray.remote
def run_training_random(regions):
    logger = Logger()
    logfile = os.path.join(config["base_dir"], "training_log_all.log")
    logger.set_file_handle(logfile)
    run_training_n_times(config, regions, is_read_text, logger)

def get_data():
    logger = Logger()
    logfile = os.path.join(config["base_dir"], "all-data-loading.log")
    logger.set_file_handle(logfile)
    logger.log("start loading all datasets")
    with open(config["testing_files"]) as f:
        all_testing_files = f.readlines()
    data = get_all_data(config["base_dir"], all_testing_files, regions, is_read_text, logger)
    logger.log("finished loading all testing data")
    return data


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print(usage_msg)
        sys.exit(1)
    if sys.argv[1].lower() not in param1 or sys.argv[2].lower() not in param2:
        print("Cannot understand the parameters.")
        print(usage_msg)
        sys.exit(1)
    with open(sys.argv[3]) as f:
        config = json.load(f)
    config["base_dir"] = os.path.expanduser(config["base_dir"])
    is_read_text = (sys.argv[1].lower() == "tsv")
    init_setup(config["base_dir"])
    task = sys.argv[2].lower()

    ray.init(num_cpus=10)
    result_ids = []
    if task == "train":
        for region in regions:
            result_ids.append(run_training_one_region.remote(region))
    elif task == "train-all":
        run_training_all_regions(regions)
    elif task == "train-random":
        result_ids.append(run_training_random.remote(regions))
    elif task == "test-cross":
        for region in regions:
            result_ids.append(run_test.remote(region, regions, task))
    elif task == "test-all":
        result_ids.append(run_test.remote("all", regions, "test-cross"))
    elif task == "train-instances":
        for region in regions:
            result_ids.append(run_training_instances.remote([region], region))
        result_ids.append(run_training_instances.remote(regions, "all"))
    elif task == "test-instances":
        for region in regions:
            result_ids.append(run_testing_instances.remote(region, regions))
        result_ids.append(run_testing_instances.remote("all", regions))
    elif task == "test-self":
        for region in regions:
            result_ids.append(run_test.remote(region, [region], task))
    elif task == "test-usm2":
        for region in regions:
            #result_ids.append(run_test(region, ['US_multi2'], "test-cross"))
            result_ids.append(run_test.remote(region, ['US_multi2'], "test-cross"))
    else:
        assert(False)
    results = ray.get(result_ids)
