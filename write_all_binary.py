from load_data import read_data_from_text, write_data_to_binary, get_binary_filename
from common import Logger

#-------------------------------------------------------------
base_dir = '/swot2/features/tsv_all'
training_files = '/swot2/features/tsv_all/US_multi2.testing_files.txt'

logger = Logger()
logfile = "write-bin.log"
logger.set_file_handle(logfile)

with open(training_files) as f:
    filepaths = f.readlines()

for filename in filepaths:
    filename = filename.strip()
    bin_filename = get_binary_filename(base_dir, '', filename)

    try:
        features, labels, weights, incorrect_cols = read_data_from_text(filename)
    except Exception as err:
        logger.log("Failed to load {}, is_read_text, {}, Error, {}".format(
            filename, 0, err))
        continue
    
    logger.log("Writing file: {}".format(bin_filename))
    logger.log("To write {} examples".format(len(labels)))

    if type(features) is not list:
        features = features.tolist()
        labels = labels.tolist()

    write_data_to_binary(
            base_dir, 0, features, labels, weights.tolist(),
            bin_filename, '')
