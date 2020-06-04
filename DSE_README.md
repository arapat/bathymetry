# Readme file for homework 4 of DSC291

The data files are uploaded to `s3://bathymetry`.
There are six directories, five of them are named after the regions that collected the data, the last one is `test/`,
which contains testing examples without labels given.

* Each of training data directories (except JAMSTEC2, which contains less) contain 500 files.
* The files in each directory are in the tsv format (tab-seperated values).
* Each file contains 100K consecutive measurements from a single cruise.
* The `test/` directory contains 108 files. The labels of these examples have been removed.
Specifically, they have been set to -1 regardless of the true labels.

## Scripts in this repository

The scripts in this repository read the TSV files, and parse them to generate the features and labels,
and train a boosted tree using `lightgbm`.

* `__main__.py`: main function, also parse commandline arguments
* `booster.py`: actual code to call LightGBM
* `common.py`:	common miscellaneous functions, e.g. logging
* `load_data.py`: loading the tsv/pickle training and testing data.
If the input format is tsv, it will be written to disk in pickle files so that next time the data loading would be faster.
* `test.py`: template code to be called by "__main__.py" proper functions for testing. It outputs a pickle file that
contains scores in addition to some meta information about examples, e.g. cruise ID, longitute, latitude
* `train.py`: template code to be called by "__main__.py" proper functions for training.
* `config.json`: config such as the input data path, and the directory to write the models

## Typical usage

Let's assume the data is downloaded into a directory at `/home/user/data'.

### To train a model on each region, and test on each region

1. Do the random train/test split

* For each region, read all the filenames under this region, shuffle them, and split the files for training and testing,
by generating two files per region: `training-files.txt` that list all files that will be used for training,
and `testing-files.txt` that lists all files that will be for testing.
* Updated `config.json` with the path to `training-files.txt` and `testing-files.txt`
* Run the following command:
```
$ python -m bathymetry tsv train <config_path>
```


