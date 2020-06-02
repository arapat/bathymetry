# Train and test boosted tree on the bathymetry data

## Files:

* `__init__.py`: for python path
* `__main__.py`: main function, also parse commandline arguments
* `booster.py`: actual code to call LightGBM
* `common.py`:	common miscellaneous functions, e.g. logging
* `load_data.py`: loading the tsv/pickle training and testing data.
If the input format is tsv, it will be written to disk in pickle files so that next time the data loading would be faster.
* `test.py`: template code to be called by "__main__.py" proper functions for testing. It outputs a pickle file that
contains scores in addition to some meta information about examples, e.g. cruise ID, longitute, latitude
* `train.py`: template code to be called by "__main__.py" proper functions for training.
* `config.json`: config such as the input data path, and the directory to write the models

## Typical execution

```
python bathymetry <data_type> <task_type> <config_path>
```

* data_type: "tsv" or "pickle", if you have pickle file already written to disk, choose "pickle", otherwise choose "tsv"
* task_type:
   * "train": training models for each research institution (generate as many models as there are institutions)
   * "test-cross": cross test the trained models on the testing data from all other research institutions (if there are n models and n research institutions, there will be (n*n) tests in total
   * "train-all": train a model using all available data from all institutions
   * "test-all": test the model trained on all data on the dataset from research institutions (test n times)
   * "train-instances": training a model using a data that is splitted on the instance level (ignore for now)
   * "test-instances": testing a model using a test set that was splitted on the instance level (ignore for now)
   
Bootstrap is not yet supported by this script.
