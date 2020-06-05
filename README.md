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

## Typical usage

1. Create a train/test split

**IF CRUISE SEGMENTS ARE NOT CREATED FOR YOU**

All data files are uploaded to S3. Before training a model, you should create a random train-test split.
The training is implementing such that it reads the training data from a list of files (same for the testing data).
In the `config.json` configuration, you should provide a path to a text file, which contains a list of training data files.
Please refer to the [train-test-split](./train-test-split) folder for sample scripts that can be used to create train-test split and write the filenames in a text file.

**IF CRUISE SEGMENTS HAVE BEEN CREATED FOR YOU**

If you have been provided with the cruise segments, for example,
```
JAMSTEC-part000.tsv
JAMSTEC-part001.tsv
JAMSTEC-part002.tsv
...
```
You need to create three files - `training-filelist.txt`,
`validation-filelist.txt`, and
`test-filelist.txt`, each contains a list of filenames.
For example, `training-filelist.txt` should contain all cruise segments which you plan to use for training the model:
```
$ cat training-filelist.txt
JAMSTEC-part010.tsv
JAMSTEC-part095.tsv
JAMSTEC-part202.tsv
...
```
Then specify these three files to the training program in `config.json`.


2. Run training with bootstrap

The bathymetry module is implemented to train the models in different conditions (see `task_type` below). Note that 
 bootstrap is NOT implemented in this module.

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
   
3. Run testing

Testing is implemented in this module (see above).


## Missing parts

1. Bootstrap is not yet supported by this script.
2. The code does not do train/test split. The data files should be generated beforehand. All data files for training should be written to a train-files.txt. Similarly, all data files for testing should be written to a test-files.txt. Finally, the path of these two files should be included in config.json.

## Data columns

Each line in the `.tsv` data files should contain 35 columns. The meaning of the columns are as follows.

```
index name                                      Example              Description
00    lon                                	143.92639            longitude of the location
01    lat                                	-43.99727            latitude of the location
02    depth                              	-4637                the depth measured by the crew
03    sigh                               	0                    not sure what it means
04    sigd                               	-1                   set to 9999 if the measurement is marked as wrong, and otherwises if it is correct
05    SID                                	10088                Cruise ID, should not be used as features
06    pred                               	-4633                the predicted depth with the gravity model
07    ID                                 	1                    not sure what it means
08    (pred-depth)/depth                 	0.000862627          the relative difference between the measure and the prediction by the gravity model
09    d10                                	0.984607124262       average depth of the sea floor in the 10km grid
10    d20                                	0.972010395656       average depth of the sea floor in the 20km grid
11    d60                                	0.953167490781       average depth of the sea floor in the 60km grid
12    age                                	39.3518032149        Age of the oceanic plate
13    VGG                                	20.9209685261        not sure what it means
14    rate                               	1773.1538453         Plate half-spreading rate
15    sed                                	1002.69584759        Sediment thickness
16    roughness                          	23.1643450107        Seafloor roughness
17    G:T                                	0.58473963179        
18    NDP2.5m                            	352.591278239        depth minus MEDIAN_depth_in_2.5km_blocks
19    NDP5m                              	1227.86403676        depth minus MEDIAN_depth_in_5km_blocks        
20    NDP10m                             	4867.36187959        depth minus MEDIAN_depth_in_10km_blocks        
21    NDP30m                             	28191.8030442        depth minus MEDIAN_depth_in_30km_blocks        
22    STD2.5m                            	22.2740933357        STD_of_depth_in_2.5km_blocks
23    STD5m                              	42.2348361999        STD_of_depth_in_5km_blocks        
24    STD10m                             	86.5628813895        STD_of_depth_in_10km_blocks        
25    STD30m                             	188.407867174        STD_of_depth_in_30km_blocks        
26    MED2.5m                            	-17.3113             MEDIAN_depth_in_2.5km_blocks
27    MED5m                              	-30.9435             MEDIAN_depth_in_5km_blocks
28    MED10m                             	-0.9137              MEDIAN_depth_in_10km_blocks
29    MED30m                             	1.9221               MEDIAN_depth_in_30km_blocks
30    D-MED2.5m/STD2.5m                  	-0.777193
31    D-MED5m/STD5m                      	-0.732653
32    D-MED10m/STD10m                    	-0.0105553
33    D-MED30m/STD30m                    	0.0102018
34    year                               	2000                The year of the measurement
35    kind                               	G                   Device type used for measurements
```

## Program output

The training scripts will create 3 directories under the specified `base_dir` (in `config.json`).
The three directories are:

* `runtime_data`:  write a pickle file for each TSV data files, so that later we can load pickle files instead of parsing a text file, which is much faster
* `runtime_model`: output the trained model in two formats, pickle and text
* `runtime_scores`: see below

The scripts output the model prediction scores on the test examples in the `runtime_scores` directory.
The output files are named in the following format: model_{RegionUsedForTraining}_test_{RegionUsedForTesting}_scores.pkl.
The pickle files can be read as follows:

```python
with open(pickle_filename, "rb") as f:
  (features, label, scores, weights) = pickle.load(f)
```

where

* "features" is the a vector of Nx4 array, each row with 4 values, longitude, latitude, depth1, depth2 (you should not worry about the meanings of these features);
* "label" is an array of size N, which are the true labels of all examples;
* "scores" is an array of size N, which are the model predictions on all examples;
* "weights" is not being used.
