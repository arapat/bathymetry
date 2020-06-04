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

*IF CRUISE SEGMENTS ARE NOT CREATED FOR YOU*
All data files are uploaded to S3. Before training a model, you should create a random train-test split.
The training is implementing such that it reads the training data from a list of files (same for the testing data).
In the `config.json` configuration, you should provide a path to a text file, which contains a list of training data files.
Please refer to the [train-test-split](./train-test-split) folder for sample scripts that can be used to create train-test split and write the filenames in a text file.

*IF CRUISE SEGMENTS ARE CREATED FOR YOU*
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
Then provide these three files to the training program in `config.json`.


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
index name                                      Example
00    lon                                	143.92639
01    lat                                	-43.99727
02    depth                              	-4637
03    sigh                               	0
04    sigd                               	-1
05    SID                                	10088
06    pred                               	-4633
07    ID                                 	1
08    (pred-depth)/depth                 	0.000862627
09    d10                                	0.984607124262
10    d20                                	0.972010395656
11    d60                                	0.953167490781
12    age                                	39.3518032149
13    VGG                                	20.9209685261
14    rate                               	1773.1538453
15    sed                                	1002.69584759
16    roughness                          	23.1643450107
17    G:T                                	0.58473963179
18    NDP2.5m                            	352.591278239
19    NDP5m                              	1227.86403676
20    NDP10m                             	4867.36187959
21    NDP30m                             	28191.8030442
22    STD2.5m                            	22.2740933357
23    STD5m                              	42.2348361999
24    STD10m                             	86.5628813895
25    STD30m                             	188.407867174
26    MED2.5m                            	-17.3113
27    MED5m                              	-30.9435
28    MED10m                             	-0.9137
29    MED30m                             	1.9221
30    D-MED2.5m/STD2.5m                  	-0.777193
31    D-MED5m/STD5m                      	-0.732653
32    D-MED10m/STD10m                    	-0.0105553
33    D-MED30m/STD30m                    	0.0102018
34    year                               	2000
35    kind                               	G
```
