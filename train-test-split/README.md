# Perform train/test split

There are three ways you can create the train/test splits over the bathymetry data:
by single sample, by whole tours, and by tour segments.

In this directory, you can find scripts that create the split by whole tours and by tour segments.

## Create splits by whole tour

You can create splits by whole tour using [train-test-split.py](train-test-split/train-test-split.py). Each orginal data file contain all measurements from
one cruise/tour. This script simply assigns the data file to training, validation, or testing.

## Create splits by tour segments

After you train the models using the splits generated in the previous step, the data should have been written to disk
in pickle format so that they can be loaded faster in future.
The notebook [Chunk-data.ipynb](train-test-split/Chunk-data.ipynb) reads these pickle files, chunk them into small segments (of size 10k),
shuffle these chunks, and create a new train/test split.
