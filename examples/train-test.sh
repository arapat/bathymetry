# Generate the data file lists for training and testing

# Change this to actual directory where the data is downloaded
export DATA_DIR=/home/user/data

# Shuffle segments and create a training & testing set
ls -d $DATA_DIR/SIO/* | sort -R > all_files.txt
split -l 400 all_files.txt
mv xaa training_files.txt
mv xab testing_files.txt
touch validation_files.txt
mkdir workspace
python -m bathymetry tsv train config.json
python -m bathymetry tsv test-self ./config.json

# the logs are written under ./workspace directory
grep "eval" workspace/testing_log_SIO.log
# The output should look like
# [0.00019] eval, model_region, data_region, model_size, loss, auprc, auroc, accuracy
# [5.34235] eval, SIO, SIO, 10, 0.6154540919310048, 0.953968181103632, 0.3153308078728423, 0.9576656798709334
