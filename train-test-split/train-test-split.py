import os
import random


data_dir = "/path/to/the/root/of/the/data/directory"
regions = ['JAMSTEC', 'JAMSTEC2', 'NGDC', 'SIO', 'US_multi']

for region in regions:
    dirname = os.path.join(data_dir, region)
    ext = ".tsv"
    filenames = [filename for filename in os.listdir(dirname) if filename.endswith(ext)]
    random.shuffle(filenames)
    filenames = [os.path.join(dirname, filename) for filename in filenames]

    s0, s1 = int(len(filenames) * 0.15), int(len(filenames) * 0.30)
    tests, validates, trains = filenames[:s0], filenames[s0:s1], filenames[s1:]
    for name, dataset in [("test", tests), ("validate", validates), ("train", trains)]:
        with open("{}-{}.txt".format(region, name), "w") as f:
            f.write("\n".join(dataset))

