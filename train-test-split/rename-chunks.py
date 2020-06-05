import os
import random


data_dir = "/cryosat3/jalafate/chunk_tsv"
regions = ['JAMSTEC', 'JAMSTEC2', 'NGDC', 'SIO', 'US_multi']

rename_records = []
for region in regions:
    dirname = os.path.join(data_dir, region)
    ext = ".tsv"
    filenames = [filename for filename in os.listdir(dirname) if filename.endswith(ext)]
    random.shuffle(filenames)

    for i, filename in enumerate(filenames):
        path = os.path.join(dirname, filename)
        if i >= 500:
            os.remove(path)
            rename_records.append("{}\t{}\t{}".format(region, filename, "removed"))
        else:
            new_name = "{}-part{:05d}.tsv".format(region, i)
            new_path = os.path.join(dirname, new_name)
            os.rename(path, new_path)
            rename_records.append("{}\t{}\t{}".format(region, filename, new_name))

with open("rename-records.tsv", "w") as f:
    f.write('\n'.join(rename_records))

