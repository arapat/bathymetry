import os
import random


data_dir = "/cryosat3/btozer/CREATE_ML_FEATURES/tsv_all"
write_dir = "/cryosat3/jalafate/chunk_tsv"
regions = ['JAMSTEC', 'JAMSTEC2', 'NGDC', 'SIO', 'US_multi']

os.mkdir(write_dir)
for region in regions:
    os.mkdir(os.path.join(write_dir, region))

    dirname = os.path.join(data_dir, region)
    ext = ".tsv"
    filenames = [filename for filename in os.listdir(dirname) if filename.endswith(ext)]

    chunk_size = 100000
    for filename in filenames:
        basename = filename[:-4]
        path = os.path.join(dirname, filename)
        with open(path) as f:
            data = f.readlines()

        num_line = len(data)
        cursor = 0
        part = 0
        while cursor < num_line:
            start, end = cursor, cursor + chunk_size
            if (num_line - end) * 2 < chunk_size:
                end = num_line
            cursor = end

            part_filename = basename + ".part{}.tsv".format(part)
            part += 1
            write_loc = os.path.join(write_dir, region, part_filename)
            with open(write_loc, "w") as f:
                f.write("".join(data[start:end]))

