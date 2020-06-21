import json
import sys

from .load_data import init_setup
from .experiments import run_cv_custom


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        config = json.load(f)
    init_setup(config["work_dir"])
    run_cv_custom(config)
