import os
from tqdm import tqdm

from .logger import LOGGER

RANK = int(os.getenv("RANK", -1))
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))
VERBOSE = str(os.getenv("VERBOSE", True)).lower() == "true"  # global verbose mode
TQDM_BAR_FORMAT = "{l_bar}{bar:10}{r_bar}" if VERBOSE else None  # tqdm bar format


class TQDM(tqdm):

    def __init__(self, *args, **kwargs):

        kwargs["disable"] = not VERBOSE or kwargs.get(
            "disable", False
        )  # logical 'and' with default value if passed
        kwargs.setdefault(
            "bar_format", TQDM_BAR_FORMAT
        )  # override default value if passed
        super().__init__(*args, **kwargs)


def is_dir_writeable(dir_path):
    return os.access(str(dir_path), os.W_OK)
