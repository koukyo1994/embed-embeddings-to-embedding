import re
import sys
import time
import logging

from pathlib import Path
from datetime import datetime as dt
from contextlib import contextmanager


@contextmanager
def timer(name, logger=None):
    t0 = time.time()
    yield
    msg = f"[{name}] done in {time.time()-t0:.0f} s"
    if logger:
        logger.info(msg)
    else:
        print(msg)


def get_logger(name="Main", tag="exp", log_dir="log/"):
    log_path = Path(log_dir)
    path = log_path / tag
    path.mkdir(exist_ok=True, parents=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(
        path / (dt.now().strftime("%Y-%m-%d-%H-%M-%S") + ".log"))
    sh = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s")

    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def check_format(path):
    if "bin" in path:
        return True
    else:
        return False


def count_words_in_expanded_words(x: list, expanded: set):
    x_set = set(x)
    intersection = x_set.intersection(expanded)
    count = 0
    for word in intersection:
        count += len(re.findall(re.escape(word), " ".join(x)))
    return count
