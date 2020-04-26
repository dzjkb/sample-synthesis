import logging
from datetime import datetime

from ..data.fs_utils import git_root


LOG_DIR = f'{git_root()}/logs'


def get_logger(name, levelstr):
    lg = logging.getLogger(name)
    lg.setLevel(logging.DEBUG)

    day = datetime.now().strftime('%Y-%m-%d')

    logfile = f'{LOG_DIR}/{name}_{day}.logs'
    fh = logging.FileHandler(logfile, mode='a')
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    level = getattr(logging, levelstr)
    ch.setLevel(level)

    formatter = logging.Formatter('[%(asctime)s %(name)s] %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    lg.addHandler(fh)
    lg.addHandler(ch)

    return lg