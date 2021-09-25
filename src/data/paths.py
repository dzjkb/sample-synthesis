from os.path import join

from .fs_utils import git_root


DATA_ROOT = join(git_root(), 'data')
RAW = join(DATA_ROOT, 'raw')
INTERIM = join(DATA_ROOT, 'interim')
PROCESSED = join(DATA_ROOT, 'processed')
