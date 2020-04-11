import json
import os
from argparse import ArgumentParser
from shutil import copy
from glob import iglob

from fs_utils import git_root


GIT_ROOT = git_root()
RAW_ROOT = GIT_ROOT + '/data/raw'
OUT_FOLDER = GIT_ROOT + '/data/interim'


def main(
    files,
    out
):
    full_paths = list(map(lambda s: f'{RAW_ROOT}/{s}', files))
    destination = f'{OUT_FOLDER}/{out}'

    if os.path.exists(destination):
        print('Output location exists - remove it pls, aborting')
        return

    os.mkdir(destination)

    for p in full_paths:
        print(f'Copying from {p}')
        for f in iglob(p):
            copy(f, destination)

    print('Done')


if __name__ == '__main__':
    parser = ArgumentParser('gather')
    parser.add_argument("path_spec", type=str, help='path to json specyfing folders to gather')
    args = parser.parse_args()
    main(**json.load(open(args.path_spec)))