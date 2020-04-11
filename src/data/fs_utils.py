from subprocess import check_output


def git_root():
    return check_output(['git', 'rev-parse', '--show-toplevel']).decode('utf-8')[:-1]
