import glob
import os


def get_exec_path():

    return glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), '*feff*.exe'), recursive=False)[0]


feff_exec = get_exec_path()


if __name__ == '__main__':
    print(feff_exec)
