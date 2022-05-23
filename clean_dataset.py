import os

import numpy as np
import skimage.io as io


def list_files_path(path: str) -> list[str]:
    """
    List files from a path.

    :param path: Folder path
    :type path: str
    :return: A list containing all files in the folder
    :rtype: List
    """
    return [path + f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


if __name__ == "__main__":
    for letter in ["A", "B"]:
        img_list = list_files_path(f"dataset/horse2zebra/train{letter}/")
        for path in img_list:
            print(path)
            img = io.imread(path)
            if np.shape(img) != (256, 256, 3):
                print("suppr")
                os.system(f"rm {path}")
