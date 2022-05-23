# -*- coding: utf-8 -*-
import argparse
import os
import re
from typing import Tuple

import numpy as np
import skimage.io as io
import torch
import torch.utils.data as tud


def list_files_path(path: str) -> list[str]:
    """
    List files from a path.

    :param path: Folder path
    :type path: str
    :return: A list containing all files in the folder
    :rtype: List
    """
    return sorted_alphanumeric(
        [path + f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    )


def sorted_alphanumeric(data: list[str]) -> list[str]:
    """
    Sort function.

    :param data: str list
    :type data: List
    :return: Sorted list
    :rtype: List
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()  # noqa
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]  # noqa
    return sorted(data, key=alphanum_key)


def get_datasets(path_imgs: str, path_labels: str) -> Tuple[tud.Dataset, tud.Dataset]:
    """
    Get the datasets for the training and the validation set.

    :param path_imgs: Path to the images
    :type path_imgs: str
    :param path_labels: Path to the labels
    :type path_labels: str
    :param args: Arguments
    :type args: argparse.Namespace
    :return: Dictionary of the datasets
    :rtype: dict[str, DataLoader[Any]]
    """
    img_path_list = list_files_path(path_imgs)
    label_path_list = list_files_path(path_labels)
    # not good if we need to do metrics

    dataset_train = Cyclegan_dataset(1, img_path_list, label_path_list)
    dataset_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=5,
        shuffle=True,
        num_workers=4,
    )
    return dataset_train


class Cyclegan_dataset(tud.Dataset):
    """
    Dataset for the DeepMeta model.
    """

    def __init__(
        self,
        batch_size: int,
        input_imgh_paths: list[str],
        input_imgz_paths: list[str],
    ) -> None:
        """
        Initialize the dataset.

        :param batch_size: Batch size
        :type batch_size: int
        :param input_imgh_paths: Path to the images A
        :type input_imgh_paths: list[str]
        :param input_imgz_paths: Path to the images B
        :type input_imgz_paths: list[str]
        """
        self.batch_size = batch_size
        self.img_size = 256
        self.input_imgh_paths = input_imgh_paths
        self.input_imgz_paths = input_imgz_paths
        print(f"Nb of images : {len(input_imgh_paths)}")

    def __len__(self) -> int:
        return len(self.input_imgh_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns tuple (input, target) correspond to batch #idx.
        """
        try:
            path_imgh = self.input_imgh_paths[idx]
            imgh = np.array(io.imread(path_imgh)) / 255
            imgh = torch.Tensor(np.transpose(imgh, (2, 0, 1)))
            path_imgz = self.input_imgz_paths[idx]
            imgz = np.array(io.imread(path_imgz)) / 255
            imgz = torch.Tensor(np.transpose(imgz, (2, 0, 1)))
        except:
            print(f"Error with {path_imgh}")
            print(f"Error with {path_imgz}")
            print(imgz.shape)
            print(imgh.shape)
            raise
        return imgh, imgz
