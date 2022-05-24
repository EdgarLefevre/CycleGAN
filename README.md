# CycleGan

[![License](https://img.shields.io/github/license/EdgarLefevre/cyclegan?label=license)](https://github.com/EdgarLefevre/cyclegan/blob/main/LICENSE)

![CBiB Logo](imgs/cbib_logo.png)
-----------

Straight forward implementation of [CycleGAN](https://arxiv.org/pdf/1703.10593v7.pdf) in pytorch.


## Installation

```shell
git clone git@github.com:EdgarLefevre/CycleGAN.git
cd CycleGAN
pip install -r requirements.txt # if you use pip
conda env create -f environment.yml # if you use conda
```

## Usage
Download data: 
```shell
wget -N http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/horse2zebra.zip -O datasets/horse2zebra.zip
unzip datasets/horse2zebra.zip -d ./datasets/
rm datasets/horse2zebra.zip
```

Run training:
```shell
python -m src.train
```
