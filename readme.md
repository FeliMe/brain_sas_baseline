# Uncertainty Prediction Autoencoders

This repository contains the code for "A Baseline For Unsupervised Anomaly Segmentation in Brain MR Images".

## Set-up

Clone the git project:

```
$ git clone https://github.com/FeliMe/xxx.git
```

Create a virtual environment and install the requirements:

```
$ conda create -f environment.yml
```

Activate the newly created environment:

```
$ conda activate anomaly_detection
```

## Download ROBEX and SRI ATLAS

Download and install ROBEX from https://www.nitrc.org/projects/robex
Download the SRI ATLAS from https://www.nitrc.org/projects/sri24/ and place it into DATAROOT/BrainAtlases/

## Download and pre-process Datasets

### BraTS

```
$ python download_data.py --dataset BraTS
$ python download_data.py --dataset BraTS --register
```

### MSLUB

```
$ python download_data.py --dataset MSLUB
$ python download_data.py --dataset MSLUB --skull_strip
$ python download_data.py --dataset MSLUB --register
```

### WMH

```
$ python download_data.py --dataset WMH
$ python download_data.py --dataset WMH --skull_strip
$ python download_data.py --dataset WMH --register
```

### MSSEG2015

```
$ python download_data.py --dataset MSSEG2015
$ python download_data.py --dataset MSSEG2015 --register
```

## Run the experiments (Here Experiment 1 from the paper on BraTS)

```
$ python baseline.py --test_ds BraTS --img_size 128 --slices_lower_upper 15 125
```
