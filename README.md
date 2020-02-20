`"Give me a sign!"` - Britney Spears, Circa October 23, 1998

`"Jerry, just remember, it's not a lie if you believe it."`- George Costanza (Seinfeld), Circa July 5, 1989

## :no_bicycles: Traffic sign detection with synthetic data

This repository collates our efforts on generating traffic sign detection model in low sample regimes (with little to no human annotations). We leverage templates of known traffic signs to train our detector. We married two ideas of synthetic text & object detection for this work to bear fruit.

# Table of Contents
* [Installation](#computer-installation)
  - [Conda](#snake-conda) or [Docker](#whale-docker)
* [Inference only](#tada-usage)
* [Training](#train-training)
  - [Neutral images](https://github.com/moabitcoin/Signfeld/blob/master/docs/download.md)
  - [Synthetic data gen](https://github.com/moabitcoin/Signfeld/blob/master/docs/datagen.md)
  - [Detectron training](https://github.com/moabitcoin/Signfeld/blob/master/docs/train.md)
* [Evaluation](https://github.com/moabitcoin/Signfeld/blob/master/docs/evaluate.md)
## :computer: Installation

You can either install the code in a virtual environment or docker. Use the docker if you want a reproducable environment.

### :snake: Conda

First, create a virtual environment:
```
conda create -n synth-signs
conda activate synth-signs
conda install pip
```
Then install dependencies and software:
```
pip install .           # If you only want to generate a dataset.
pip install .[trainer]  # If you also want to train a model.
```

### :whale: Docker

The following two commands install and run the docker image:
```
make docker-install
make docker-run
```
### :tada: Usage

### :train: Training

## :rabbit2: Pre-trained German Traffic Sign Model
Trained models are include in the repository [Calzone].

| Name    | Description                                | GTSDB mAP | Remarks                               |
| ---     | ---                                        | ---       | ---                                   |
| Calzone | Detector: RetinaNet, backbone: ResNet 50   | 67.23     | Location : pre-trained-models/Calzone |
