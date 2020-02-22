<h1 align='center'>:no_pedestrians: Signfeld in PyTorch</h1>

`"Give me a sign!"` - Britney Spears, Circa October 23, 1998

`"Jerry, just remember, it's not a lie if you believe it."`- George Costanza (Seinfeld), Circa July 5, 1989

## :no_bicycles: Synthetic traffic sign detection

This repository collates our efforts on building traffic sign detection model in low (to zero) sample regime (with little to no human annotations). We leverage templates of known traffic signs to train our detector. We married the ideas of [synthetic text](https://github.com/ankush-me/SynthText) & [object detection](https://github.com/LCAD-UFES/publications-tabelini-ijcnn-2019) for this work to bear fruit. We provide a pre-trained traffic sign detection model trained on [169 German Traffic sign(s)](https://github.com/moabitcoin/Signfeld/blob/master/synthetic_signs/templates/signs.md)

<p align="center">
  <img src="https://github.com/moabitcoin/Signfeld/blob/master/synthetic_signs/images/results/frame-000547.jpg">
</p>

# :feet: Table of Contents
* [Installation](#computer-installation)
  - [Conda](#snake-conda) or [Docker](#whale-docker)
* [Inference only](#tada-usage)
* [Training](#train-training)
  - [Neutral images](https://github.com/moabitcoin/Signfeld/blob/master/docs/download.md)
  - [Synthetic data gen](https://github.com/moabitcoin/Signfeld/blob/master/docs/datagen.md)
  - [Detectron training](https://github.com/moabitcoin/Signfeld/blob/master/docs/train.md)
* [Evaluation](https://github.com/moabitcoin/Signfeld/blob/master/docs/evaluate.md)

## :computer: Installation

You can either install the code in a virtual environment or docker. Use the docker if you want a reproducible environment.

### :snake: Conda

First, create a virtual environment:
```
conda create -n synth-signs
conda activate synth-signs
conda install pip
```
Then install dependencies and software:
```
pip install .            # If you only want to generate a dataset.
pip install .[inference] # Include Inference
pip install .[trainer]   # Include Model training
```

### :whale: Docker

The following two commands install and run the docker image:
```
make docker-install
make docker-run
```
### :tada: Usage

Download the pre-trained models from [here]() at `resources/models/retinanet`.

#### Detection
```
detect-synthetic-signs --images=synthetic_signs/images/test_samples/*.jpeg \
                       --label-map=resources/labels/synthetic-signs-169.yaml \
                       --target-label-map=resources/labels/gtsdb-label-to-name.yaml \
                       --config=resources/models/retinanet/config.yaml \
                       --weights=resources/models/retinanet/model_final.pth \
                       --output-dir=/tmp/signfeld
```
#### Visualisation
```
visualize-synthetic-sign-detections --images=synthetic_signs/images/test_samples/*.jpeg \
                                    --template-dir=synthetic_signs/templates \
                                    --detections=/tmp/signfeld \
                                    --destination=/tmp/signfeld-viz \
                                    --min-confidence=0.5
```
```
Disclaimer : The model was trained with zero real samples, and is known to miss signs which are obscure, small and skewed from camera axis.
```

### :rabbit2: German Traffic Signs
Trained models are include in the repository [Calzone]. Evaluated on [GTSDB](http://benchmark.ini.rub.de/?section=gtsdb&subsection=news)

| Name    | Description                                | GTSDB mAP | Remarks                                 |
| ---     | ---                                        | ---       | ---                                     |
| Calzone | Detector: RetinaNet, backbone: ResNet 50   | 67.23     | Location : [download]()                 |

### :squirrel: Contributors
- [Daniel](https://github.com/daniel-j-h)
- [Nicolai](https://www.github.com/nwojke)
- [Harsimrat](https://github.com/sandhawalia)
