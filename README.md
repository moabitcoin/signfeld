`"Give me a sign!"` - Britney Spears, Circa October 23, 1998

## :no_bicycles: Traffic sign detection with synthetic data

This repository collates our efforts on generating traffic sign detection model in low sample regimes (with little to no human annotations). We leverage templates of known traffic signs to train our detector. We married two ideas of synthetic text & object detection for this work to bear fruit.

## :computer: Installation

You can either install the code in a virtual environment or docker. Use the docker if you want a reproducable environment.

### :snake: Using Conda (virtual environment)

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

### :whale: Using docker

The following two commands install and run the docker image:
```
make docker-install
make docker-run
```

## :no_pedestrians: Traffic sign templates

A subset of the near complete [list](https://de.wikipedia.org/wiki/Bildtafel_der_Verkehrszeichen_in_der_Bundesrepublik_Deutschland_seit_2017) of German traffic sign(s) is of interest to us. More specifically [these](https://github.com/moabitcoin/Signfeld/tree/master/synthetic_signs/templates). These signs form a subset (signs of interest) which formalise [`turn-restrictions`](https://wiki.openstreetmap.org/wiki/Relation:restriction) in OSM. Using the templates for the signs of interest we build a synthetic training set following the idea presented in [IJCN2019](https://github.com/LCAD-UFES/publications-tabelini-ijcnn-2019) and [CVPR 2016](https://github.com/ankush-me/SynthText).

## :innocent: Neutral images
For the next steps we'd need images which do not contain any known traffic signs. We leverage [OpenimagesV5](https://storage.googleapis.com/openimages/web/index.html) and build a neutral image set by querying for [`Building`](https://storage.googleapis.com/openimages/web/visualizer/index.html?set=train&type=detection&c=%2Fm%2F0cgh4) and filtering out images containing [`traffic signs`](https://storage.googleapis.com/openimages/web/visualizer/index.html?set=train&type=detection&c=%2Fm%2F01mqdt), referred below as `Building_without_signs.list`. Please refer to instructions on [Figure8.](https://www.figure-eight.com/dataset/open-images-annotated-with-bounding-boxes/). Please download the following file(s) and place them under `synthetic_signs/external/lists` and zips:
- [train-annotations-bbox.csv](https://datasets.figure-eight.com/figure_eight_datasets/open-images/train-annotations-bbox.csv)
- [validation-annotations-bbox.csv](https://datasets.figure-eight.com/figure_eight_datasets/open-images/validation-annotations-bbox.csv)
- [test-annotations-bbox.csv](https://datasets.figure-eight.com/figure_eight_datasets/open-images/test-annotations-bbox.csv)
- [train_01.zip](https://datasets.figure-eight.com/figure_eight_datasets/open-images/zip_files_copy/train_01.zip)

### Neutral image list gen

```
python scripts/download-openimages-v5.py --help
usage: download-openimages-v5.py [-h] --mode MODE --classes CLASSES
                                 [--nthreads NTHREADS] --dest DEST --csvs CSVS
                                 [--limit LIMIT] [--occluded] [--truncated]
                                 [--groupOf] [--depiction] [--inside]

Download Class specific images from OpenImagesV5

optional arguments:
  -h, --help           show this help message and exit
  --mode MODE          Dataset category - train, validation or test
  --classes CLASSES    Names of object classes to be downloaded
  --nthreads NTHREADS  Number of threads to use
  --dest DEST          Destination directory
  --csvs CSVS          CSV file(s) directory
  --limit LIMIT        Cap downloaded files to limit
```

```
python scripts/download-openimages-v5.py --classes 'Building,Traffic_sign,Traffic_light' --mode train --dest <Directory for OpenImagesV5> --csvs <path to Signfeld>/synthetic_signs/external/lists
```

## :factory: Generate synthetic data

Use [generate-synthetic-dataset](bin/generate-synthetic-dataset) to generate a synthetic sign dataset:
```
usage: generate-synthetic-dataset [-h] --backgrounds BACKGROUNDS
                                   --templates-path TEMPLATES_PATH
                                   --augmentations AUGMENTATIONS
                                   [--distractors-path DISTRACTORS_PATH]
                                   [--random-distractors RANDOM_DISTRACTORS]
                                   --out-path OUT_PATH --max-images
                                   MAX_IMAGES [--n JOBS]
                                   [--max-template-size MAX_TEMPLATE_SIZE]
                                   [--min-template-size MIN_TEMPLATE_SIZE]
                                   [--background-size BACKGROUND_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --backgrounds BACKGROUNDS
                        Path to the directory containing background images to
                        be used (or file list).
  --templates-path TEMPLATES_PATH
                        Path (or file list) of templates.
  --augmentations AUGMENTATIONS
                        Path to augmentation configuration file.
  --distractors-path DISTRACTORS_PATH
                        Path (or file list) of distractors.
  --random-distractors RANDOM_DISTRACTORS
                        Generate this many random distractors for each
                        template.
  --out-path OUT_PATH   Path to the directory to save the generated images to.
  --max-images MAX_IMAGES
                        Number of images to be generated.
  --n JOBS              Maximum number of parallel processes.
  --max-template-size MAX_TEMPLATE_SIZE
                        Maximum template size.
  --min-template-size MIN_TEMPLATE_SIZE
                        Minimum template size.
  --background-size BACKGROUND_SIZE
                        If not None (or empty string), image shape
                        'height, width'
```
The following example generates a dataset of 2M images. The file [augmentations.yaml](resources/configs/augmentations.yaml) specifies augmentation parameters (geometric template distortion, blending methods, etc.). Refer to the documentation of ``generate_task_args()`` in [synthetic_signs.dataset_generator](synthetic_signs/dataset_generator.py#268) for a parameter description.
```
generate-synthetic-dataset --backgrounds=lists/Building_without_signs.list \
                           --templates-path=synthetic_signs/templates \
                           --out-path=experiments/dataset-02-11-2019-200K \
                           --n=16 \
                           --max-images=200000 \
                           --augmentations=resources/configs/augmentations.yaml
```

### :traffic_light: Create training and validation splits

Special care must be taken when splitting the data into training and validation sets because the dataset generator creates multiple images of the same synthetic sign composition (using a different blending mode for each image). Simply randomizing and splitting the data could cause the same composition to end up in both sets. The script [generate-train-val-splits](bin/generate-train-val-splits) makes sure this doesn't happen:
```
usage: generate-train-val-splits [-h] [-s S] csv_datafile

positional arguments:
  csv_datafile  Input dataset 'multiclass.csv' file.

optional arguments:
  -h, --help    show this help message and exit
  -s S          Percentage of data to split off for validation
```
Based on example dataset generation command from the previous section, you can generate separate training and validtion splits using the following command:
```
generate-train-vali-splits -s 20 /nas/team-space/experiments/turn_restrictions/synthetic-signs/dataset-02-11-2019-200K/multiclass.csv
```
This command splits off 20% (specified by ``-s 20``) of the dataset for validation and uses the rest for training. The following two files contain the data splits:
```
<exp_folder>/multiclass_train.csv
<exp_folder>/multiclass_valid.csv
```

## :bullettrain_side: Train a model

Use the script [train-synthetic-sign-detector](bin/train-synthetic-sign-detector) to train a model:
```
usage: train-synthetic-sign-detector [-h] [--config-file FILE] [--resume]
                                     [--eval-only] [--num-gpus NUM_GPUS]
                                     [--num-machines NUM_MACHINES]
                                     [--machine-rank MACHINE_RANK]
                                     [--dist-url DIST_URL]
                                     [--label-map LABEL_MAP] --train-csv
                                     TRAIN_CSV [--valid-csv VALID_CSV]
                                     ...
Detectron2 Training

positional arguments:
  opts                  Modify config options using the command-line

optional arguments:
  -h, --help            show this help message and exit
  --config-file FILE    path to config file
  --resume              whether to attempt to resume from the checkpoint
                        directory
  --eval-only           perform evaluation only
  --num-gpus NUM_GPUS   number of gpus *per machine*
  --num-machines NUM_MACHINES
  --machine-rank MACHINE_RANK
                        the rank of this machine (unique per machine)
  --dist-url DIST_URL
  --label-map LABEL_MAP
                        Label map in YAML format which maps from category ID
                        to name.
  --train-csv TRAIN_CSV
                        Path to training data CSV file.
  --valid-csv VALID_CSV
                        Optional path to validation data CSV file.
  --image-width IMAGE_WIDTH
                        Image width (optional, used to speed up dataset
                        processing).
  --image-height IMAGE_HEIGHT
                        Image height (optional, used to speed up dataset
                        processing).
```
Following previous sections, we may use the following commands to train a RetinaNet detector on the generated data:
```
train-synthetic-sign-detector --config-file resources/configs/signs_169_retinanet_R_50_FPN_3x.yaml \
                              --label-map=resources/labels/synthetic-signs-169.yaml \
                              --train-csv=<exp_folder>/multiclass_train.csv \
                              --valid-csv=<exp_folder>/multiclass_valid.csv \
                              OUTPUT_DIR data-retinanet-02-11-2019-200K
```
The configuration file [signs_169_retinanet_R_50_FPN_3x.yaml](resources/configs/signs_169_retinanet_R_50_FPN_3x.yaml) contains model hyperparameters and a configuration of the training process. Note that you must leave the DATASETS section of this configuration file empty when you are training with the provided script. It will be filled out by the trainer itself based on the provided ``--train-csv`` and ``--valid-csv`` arguments. The label map [synthetic-signs-169.yaml](resources/labels/synthetic-signs-169.yaml) contains the mapping from class ID to class name. If you change the template set you can re-generate it using [generate-label-map](bin/generate-label-map).

Training can be monitored using tensboard. Following previous example commands:
```
tensorboard --logdir data-retinanet-02-11-2019-200K
```

## :trophy: Evaluation on [GTSDB](http://benchmark.ini.rub.de/?section=gtsdb&subsection=dataset)

Evaluation on the public GTSDB traffic sign detection benchmark follows the following steps: (1) Convert ground truth to an appropriate format, (2) run detector on benchmark images, (3) call evaluation software. The repository contains convenience scripts for these tasks. In the following description we store all evaluation data in a folder ``/tmp/gtsdb-evaluation``. This is not a specific requirement and you may change this location to something different.

### :gem: Convert ground truth

The GTSDB dataset is stored on our NAS in ``/nas/3rd_party/FullIJCNN2013/``. This folder contains images in ``PPM`` format as well as the ground truth annotation in ``gt.txt``. Use the following command to convert the dataset ground truth:
```
convert-gtsdb \
    --gtsdb-label-map=resources/labels/gtsdb-label-to-name.yaml \
    /nas/3rd_party/FullIJCNN2013/gt.txt \
    /tmp/gtsdb-evaluation/groundtruth
```
The file [gtsdb-label-to-name.ymal](resources/labels/gtsdb-label-to-name.yaml) contains the mapping from GTSDB class ID to class name as it is used in the synthetic signs dataset.

### Run detector on benchmark

Use the script [detect-synthetic-signs](bin/detect-synthetic-signs) to generate detections on GTSDB images. Following previous example commands the call would look as follows.
```
detect-synthetic-signs --images=<path to GTSB folder>/*.ppm \
                       --label-map=resources/labels/synthetic-signs-169.yaml \
                       --target-label-map=resources/labels/gtsdb-label-to-name.yaml \
                       --config=data-retinanet-02-11-2019-200K/config.yaml \
                       --weights=data-retinanet-02-11-2019-200K/model_final.pth \
                       --output_dir=/tmp/gtsdb-evaluation/detections-retinanet-02-11-2019-200K
```
In addition to the detectors configuration (``--config``), weights (``--weights``) and label map (``--label-map``), we also supply the label map of the GTSDB dataset (``--target-label-map``). By supplying a target label map, the script suppresses all detections which are of a class which is not part of the GTSDB dataset. Otherwise, they would be counted as false alarms during evaluation.

### :game_die: Evaluate model

First, download and install the evaluation software:
```
wget https://github.com/rafaelpadilla/Object-Detection-Metrics/archive/v0.2.zip -O Object-Detection-Metrics-v0.2.zip
unzip Object-Detection-Metrics-v0.2.zip
```
Now call the software to evaluate detections against ground truth. The script prints per-class AP and mean average precision according to Pascal VOC metrics.
```
python Object-Detection-Metrics-0.2/pascalvoc.py \
    -gt /tmp/gtsdb-evaluation/groundtruth \
    -det /tmp/gtsdb-evaluation/detections-retinanet-02-11-2019-200K \
    -gtformat xyrb \
    -detformat xyrb \
    -np
```

### :tv: Visualize results.

The repository contains a [script](bin/visualize-synthetic-sign-detections) to visualise detection results. For the data generated in this section, call it as follows:
```
visualize-synthetic-sign-detections --images=<path to GTSB folder>/*.ppm \
                                    --template-dir=synthetic-signs/templates \
                                    --detections=/tmp/gtsdb-evaluation/detections-retinanet-02-11-2019-200K \
                                    --destination=/tmp/gtsdb-evaluation/images-retinanet-02-11-2019-200K \
                                    --min-confidence=0.5
```

## :rabbit2: Model Zoo

Trained models are include in the repository [Calzone].

| Name    | Description                                | GTSDB mAP | Remarks                               |
| ---     | ---                                        | ---       | ---                                   |
| Calzone | Detector: RetinaNet, backbone: ResNet 50   | 67.23     | Location : pre-trained-models/Calzone |
