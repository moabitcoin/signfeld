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
Based on example dataset generation command from the previous section, you can generate separate training and validation splits using the following command:
```
generate-train-vali-splits -s 20 experiments/synthetic-signfeld-dataset/multiclass.csv
```
This command splits off 20% (specified by ``-s 20``) of the dataset for validation and uses the rest for training. The following two files contain the data splits:
```
ls experiments/synthetic-signfeld-dataset/multiclass_train.csv
ls experiments/synthetic-signfeld-dataset/multiclass_valid.csv
```

## :bullettrain_side: Train a model

Use the script [train-synthetic-sign-detector](bin/train-synthetic-sign-detector) to train a model:
```
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
                              --label-map=resources/labels/labels-DE-169.yaml \
                              --train-csv=experiments/synthetic-signfeld-dataset/multiclass_train.csv \
                              --valid-csv=experiments/synthetic-signfeld-dataset/multiclass_valid.csv \
                              OUTPUT_DIR experiments/synthetic-signfeld-dataset/logs
```
The configuration file [signs_169_retinanet_R_50_FPN_3x.yaml](resources/configs/signs_169_retinanet_R_50_FPN_3x.yaml) contains model hyperparameters and a configuration of the training process. Note that you must leave the DATASETS section of this configuration file empty when you are training with the provided script. It will be filled out by the trainer itself based on the provided ``--train-csv`` and ``--valid-csv`` arguments. The label map [synthetic-signs-169.yaml](resources/labels/synthetic-signs-169.yaml) contains the mapping from class ID to class name. If you change the template set you can re-generate it using [generate-label-map](bin/generate-label-map).

Training can be monitored using tensorboard. Following previous example commands:
```
tensorboard --logdir experiments/synthetic-signfeld-dataset/logs
```
