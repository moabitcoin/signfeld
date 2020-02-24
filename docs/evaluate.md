## :trophy: Evaluation on [GTSDB](http://benchmark.ini.rub.de/?section=gtsdb&subsection=dataset)

Evaluation on the public GTSDB traffic sign detection benchmark follows the following steps: (1) Convert ground truth to an appropriate format, (2) run detector on benchmark images, (3) call evaluation software. The repository contains convenience scripts for these tasks. In the following description we store all evaluation data in a folder ``/tmp/gtsdb-evaluation``. This is not a specific requirement and you may change this location to something different.

### :raising_hand: Convert ground truth

Download [GTSDB](http://benchmark.ini.rub.de/?section=gtsdb&subsection=news) dataset. This dataset contains images in ``PPM`` format as well as the ground truth annotation in ``gt.txt``. Use the following command to convert the dataset ground truth:
```
convert-gtsdb --gtsdb-label-map=resources/labels/gtsdb-label-to-name.yaml  <path_to_gtsdb>/FullIJCNN2013/gt.txt /tmp/gtsdb-evaluation/groundtruth
```
The file [gtsdb-label-to-name.ymal](https://github.com/moabitcoin/Signfeld/blob/master/resources/labels/gtsdb-label-to-name.yaml) contains the mapping from GTSDB class ID to class name as it is used in the synthetic signs dataset.

### Run detector on benchmark

Use the script [detect-synthetic-signs](https://github.com/moabitcoin/Signfeld/blob/master/bin/detect-synthetic-signs) to generate detections on GTSDB images. Following previous example commands the call would look as follows.
```
detect-synthetic-signs --images=<path_to_gtsdb>/*.ppm \
                       --label-map=resources/labels/synthetic-signs-169.yaml \
                       --target-label-map=resources/labels/gtsdb-label-to-name.yaml \
                       --config=experiments/synthetic-signfeld-dataset/logs/config.yaml \
                       --weights=experiments/synthetic-signfeld-dataset/logs/model_final.pth \
                       --output_dir=/tmp/gtsdb-evaluation/detections-signfeld
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
python Object-Detection-Metrics-0.2/pascalvoc.py -gt /tmp/gtsdb-evaluation/groundtruth -det /tmp/gtsdb-evaluation/detections-retinanet-02-11-2019-200K -gtformat xyrb -detformat xyrb -np
```

### :tv: Visualise results.

The repository contains a [script](https://github.com/moabitcoin/Signfeld/blob/master/bin/visualize-synthetic-sign-detections) to visualise detection results. For the data generated in this section, call it as follows:
```
visualize-synthetic-sign-detections --images=<path to GTSB folder>/*.ppm \
                                    --template-dir=synthetic-signs/templates \
                                    --detections=/tmp/gtsdb-evaluation/detections-retinanet-02-11-2019-200K \
                                    --destination=/tmp/gtsdb-evaluation/images-retinanet-02-11-2019-200K \
                                    --min-confidence=0.5
```
