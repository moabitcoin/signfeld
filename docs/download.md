## :innocent: Neutral images
For the next steps we'd need images which do not contain any known traffic signs. We leverage [OpenimagesV5](https://storage.googleapis.com/openimages/web/index.html) and build a neutral image set by querying for [`Building`](https://storage.googleapis.com/openimages/web/visualizer/index.html?set=train&type=detection&c=%2Fm%2F0cgh4) and filtering out images containing [`traffic signs`](https://storage.googleapis.com/openimages/web/visualizer/index.html?set=train&type=detection&c=%2Fm%2F01mqdt), referred below as `Building_without_signs.list`. Please refer to instructions on [Figure8.](https://www.figure-eight.com/dataset/open-images-annotated-with-bounding-boxes/). Please download the following file(s) and place them under `synthetic_signs/external/lists`:
- [train-annotations-bbox.csv](https://datasets.figure-eight.com/figure_eight_datasets/open-images/train-annotations-bbox.csv)
- [validation-annotations-bbox.csv](https://datasets.figure-eight.com/figure_eight_datasets/open-images/validation-annotations-bbox.csv)
- [test-annotations-bbox.csv](https://datasets.figure-eight.com/figure_eight_datasets/open-images/test-annotations-bbox.csv)

### :neutral_face: Neutral images list gen

```
python scripts/download-openimages-v5.py --help
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
Sample command
```
python scripts/download-openimages-v5.py --classes 'Building,Traffic_sign,Traffic_light' --mode train --dest <path_to_openimages_v5> --csvs synthetic_signs/external/lists
```
Build class list and filter image set & Filtering for outdoor images with no labeled signs.
```
find path_to_openimages_v5/Building -type f -name '*.jpg' > synthetic_signs/external/lists/Building.list
find path_to_openimages_v5/Traffic_sign -type f -name '*.jpg > synthetic_signs/external/lists/Traffic_sign.list
find path_to_openimages_v5/Traffic_light -type f -name '*.jpg > synthetic_signs/external/lists/Traffic_light.list
```
