## :no_pedestrians: Traffic sign templates

A subset of the near complete [list](https://de.wikipedia.org/wiki/Bildtafel_der_Verkehrszeichen_in_der_Bundesrepublik_Deutschland_seit_2017) of German traffic sign(s) is of interest to us. More specifically [these](https://github.com/moabitcoin/Signfeld/tree/master/synthetic_signs/templates). These signs form a subset (signs of interest) which formalise [`turn-restrictions`](https://wiki.openstreetmap.org/wiki/Relation:restriction) in OSM. Using the templates for the signs of interest we build a synthetic training set following the idea presented in [IJCN2019](https://github.com/LCAD-UFES/publications-tabelini-ijcnn-2019) and [CVPR 2016](https://github.com/ankush-me/SynthText).

## :factory: Generate synthetic data

Use [generate-synthetic-dataset](bin/generate-synthetic-dataset) to generate a synthetic sign dataset:
```
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
generate-synthetic-dataset --backgrounds=synthetic_signs/external/lists/Building_without_signs.list \
                           --templates-path=synthetic_signs/templates \
                           --out-path=experiments/synthetic-signfeld-dataset \
                           --n=16 \
                           --max-images=200000 \
                           --augmentations=resources/configs/augmentations.yaml
```
