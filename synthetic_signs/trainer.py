import json
import logging
import os
import pathlib
import sys
import time
from tqdm import tqdm

import PIL
import torch
import yaml

from detectron2.config import get_cfg
from detectron2.data import (DatasetCatalog, DatasetMapper, MetadataCatalog,
                             build_detection_train_loader, transforms)
from detectron2.engine import (DefaultTrainer, default_argument_parser,
                               default_setup, launch)
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.structures import BoxMode

# TODO(niwojke): Make this configurable via command line arguments.
CLIP_GRAD_NORM = 10.0

TRAIN_SPLIT_NAME = "synthetic_signs_train"
VALID_SPLIT_NAME = "synthetic_signs_valid"


def get_synthetic_sign_dicts(csv_filepath, name_to_id, image_shape=None):
  """Get dataset dictionaries.

  Args:
    csv_filepath (pathlib.Path): Path to dataset CSV file. Expected format of
      this file is ``filename,x0,y0,x1,y1,category`` where (y0, x0) is the
      top-left corner and (y1, x1) is the bottom-right corner of the ground
      truth bounding box.
    name_to_id (Dict[str, int]): A dictionary that maps from category name
      as it is found in the CSV file to a unique category identifier in [0, N)
      with N being the number of categories.
    image_shape (Tuple[int, int], optional): Optional fixed image height and
      width or all images in the dataset. If not given, image shape is
      determined for each image individually by reading the file header.

  Returns:
    output (List[Dict[str, T]]): Dataset dictionary in detectron2 format.
  """
  filename_to_objects = dict()
  with open(csv_filepath, "r") as file_handle:
    for line in file_handle:
      filename, x0, y0, x1, y1, category = line.rstrip().split(",")
      category_id = name_to_id[category]

      filename_to_objects.setdefault(filename, []).append(
          (int(x0), int(y0), int(x1), int(y1), category_id))

  filenames = sorted(filename_to_objects.keys())
  dataset_dicts = []

  progress_bar = tqdm(filenames)
  progress_bar.set_description("Loading {}".format(csv_filepath))

  for i, filename in enumerate(progress_bar):
    annotations = []
    for x0, y0, x1, y1, category_id in filename_to_objects[filename]:
      annotations.append({
          "bbox": [x0, y0, x1, y1],
          "bbox_mode": BoxMode.XYXY_ABS,
          "category_id": category_id,
          "iscrowd": 0,
      })

    if image_shape is not None:
      height, width = image_shape
    else:
      width, height = PIL.Image.open(filename).size

    dataset_dicts.append({
        "file_name": filename,
        "image_id": i,
        "height": height,
        "width": width,
        "annotations": annotations,
    })

  return dataset_dicts


def write_coco_json(filepath, dataset_dicts, name_to_id, **kwargs):
  """Write COCO annotation JSON file for a given dataset.

  NOTE(niwojke): This function may be replaced by official detectron code in the
  future. See ``https://github.com/facebookresearch/detectron2/pull/175``.

  Args:
    filename (pathlib.Path): Output path.
    dataset_dicts (List[Dict[str, T]]): Dataset dictionaries in native
      detectron2 format.
    name_to_id (Dict[str, int]): A dictionary that maps from category name
      as it is found in the CSV file to a unique category identifier in [0, N)
      with N being the number of categories.
    kwargs (Dict[str, str]): Additional keyword arguments used to set the
      dataset information record.
  """
  info = {
      "description": kwargs.get("description", ""),
      "url": kwargs.get("url", ""),
      "version": kwargs.get("version", "0.0"),
      "year": kwargs.get("year", "2017"),
      "contributor": kwargs.get("contributor", ""),
      "date_created": kwargs.get("date_created", "2017/01/01"),
  }

  licenses = {
      "url": "closed",
      "id": 0,
      "name": "closed",
  }

  images, annotations = [], []
  annotation_id = 1
  for record in dataset_dicts:
    images.append({
        "id": record["image_id"],
        "width": record["width"],
        "height": record["height"],
        "file_name": record["file_name"]
    })

    for annotation in record["annotations"]:
      x0, y0, x1, y1 = annotation["bbox"]
      annotations.append({
          "id": annotation_id,
          "category_id": annotation["category_id"],
          "bbox": [x0, y0, x1 - x0, y1 - y0],
          "iscrowd": annotation["iscrowd"],
          "image_id": record["image_id"],
          "area": (x1 - x0) * (y1 - y0),
      })
      annotation_id += 1

  categories = [{
      "id": category_id,
      "name": "{}".format(category_name),
      "supercategory": ""
  } for category_name, category_id in name_to_id.items()]

  coco_dict = {
      "info": info,
      "licenses": licenses,
      "images": images,
      "annotations": annotations,
      "categories": categories,
  }

  with filepath.open(mode="w") as file_handle:
    json.dump(coco_dict, file_handle)


def register_synthetic_signs(csv_filepath,
                             label_filepath,
                             cfg,
                             name,
                             image_shape=None):
  """Register a synthetic signs dataset with the detectron2 catalog.

  Args:
    csv_filepath (pathlib.Path): Path to dataset CSV file. Expected format of
      this file is ``filename,x0,y0,x1,y1,category`` where (y0, x0) is the
      top-left corner and (y1, x1) is the bottom-right corner of the ground
      truth bounding box.
    label_filepath (pathlib.Path): Path to label YAML file. This file should
      contain a dictionary that maps from a unique category identifier in
      [0, N), where N is the number of categories, to the category name.
    cfg (detectron2.config.CfgNode): Configuration file, used to query the
      output directory.
    name (str): The name under which this dataset is registered in the
      detectron2 catalog.
    image_shape (Tuple[int, int], optional): Optional fixed image height and
      width or all images in the dataset. If not given, image shape is
      determined for each image individually by reading the file header.
  """
  with label_filepath.open() as file_handle:
    id_to_name = yaml.safe_load(file_handle)
    name_to_id = {v: k for (k, v) in id_to_name.items()}

  dataset_dicts = get_synthetic_sign_dicts(csv_filepath, name_to_id,
                                           image_shape)
  DatasetCatalog.register(name, lambda: dataset_dicts)

  coco_json_path = pathlib.Path(cfg.OUTPUT_DIR).joinpath("{}.json".format(name))
  write_coco_json(coco_json_path, dataset_dicts, name_to_id)

  thing_classes = [None for _ in name_to_id]
  for class_name, category_id in name_to_id.items():
    thing_classes[category_id] = class_name
  assert all([x is not None for x in thing_classes])

  MetadataCatalog.get(name).set(thing_classes=thing_classes,
                                json_file=str(coco_json_path))


class NoFlipDatasetMapper(DatasetMapper):
  """
  This class implements a simple wrapper around the default DatasetMapper,
  which overwrites the input data transformations. In particular it removes
  the random flipping applied by the default mapper.
  """

  BLACKLIST = [
      transforms.RandomFlip,
  ]

  @staticmethod
  def is_blacklisted(x):
    return any([isinstance(x, y) for y in NoFlipDatasetMapper.BLACKLIST])

  def __init__(self, cfg, is_train=True):
    super().__init__(cfg, is_train)
    self.tfm_gens = [
        x for x in self.tfm_gens if not NoFlipDatasetMapper.is_blacklisted(x)
    ]
    logging.getLogger(
        transforms.__name__).info("Overwriting TransformGens to: " +
                                  str(self.tfm_gens))


class Trainer(DefaultTrainer):
  """
  This class implements a simple wrapper around the default trainer.
  Modifications are::

    * Set COCO evaluator.
    * Set custom DatasetMapper.
    * Apply gradient clipping.
  """

  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):
    if output_folder is None:
      output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    return COCOEvaluator(dataset_name, cfg, True, output_folder)

  @classmethod
  def build_train_loader(self, cfg):
    return build_detection_train_loader(cfg,
                                        mapper=NoFlipDatasetMapper(
                                            cfg, is_train=True))

  def run_step(self):
    assert self.model.training, "[Trainer] model was changed to eval mode!"
    start = time.perf_counter()

    data = next(self._data_loader_iter)
    data_time = time.perf_counter() - start

    loss_dict = self.model(data)
    losses = sum(loss for loss in loss_dict.values())
    self._detect_anomaly(losses, loss_dict)

    metrics_dict = loss_dict
    metrics_dict["data_time"] = data_time
    self._write_metrics(metrics_dict)

    self.optimizer.zero_grad()
    losses.backward()

    torch.nn.utils.clip_grad_norm_(self.model.parameters(), CLIP_GRAD_NORM)
    self.optimizer.step()


def main(args):
  """Main program entry point."""
  # Generate detectron2 config from command line arguments.
  cfg = get_cfg()
  cfg.merge_from_file(args.config_file)
  cfg.merge_from_list(args.opts)

  # The configuration file should not contain any datasets. They are configured
  # from command line arguments instead.
  if len(cfg.DATASETS.TRAIN) > 0 or len(cfg.DATASETS.TEST) > 0:
    logging.error("Please set DATASETS.TRAIN = () and DATASETS.TEST = ().")
    sys.exit(1)
  cfg.DATASETS.TRAIN = (TRAIN_SPLIT_NAME, )
  cfg.DATASETS.TEST = (VALID_SPLIT_NAME, )

  cfg.freeze()
  default_setup(cfg, args)

  # Register synthetic sign datasets.
  if args.image_width is not None or args.image_height is not None:
    if args.image_width is None or args.image_height is None:
      logging.error(
          "Please specify both, image-width and image-height (or none).")
      sys.exit(1)
    image_shape = args.image_height, args.image_width
  else:
    image_shape = None

  register_synthetic_signs(args.train_csv,
                           args.label_map,
                           cfg,
                           name=TRAIN_SPLIT_NAME,
                           image_shape=image_shape)
  if args.valid_csv is not None:
    register_synthetic_signs(args.valid_csv,
                             args.label_map,
                             cfg,
                             name=VALID_SPLIT_NAME,
                             image_shape=image_shape)

  # Run training or evaluation.
  if args.eval_only:
    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume)
    res = Trainer.test(cfg, model)
    if comm.is_main_process():
      verify_results(cfg, res)
    return res

  trainer = Trainer(cfg)
  trainer.resume_or_load(resume=args.resume)
  return trainer.train()


def parse_args():
  """Parse command line arguments."""
  parser = default_argument_parser()
  parser.add_argument("--label-map",
                      dest="label_map",
                      type=pathlib.Path,
                      help="Label map in YAML format which maps from category "
                      "ID to name.")
  parser.add_argument("--train-csv",
                      dest="train_csv",
                      required=True,
                      type=pathlib.Path,
                      help="Path to training data CSV file.")
  parser.add_argument("--valid-csv",
                      dest="valid_csv",
                      required=False,
                      type=pathlib.Path,
                      help="Optional path to validation data CSV file.")
  parser.add_argument(
      "--image-width",
      type=int,
      help="Image width (optional, used to speed up dataset processing).")
  parser.add_argument(
      "--image-height",
      type=int,
      help="Image height (optional, used to speed up dataset processing).")
  return parser.parse_args()


if __name__ == "__main__":
  args = parse_args()
  launch(
      main,
      args.num_gpus,
      num_machines=args.num_machines,
      machine_rank=args.machine_rank,
      dist_url=args.dist_url,
      args=(args, ),
  )
