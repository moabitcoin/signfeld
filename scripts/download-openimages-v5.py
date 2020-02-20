# Modified scripts original author below

# Author : Sunita Nayak, Big Vision LLC
# Usage: python3 downloadOI.py --classes 'Ice_cream,Cookie' --mode train

import sys
import csv
import os
import logging
import argparse
import subprocess
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool as thread_pool

CLI_LOGGING_FORMAT = '[%(filename)s][%(funcName)s:%(lineno)d]' + \
    '[%(levelname)s] %(message)s'
CLI_LOGGING_LEVEL = logging.INFO
CLI_LOGGING_STREAM = sys.stdout


def get_logger(logger_name):

  logger = logging.getLogger(logger_name)
  logger.setLevel(CLI_LOGGING_LEVEL)
  ch = logging.StreamHandler(CLI_LOGGING_STREAM)
  formatter = logging.Formatter(CLI_LOGGING_FORMAT)
  ch.setFormatter(formatter)
  ch.setLevel(CLI_LOGGING_LEVEL)
  logger.addHandler(ch)
  logger.propagate = False

  return logger


logger = get_logger(__file__)


cpu_count = multiprocessing.cpu_count()

parser = argparse.ArgumentParser(description="Download Class specific "
                                 "images from OpenImagesV5")
parser.add_argument("--mode", help="Dataset category - train"
                    ", validation or test", required=True)
parser.add_argument("--classes", help="Names of object classes(comma seperated)"
                    "to be downloaded", required=True)
parser.add_argument("--nthreads", help="Number of threads to use",
                    required=False, type=int, default=cpu_count * 2)
parser.add_argument("--dest", help="Destination directory", required=True)
parser.add_argument("--csvs", help="CSV file(s) directory", required=True)
parser.add_argument("--limit", help="Cap downloaded files to limit", type=int,
                    default=-1)
parser.add_argument("--occluded", help="Include occluded images",
                    required=False, action='store_true', default=False)
parser.add_argument("--truncated", help="Include truncated images",
                    required=False, action='store_true', default=False)
parser.add_argument("--groupOf", help="Include groupOf images",
                    required=False, action='store_true', default=False)
parser.add_argument("--depiction", help="Include depiction images",
                    required=False, action='store_true', default=False)
parser.add_argument("--inside", help="Include inside images",
                    required=False, action='store_true', default=False)

args = parser.parse_args()

run_mode = args.mode
destination = args.dest
csvs = args.csvs
limit = args.limit
threads = args.nthreads

classes = []
for class_name in args.classes.split(','):
    classes.append(class_name)

class_description_file = '{}/class-descriptions-boxable.csv'.format(csvs)
with open(class_description_file, mode='r') as infile:
    reader = csv.reader(infile)
    dict_list = {rows[1]: rows[0] for rows in reader}

logger.info('Finished reading {}'.format(class_description_file))

os.makedirs(destination, exist_ok=True)

pool = thread_pool(threads)
commands = []
cnt = 0

for ind in range(0, len(classes)):

    class_name = classes[ind]
    logging.info("Class " + str(ind) + " : " + class_name)

    class_dir = os.path.join(destination, class_name)
    os.makedirs(class_dir, exist_ok=True)
    logger.info('Making {}'.format(class_dir))

    os.makedirs(run_mode + '/' + class_name, exist_ok=True)

    command = "grep " + dict_list[class_name.replace('_', ' ')] + " {}/".format(csvs) \
        + run_mode + "-annotations-bbox.csv"
    class_annotations = subprocess.run(command.split(),
                                       stdout=subprocess.PIPE) \
                                  .stdout.decode('utf-8')
    class_annotations = class_annotations.splitlines()

    for line in tqdm(class_annotations):

        line_parts = line.split(',')

        # IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside
        if (not args.occluded and int(line_parts[8]) > 0):
            logging.debug("Skipped %s", line_parts[0])
            continue
        if (not args.truncated and int(line_parts[9]) > 0):
            logging.debug("Skipped %s", line_parts[0])
            continue
        if (not args.groupOf and int(line_parts[10]) > 0):
            logging.debug("Skipped %s", line_parts[0])
            continue
        if (not args.depiction and int(line_parts[11]) > 0):
            logging.debug("Skipped %s", line_parts[0])
            continue
        if (not args.inside and int(line_parts[12]) > 0):
            logging.debug("Skipped %s", line_parts[0])
            continue

        cnt = cnt + 1

        command = 'aws s3 --no-sign-request ' + \
            '--only-show-errors cp s3://open-images-dataset/' + \
            run_mode + '/' + line_parts[0] + '.jpg ' + destination + \
            '/' + class_name + '/' + line_parts[0] + '.jpg'

        commands.append(command)

logging.info("Annotation Count : " + str(cnt))
commands = list(set(commands))

if limit > 0:
  commands = commands[:limit]
logging.info("Number of images to be downloaded : " + str(len(commands)))

list(tqdm(pool.imap(os.system, commands), total=len(commands)))

pool.close()
pool.join()
