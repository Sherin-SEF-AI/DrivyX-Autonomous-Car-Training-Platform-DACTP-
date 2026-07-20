#!/usr/bin/python
#

# python imports
from __future__ import print_function
import os
import glob
import sys
# DRIVYX patch 3: `from imageio import imread, imsave` and `from numpngw import write_png`
# were imported at module scope but are never used in this file. Both packages are absent
# here, so the unused imports made the script unimportable. Removed rather than installing
# two dependencies for dead code.
import numpy as np

# DRIVYX patch 4: upstream appends helpers/ to sys.path inside main(), but the two imports
# below run at module scope and transitively import anue_labels from helpers/. The append
# was therefore too late and the script only worked if the caller pre-seeded PYTHONPATH.
# Hoisting the path setup above the imports makes the script self-contained.
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'helpers')))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from json2labelImg import json2labelImg
from json2instanceImg import json2instanceImg


from tqdm import tqdm

from argparse import ArgumentParser
import os

import pandas as pd
import shutil

args = None


def process_folder(fn):
    global args

    dst = fn.replace("_polygons.json", "_label{}s.png".format(args.id_type))

    # do the conversion
    try:
        json2labelImg(fn, dst, args.id_type)
    except:
        tqdm.write("Failed to convert: {}".format(fn))
        raise

    if args.instance:
        dst = fn.replace("_polygons.json",
                         "_instance{}s.png".format(args.id_type))

        # do the conversion
        # try:
        json2instanceImg(fn, dst, args.id_type)
        # except:
        #     tqdm.write("Failed to convert: {}".format(f))
        #     raise

    if args.color:
        # create the output filename
        dst = fn.replace("_polygons.json", "_labelColors.png")

        # do the conversion
        try:
            json2labelImg(fn, dst, 'color')
        except:
            # DRIVYX patch 6: upstream formatted `f`, undefined in this scope (the parameter
            # is `fn`), so a colour-conversion failure raised NameError and hid the real
            # exception.
            tqdm.write("Failed to convert: {}".format(fn))
            raise

    # if args.panoptic and args.instance:
        # panoptic_converter(f, out_folder, out_file)


def get_args():
    parser = ArgumentParser()

    parser.add_argument('--datadir', default="")
    parser.add_argument('--id-type', default='level3Id')
    parser.add_argument('--color', type=bool, default=False)
    parser.add_argument('--instance', type=bool, default=False)
    parser.add_argument('--panoptic', type=bool, default=False)
    parser.add_argument('--semisup_da', type=bool, default=False)
    parser.add_argument('--unsup_da', type=bool, default=False)
    parser.add_argument('--weaksup_da', type=bool, default=False)
    parser.add_argument('--num-workers', type=int, default=10)

    args = parser.parse_args()

    return args

# The main method


def main(args):
    import sys
    if args.panoptic:
        args.instance = True
    sys.path.append(os.path.normpath(os.path.join(
        os.path.dirname(__file__), '..', 'helpers')))
    # how to search for all ground truth
    searchFine = os.path.join(args.datadir, "gtFine",
                              "*", "*", "*_gt*_polygons.json")

    # search files
    filesFine = glob.glob(searchFine)
    filesFine.sort()

    files = []#filesFine

    #for semi supervised domain adaptation, convert only selected images
    filesnew_semisup = []
    filesnewunsup = []
    if args.semisup_da:
        d_strat = list(pd.read_csv('./domain_adaptation/target/semi-supervised/selected_samples.csv',header=None)[0])
        d_strat = ["/".join(filenew.replace("_labellevel3Ids.png", "").split("/")[-3:]) for filenew in d_strat]
        print(d_strat)
        for fileold in filesFine:
            if "val/" not in fileold:
                searchstr = "/".join(fileold.replace("_polygons.json", "").split("/")[-3:])
                if searchstr in d_strat:
                    print(searchstr)
                    filesnew_semisup.append(fileold)
            else: filesnew_semisup.append(fileold)
        files = filesnew_semisup
    elif args.unsup_da or args.weaksup_da:    #for unsupervised domain adaptation, convert only val images
        for fileold in filesFine:
            if "val/" in fileold:
                filesnewunsup.append(fileold)
        files = filesnewunsup
    else: files = filesFine

    #print('args.semisup_da', args.semisup_da, len(files))
    if not files:
        # DRIVYX patch 5: upstream called tqdm.writeError(), which does not exist (tqdm has
        # write()), so an empty input set raised AttributeError and then hit a division by
        # len(files) below. Report the real problem and stop.
        raise SystemExit(
            "Did not find any *_polygons.json under {}. Nothing to convert.".format(
                os.path.join(args.datadir, "gtFine")))

    # a bit verbose
    tqdm.write(
        "Processing {} annotation files for Sematic/Instance Segmentation".format(len(files)))

    # iterate through files
    progress = 0
    tqdm.write("Progress: {:>3} %".format(
        progress * 100 / len(files)), end=' ')

    from multiprocessing import Pool
    import time

    pool = Pool(args.num_workers)
    # results = pool.map(process_pred_gt_pair, pairs)
    results = list(
        tqdm(pool.imap(process_folder, files), total=len(files)))
    pool.close()
    pool.join()

    if args.panoptic:
        from cityscape_panoptic_gt import panoptic_converter
        for split in ['train', 'val']:

            tqdm.write("Panoptic Segmentation {} split".format(split))
            folder_name = os.path.join(args.datadir, 'gtFine')
            output_folder = os.path.join(folder_name, split + "_panoptic")
            os.makedirs(output_folder, exist_ok=True)
            out_file = os.path.join(folder_name, split + "_panoptic.json")
            panoptic_converter(args.num_workers, os.path.join(
                folder_name, split), output_folder, out_file)


if __name__ == "__main__":
    args = get_args()
    main(args)
