#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Wrapper to train and test a video classification model."""

import argparse
import sys
import torch
import cv2
import numpy as np
import time

import slowfast.utils.checkpoint as cu
import slowfast.utils.multiprocessing as mpu
from slowfast.config.defaults import get_cfg
from slowfast.models import model_builder
from slowfast.datasets import transform as transform

from test_net import test
from train_net import train


def parse_args():
    """
    Parse the following arguments for the video training and testing pipeline.
    Args:
        shard_id (int): shard id for the current machine. Starts from 0 to
            num_shards - 1. If single machine is used, then set shard id to 0.
        num_shards (int): number of shards using by the job.
        init_method (str): initialization method to launch the job with multiple
            devices. Options includes TCP or shared file-system for
            initialization. details can be find in
            https://pytorch.org/docs/stable/distributed.html#tcp-initialization
        cfg (str): path to the config file.
        opts (argument): provide addtional options from the command line, it
            overwrites the config loaded from file.
        """
    parser = argparse.ArgumentParser(
        description="Provide SlowFast video training and testing pipeline."
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default="configs/Kinetics/SLOWFAST_4x16_R50.yaml",
        type=str,
    )
    parser.add_argument(
        "--path",
        dest="path",
        help="Path to the config file",
        type=str,
    )
    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


def load(path, size, _use_bgr, _data_mean, _data_std, alpha):
    cap = cv2.VideoCapture(path)
    results = []
    print('video load start')
    if cap.isOpened():
        print(cap.get(cv2.CAP_PROP_FPS))
        fps = cap.get(cv2.CAP_PROP_FPS)
    while(cap.isOpened()):
        
        ret = cap.grab()
        frame_count = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if frame_count % int(fps/10) == 0:
            ret, frame = cap.retrieve()
            if not ret or len(results) > 500:
                break
            frame = cv2.resize(frame, (size, size)) / 255
            results.append(frame)
    raw = results
    print('preprocessing start')
    results = transform.color_normalization(
            torch.tensor(results).permute(3, 0, 1, 2).unsqueeze(0),
            np.array(_data_mean, dtype=np.float32),
            np.array(_data_std, dtype=np.float32),
        )
    print('load complete')
    for seek in range(results.shape[2] - 23):
        part = results[0:, :, seek: seek+24]
        last_frame = raw[seek+23]
        print(last_frame.shape)

        if not _use_bgr:
            # Convert image format from BGR to RGB.
            # Note that Kinetics pre-training uses RGB!
            part = part[:, [2, 1, 0], ...]
        slow_pathway = torch.index_select(
            part,
            2,
            torch.linspace(
                0, part.shape[2] - 1, part.shape[2] // alpha
            ).long(),
        )
        # print(slow_pathway)
        yield [slow_pathway.float(), part.float()], last_frame
        # del results[0]
    yield None


def load_config(args):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    cfg = get_cfg()
    # Load config from cfg.
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)

    # Inherit parameters from args.
    if hasattr(args, "num_shards") and hasattr(args, "shard_id"):
        cfg.NUM_SHARDS = args.num_shards
        cfg.SHARD_ID = args.shard_id
    if hasattr(args, "rng_seed"):
        cfg.RNG_SEED = args.rng_seed
    if hasattr(args, "output_dir"):
        cfg.OUTPUT_DIR = args.output_dir

    # Create the checkpoint dir.
    cu.make_checkpoint_dir(cfg.OUTPUT_DIR)
    return cfg


def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)
    last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
    cfg.NUM_GPUS = 1 # fix single gpu to demo
    model = model_builder.build_model(cfg)
    cu.load_checkpoint(last_checkpoint, model, False)

    path = args.path
    cat = {
            1: 'walking',
            2: 'standing',
            3: 'rising',
            4: 'lying',
            5: 'falling'
        }
    model.eval().cuda()
    with torch.no_grad():
        for inputs in load(path, cfg.DATA.TEST_CROP_SIZE, False, cfg.DATA.MEAN, cfg.DATA.STD, cfg.SLOWFAST.ALPHA):
            if isinstance(inputs, (list, tuple)) and isinstance(inputs[0], list) and torch.is_tensor(inputs[0][0]):
                data, frame = inputs
                startTime = time.time()
                for i in range(len(data)):
                    data[i] = data[i].cuda(non_blocking=True)
                
                results = model(data)
                endTime = time.time() - startTime
                print(endTime) 
                scores = results[0].get_field('scores')
                index = scores > 0.3
                results = results[0][index]
                bbox = results.bbox.int()
                scores = results.get_field("scores").tolist()
                labels = results.get_field("labels").tolist()
                labels = [cat[i] for i in labels]
                # bbox = results[0].bbox.int()
                # print(data[0].shape)
                # print(data[1].shape)
                # frame = data[1][0, :, -1].permute(1, 2, 0).cpu().numpy()
                # print(frame.shape)
                template = "{}: {:.2f}"
                    
                if bbox.shape[0] > 0:
                    for box, score, label in zip(bbox, scores, labels):
                        x, y = box[:2]
                        s = template.format(label, score)
                        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
                        frame = cv2.rectangle(frame, tuple(top_left), tuple(bottom_right), (255, 0, 0), 2)
                        frame = cv2.putText(
                            frame, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
                        )
                    cv2.imshow('show', frame)
                else:
                    cv2.imshow('show', frame)
                
                cv2.waitKey(5)
                # print(results[0].bbox)
                # print(results[0].get_field('scores'))
            else:
                break

if __name__ == "__main__":
    main()
