import os
import torch
import torch.nn.functional as functional
import numpy as np
import cv2
import time
from config import get_config
import argparse

from scipy import ndimage

from src.utils import log, is_only_one_bool_is_true, gray_to_bgr, make_path, calculate_game, ndarray_to_tensor
from src.crowd_count import CrowdCount
from src.data import Data
from src.data_multithread_preload import multithread_dataloader
from src.data_multithread_preload import PreloadData
from src import network

import torch.multiprocessing as mp

def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg',default='./configs/swin_tiny_patch4_window7_224.yaml' ,type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, default=16, help="batch size for single GPU")
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config

_, config = parse_option()



test_flag = dict()
test_flag['preload'] = False
test_flag['label'] = False
test_flag['mask'] = False


test_model_path = './final_models/shtechA.h5'
original_dataset_name = 'shtechA'
test_data_config = dict()
test_data_config['shtA1_test'] =test_flag.copy()

# load data
all_data = multithread_dataloader(test_data_config)
net = CrowdCount(config)
network.load_net_safe(test_model_path, net)

net.cuda()
net.eval()

# log_info = []

total_forward_time = 0.0
txt_log = list()

# calculate error on the test dataset
for data_name in test_data_config:
    data = all_data[data_name]['data']
    mae = 0.0
    mse = 0.0
    game_0 = 0.0
    game_1 = 0.0
    game_2 = 0.0
    game_3 = 0.0
    index = 0


    for blob in data:
        image_data = blob['image']
        ground_truth_data = blob['density']
        image_name = blob['image_name'][0]

        start_time = time.perf_counter()
        estimate_map = net(image_data)
        total_forward_time += time.perf_counter() - start_time

        ground_truth_map = ground_truth_data.data.cpu().numpy()
        ground_truth_count = np.sum(ground_truth_map)


        estimate_map = estimate_map[0].data.cpu().numpy()
        estimate_map_count = np.sum(estimate_map)

        mae += np.abs(ground_truth_count - estimate_map_count)
        mse += (ground_truth_count - estimate_map_count) ** 2
        game_0 += calculate_game(ground_truth_map, estimate_map, 0)
        game_1 += calculate_game(ground_truth_map, estimate_map, 1)
        game_2 += calculate_game(ground_truth_map, estimate_map, 2)
        game_3 += calculate_game(ground_truth_map, estimate_map, 3)
        index += 1

    mae = mae / index
    mse = np.sqrt(mse / index)
    game_0 = game_0 / index
    game_1 = game_1 / index
    game_2 = game_2 / index
    game_3 = game_3 / index
    print('mae: %.2f mse: %.2f game: %.2f %.2f %.2f %.2f' % (mae, mse, game_0, game_1, game_2, game_3))
print('total forward time is %f seconds of %d samples. ' % (total_forward_time, index))





