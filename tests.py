import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image


def test_depth_values(data_dir):
    count = 0
    for image_name in sorted(glob.glob(os.path.join(data_dir, "*.png"))):
        depth_im = np.array(Image.open(image_name)).astype(np.float) / 256.
        if (np.max(depth_im) < 1) | (np.max(depth_im > 6)) | (np.min(depth_im) < 0):
            count += 1
            print(image_name)
    return count


def main():

    sets = ['train', 'val', 'test']
    for dataset in sets:
        print('Checking ' + dataset + ' dataset')
        gt_dir = os.path.join('../data/Nachsholim/depth_lft/uint16_slam', dataset, 'truncate')
        count = test_depth_values(gt_dir)
        if count > 0:
            print('There are ' + str(count) + ' invalid samples in ' + dataset + ' set.')
        rgb_dir = os.path.join('../data/Nachsholim/rgb_unenhanced_slam', dataset)
        count = test_depth_values(rgb_dir)
        if count > 0:
            print('There are ' + str(count) + ' invalid samples in ' + dataset + ' set.')
        sparse_dir = os.path.join('../data/Nachsholim/depth_lft/sparse_manual_slam', dataset)
        count = test_depth_values(sparse_dir)
        if count > 0:
            print('There are ' + str(count) + ' invalid samples in ' + dataset + ' set.')
