import os
import cv2
import glob
import imageio
import numpy as np
from PIL import Image
from dataloaders import transforms
from matplotlib import pyplot as plt

o_height, o_width = 416, 736  # 512, 800
transform_geometric = transforms.Compose([transforms.CenterCrop((o_height, o_width))])

c_map = plt.cm.jet
v_map = plt.cm.gray

path = '/home/reuta/self-supervised-depth-completion/gifs/'
gif_name = "movie_depth_Tukey_05var_masked.gif"

rgb_dir = '../data/Nachsholim/rearranged/rgb/test'
rgb_images = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
sparse_dir = '../data/Nachsholim/rearranged/sparse/test'
sparse_images = sorted(glob.glob(os.path.join(sparse_dir, "*.png")))
gt_dir = '../data/Nachsholim/rearranged/gt/test'
gt_images = sorted(glob.glob(os.path.join(gt_dir, "*.png")))
output_dir = '../pretrained_models/supervised/nachsholim_manual_slam/rearranged/test/' \
             'mode=dense.data=nachsholim.input=gd.resnet18.epochs20.criterion=Tukey.lr=0.0001.bs=2.wd=0.' \
             'pretrained=True.jitter=0.1.rank_metric=diff_thresh.time=2021-08-04@10-21_with_var_test'
output_images = sorted(glob.glob(os.path.join(output_dir, 'val_output', "*.png")))
var_images = sorted(glob.glob(os.path.join(output_dir, 'val_output_var', "*.tif")))

N = 150
skip = 150
with imageio.get_writer(path+gif_name, mode='I') as writer:
    for ind in range(N):
        rgb_im = transform_geometric(np.array(Image.open(rgb_images[ind+skip])))
        gt_im = transform_geometric(np.array(Image.open(gt_images[ind+skip]))).astype(np.float) / 256.
        gt_im[gt_im > 6] = 0
        sparse_im = transform_geometric(np.array(Image.open(sparse_images[ind + skip]))).astype(np.float) / 256.
        sparse_im[sparse_im > 6] = 0
        sparse_dilate = cv2.dilate(sparse_im, np.ones((5, 5), np.uint8), iterations=1)
        out_im = transform_geometric(np.array(Image.open(output_images[ind+skip]))).astype(np.float) / 256.
        depth_tot = np.concatenate((sparse_dilate, out_im), axis=1)
        depth_norm = (depth_tot - np.min(depth_tot)) / (np.max(depth_tot) - np.min(depth_tot))
        depth_colored = (255 * c_map(depth_norm)[:, :, :3]).astype('uint8')
        depth_colored_var = (255 * c_map(depth_norm)[:, :, :3]).astype('uint8')
        if var_images:
            var_im = transform_geometric(np.array(Image.open(var_images[ind + skip])))
            var_norm = (var_im - np.min(var_im)) / (np.max(var_im) - np.min(var_im))
            var_colored = (255 * v_map(var_norm)[:, :, :3]).astype('uint8')
            # row = np.concatenate((rgb_im, depth_colored, var_colored), axis=1)

            mask_colored = depth_colored_var[:, o_width:o_width * 2, :]
            (mask_colored[:, :, 0])[np.sqrt(np.exp(var_im)) > 0.5] = 0
            # (mask_colored[:, :, 0])[gt_im == 0] = (depth_colored[:, o_width:o_width * 2, 0])[gt_im == 0]
            (mask_colored[:, :, 1])[np.sqrt(np.exp(var_im)) > 0.5] = 0
            # (mask_colored[:, :, 1])[gt_im == 0] = (depth_colored[:, o_width:o_width * 2, 1])[gt_im == 0]
            (mask_colored[:, :, 2])[np.sqrt(np.exp(var_im)) > 0.5] = 0
            # (mask_colored[:, :, 2])[gt_im == 0] = (depth_colored[:, o_width:o_width * 2, 2])[gt_im == 0]

            depth_masked = cv2.addWeighted(depth_colored[:, o_width:o_width*2, :], 0.6, mask_colored, 0.4, 0)

            row = np.concatenate((rgb_im, depth_colored[:, 0:o_width, :], depth_masked, var_colored), axis=1)
        else:
            row = np.concatenate((rgb_im, depth_colored), axis=1)
        writer.append_data(row)


# def func(x, tar, a, c):
#     return np.abs(x-tar) / (a * np.exp(tar)) + c
