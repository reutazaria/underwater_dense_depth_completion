import os
import cv2
import glob
import imageio
import numpy as np
from PIL import Image
from dataloaders import transforms
from matplotlib import pyplot as plt

o_height, o_width = 416, 736  # 512, 800
transform_geometric = transforms.Compose([
    transforms.CenterCrop((o_height, o_width))])

c_map = plt.cm.jet

path = '/home/reuta/self-supervised-depth-completion/gifs/'
gif_name = "movie_depth.gif"

rgb_dir = '../data/Nachsholim/rearranged/rgb/test'
rgb_images = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
sparse_dir = '../data/Nachsholim/rearranged/sparse/test'
sparse_images = sorted(glob.glob(os.path.join(sparse_dir, "*.png")))
output_dir = '../results/mode=dense.data=nachsholim.input=gd.resnet18.epochs20.criterion=Tukey.lr=0.0001.bs=2.wd=0.' \
             'pretrained=True.jitter=0.1.rank_metric=diff_thresh.time=2021-08-04@10-21_with_var_test/val_output'
output_images = sorted(glob.glob(os.path.join(output_dir, "*.png")))

N = 100
with imageio.get_writer(path+gif_name, mode='I') as writer:
    for inx in range(N):
        rgb_im = transform_geometric(np.array(Image.open(rgb_images[inx])))
        sparse_im = transform_geometric(np.array(Image.open(sparse_images[inx]))).astype(np.float) / 256.
        sparse_im[sparse_im > 6] = 0
        sparse_dilate = cv2.dilate(sparse_im, np.ones((5, 5), np.uint8), iterations=1)
        out_im = transform_geometric(np.array(Image.open(output_images[inx]))).astype(np.float) / 256.
        depth_tot = np.concatenate((sparse_dilate, out_im), axis=1)
        depth_norm = (depth_tot - np.min(depth_tot)) / (np.max(depth_tot) - np.min(depth_tot))
        depth_colored = (255 * c_map(depth_norm)[:, :, :3]).astype('uint8')
        row = np.concatenate((rgb_im, depth_colored), axis=1)
        writer.append_data(row)


# def func(x, tar, a, c):
#     return np.abs(x-tar) / (a * np.exp(tar)) + c
