import math

import cv2
import os
import rawpy
import imageio
import glob

import torch
from PIL import Image
from scipy.interpolate import griddata
# import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import random
from dataloaders import transforms


def crop_image(image):
    h, w, c = image.shape
    orig_im = cv2.imread(
        '../data/depth_selection/val_selection_cropped/image/2011_09_26_drive_0002_sync_image_0000000005_image_02.png')
    h_orig, w_orig, c = orig_im.shape
    x = round((h - h_orig) / 2)
    y = round((w - w_orig) / 2)
    crop_img = image[x:x + h_orig, y:y + w_orig]
    # crop_img = image[x - 100:x + h_orig - 100, y:y + w_orig]
    return crop_img


def extract_video_images(video_name):
    video_path = os.path.join('..', video_name)
    video_cap = cv2.VideoCapture(video_path)
    success, image = video_cap.read()
    count = 0
    output_dir_orig = os.path.join('..', video_name.split('.')[0] + '_images')
    if not os.path.isdir(output_dir_orig):
        os.mkdir(output_dir_orig)
    output_dir = os.path.join('..', video_name.split('.')[0] + '_cropped_images')
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    while success:
        scale = 1216 / image.shape[1]
        width = int(image.shape[1] * scale)
        height = int(image.shape[0] * scale)
        dim = (width, height)
        resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        image_name_orig = os.path.join(output_dir_orig, video_name.split('.')[0] + "_image_%d.png" % count)
        cv2.imwrite(image_name_orig, resized_image)
        crop_img = crop_image(resized_image)
        image_name = os.path.join(output_dir, video_name.split('.')[0] + "_image_%d.png" % count)
        cv2.imwrite(image_name, crop_img)
        # save frame as PNG file
        # if count == 1999:
        #     return
        success, image = video_cap.read()
        print('Read a new frame: ', image_name)
        count += 1


def raw_to_png_orig(images_dir):
    output_dir = os.path.join(images_dir, "png_orig")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for image in glob.glob(os.path.join(images_dir, "*NEF")):
        with rawpy.imread(image) as raw:
            rgb = raw.postprocess()
            image_name = os.path.join(output_dir, image.split('/')[-1].split('.')[0] + ".png")
            imageio.imsave(image_name, rgb)


def raw_to_png(images_dir):
    output_dir = os.path.join(images_dir, "png_resized_new")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for image in glob.glob(os.path.join(images_dir, "*NEF")):
        with rawpy.imread(image) as raw:
            rgb = raw.postprocess()
            orig_im = cv2.imread('../data/CaesareaSet/enhanced/input/seaErra_in_01914.png')
            h_orig, w_orig, c = orig_im.shape
            scale = w_orig / rgb.shape[1]
            # scale = 0.1648
            width = int(rgb.shape[1] * scale)
            height = int(rgb.shape[0] * scale)
            dim = (width, height)
            resized_rgb = cv2.resize(rgb, dim, interpolation=cv2.INTER_AREA)
            # cropped_image = crop_image(resized_rgb)
            image_name_cropped = os.path.join(output_dir, image.split('/')[-1].split('.')[0] + ".png")
            print("saving image: ", image_name_cropped)
            imageio.imsave(image_name_cropped, resized_rgb)


def tif_to_png(maps_dir):
    output_dir = os.path.join(maps_dir, "png")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    # output_dir_resized = os.path.join(maps_dir, "png_resized_new")
    # if not os.path.isdir(output_dir_resized):
    #     os.mkdir(output_dir_resized)
    # output_dir_cropped = os.path.join(maps_dir, "cropped_png")
    # if not os.path.isdir(output_dir_cropped):
    #     os.mkdir(output_dir_cropped)
    for image in glob.glob(os.path.join(maps_dir, "*.tif")):
        # for converting Tif to Png from terminal, run:
        # gdal_translate -of PNG <<current_image>> <<new_image_name>>
        im = Image.open(image)
        image_name = os.path.join(output_dir, image.split('/')[-1].split('.')[0] + ".png")
        imageio.imsave(image_name, im)
        # png_im = np.array(Image.open(image_name)).astype("uint16")
        # # png_im = np.array(Image.open(image))
        # # png_im = Image.open(image)
        # orig_im = cv2.imread(
        #     '../data/depth_selection/val_selection_cropped/image/2011_09_26_drive_0002_sync_image_0000000005_image_02.png')
        # # orig_im = cv2.imread('../data/CaesareaSet/enhanced/input/seaErra_in_01914.png')
        # h_orig, w_orig, c = orig_im.shape
        # scale = w_orig / png_im.shape[1]
        # width = int(png_im.shape[1] * scale)
        # height = int(png_im.shape[0] * scale)
        # dim = (width, height)
        # resized_map = cv2.resize(png_im, dim, interpolation=cv2.INTER_AREA)
        # resized_name = os.path.join(output_dir_resized, image.split('/')[-1].split('.')[0] + "_resized.png")
        # imageio.imsave(resized_name, resized_map)
        # # cropped_image = crop_image(resized_rgb)
        # # resized_map = im.resize(dim, Image.ANTIALIAS)
        #
        # x = round((height - h_orig) / 2)
        # y = round((width - w_orig) / 2)
        # cropped_image = resized_map[x:x + h_orig, y:y + w_orig]
        # image_name_cropped = os.path.join(output_dir_cropped, image.split('/')[-1].split('.')[0] + "_cropped.png")
        # print("saving image: ", image_name_cropped)
        # imageio.imsave(image_name_cropped, cropped_image)
        #
        # # vis_utils.save_depth_as_uint16png(cropped_image, image_name_cropped) #######
        # # convert grayscale to RGB
        # # gray_im = cv2.imread(image_name_cropped, 0)
        # # cm = plt.get_cmap('gist_rainbow')
        # # colored_image = cm(gray_im)
        # # colored_image_name = os.path.join(output_dir_colored, image.split('/')[-1].split('.')[0] + "_colored.png")
        # # Image.fromarray((colored_image[:, :, :3] * 255).astype(np.uint8)).save(colored_image_name)


def gt_to_sparse(maps_dir):
    RMSE = 0
    count = 0
    output_dir_sparse = os.path.join(maps_dir, "resized_sparse_500_png_uint16")
    if not os.path.isdir(output_dir_sparse):
        os.mkdir(output_dir_sparse)
    for image in glob.glob(os.path.join(maps_dir, "*.png")):
        png_im = np.array(Image.open(image)).astype("uint16")
        new_depth = np.zeros(png_im.shape).astype("uint16")
        y_idx, x_idx = np.where(png_im > 0)  # list of all the indices with pixel value 1
        chosen_pixels = random.sample(range(0, x_idx.size), k=500)  # k=int(x_idx.size * 0.1)
        ix = []
        iy = []
        for i in range(0, len(chosen_pixels)):
            rand_idx = chosen_pixels[i]  # randomly choose any element in the x_idx list
            x = x_idx[rand_idx]
            y = y_idx[rand_idx]
            new_depth[y, x] = png_im[y, x]
            ix.append(x)
            iy.append(y)
        image_name_cropped = os.path.join(output_dir_sparse, image.split('/')[-1].split('.')[0] + "_sparse.png")
        print("saving image: ", image_name_cropped)
        cv2.imwrite(image_name_cropped, new_depth)
        # imageio.imsave(image_name_cropped, new_depth)

        nx, ny = png_im.shape[1], png_im.shape[0]
        X, Y = np.meshgrid(np.arange(0, nx, 1), np.arange(0, ny, 1))

        # ix_o = np.random.randint(png_im.shape[1], size=500)
        # iy_o = np.random.randint(png_im.shape[0], size=500)
        samples = png_im[np.array(iy), np.array(ix)]

        interpolated_im = griddata((np.array(iy), np.array(ix)), samples, (Y, X), method='linear')
        interpolated_im_u = np.array(interpolated_im).astype("uint16")
        image_name_interp = os.path.join(output_dir_sparse, image.split('/')[-1].split('.')[0] + "_interp.png")
        # imageio.imsave(image_name_interp, interpolated_im)
        cv2.imwrite(image_name_interp, interpolated_im_u)

        # colorize image
        depth_norm = (interpolated_im_u - np.min(interpolated_im_u)) / \
                     (np.max(interpolated_im_u) - np.min(interpolated_im_u))
        cmap = plt.cm.jet
        depth_color = 255 * cmap(depth_norm)[:, :, :3]  # H, W, C
        interp_color = depth_color.astype('uint8')
        image_to_write = cv2.cvtColor(interp_color, cv2.COLOR_RGB2BGR)
        image_name_color = os.path.join(output_dir_sparse, image.split('/')[-1].split('.')[0] + "_interp_col.png")
        cv2.imwrite(image_name_color, image_to_write)

        # calc RMSE between gt and interp
        if 'val' in maps_dir:
            depth_gt = png_im.astype(np.float) / 256.
            depth_gt = np.expand_dims(depth_gt, -1)
            depth_interp = interpolated_im_u.astype(np.float) / 256.
            depth_interp = np.expand_dims(depth_interp, -1)
            valid_mask = depth_gt > 0
            # convert from meters to mm
            target_mm = 1e3 * depth_gt[valid_mask]
            output_mm = 1e3 * depth_interp[valid_mask]
            RMSE += np.sqrt(np.mean((output_mm - target_mm) ** 2))
            count += 1

    print(RMSE/count)


def save_depth_as_uint16(maps_dir):
    output_dir = os.path.join(maps_dir, "uint16")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for image in glob.glob(os.path.join(maps_dir, "*.png")):
        png_im = np.array(Image.open(image))
        img = (png_im * 256).astype('uint16')
        file_name = os.path.join(output_dir, image.split('/')[-1].split('.')[0] + ".png")
        cv2.imwrite(file_name, img)


def colorize_depth(maps_dir):
    img_list = []
    gt_dir = '../data/D5/depthMaps_2020_04_16/png_resized_new_val/uint16'
    gt_images = os.listdir(gt_dir)
    sparse_dir = '../data/D5/depthMaps_2020_04_16/resized_sparse_500_png_val/uint16'
    sparse_images = os.listdir(sparse_dir)
    pred_d_dir = '../data/D5/depthMaps_2020_04_16/results/pred_d_500'
    pred_d_images = os.listdir(pred_d_dir)
    pred_rgb_dir = '../data/D5/depthMaps_2020_04_16/results/pred_rgb_500'
    pred_rgb_images = os.listdir(pred_rgb_dir)
    pred_rgbd_dir = '../data/D5/depthMaps_2020_04_16/results/pred_rgbd_500'
    pred_rgbd_images = os.listdir(pred_rgbd_dir)
    linear_interp_dir = '../data/D5/depthMaps_2020_04_16/interpolated_resized_sparse_500_val/uint16'
    interp_images = os.listdir(linear_interp_dir)

    oheight, owidth = 448, 832
    transform_geometric = transforms.Compose([
        transforms.BottomCrop((oheight, owidth))])

    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4)

    for i in range(3):  # range(0, len(gt_images)):
        gt_im = np.array(Image.open(os.path.join(gt_dir, gt_images[i])))
        gt_im = gt_im.astype(np.float) / 256.
        gt_im = transform_geometric(gt_im)
        sparse_im = np.array(Image.open(os.path.join(sparse_dir, sparse_images[i])))
        sparse_im = sparse_im.astype(np.float) / 256.
        sparse_im = transform_geometric(sparse_im)
        pred_d_im = np.array(Image.open(os.path.join(pred_d_dir, pred_d_images[i])))
        pred_d_im = pred_d_im.astype(np.float) / 256.
        pred_d_im[pred_d_im > np.max(gt_im)] = 0.0
        pred_rgb_im = np.array(Image.open(os.path.join(pred_rgb_dir, pred_rgb_images[i])))
        pred_rgb_im = pred_rgb_im.astype(np.float) / 256.
        pred_rgbd_im = np.array(Image.open(os.path.join(pred_rgbd_dir, pred_rgbd_images[i])))
        pred_rgbd_im = pred_rgbd_im.astype(np.float) / 256.
        pred_rgbd_im[pred_rgbd_im > np.max(gt_im)] = 0.0
        interp_im = np.array(Image.open(os.path.join(linear_interp_dir, interp_images[i])))
        interp_im = interp_im.astype(np.float) / 256.
        interp_im = transform_geometric(interp_im)
        # interp_im[interp_im > np.max(gt_im)] = 0.0
        # diff_im = gt_im - pred_im
        depth_im = np.concatenate((sparse_im, pred_d_im, pred_rgb_im, pred_rgbd_im, gt_im, interp_im), axis=0)
        # depth_im = np.concatenate((sparse_im, pred_rgbd_im, gt_im), axis=0)
        img_list.append(depth_im)
    depth_tot = np.hstack(img_list)
    ax = plt.gca()
    im = ax.imshow(depth_tot, cmap="jet")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="7%", pad="2%")
    plt.colorbar(im, cax=cax, orientation='horizontal', label='depth [m]')
    ax.set_xticks([])
    ax.set_yticks([])
    # plt.colorbar(orientation='horizontal',  )
    plt.show()

    # for image in glob.glob(os.path.join(maps_dir, "*.png")):
    #     png_im = np.array(Image.open(image))
    #     depth = png_im.astype(np.float) / 256.
    #     plt.imshow(depth, cmap="jet")
    #     plt.xticks([]), plt.yticks([])
    #     plt.colorbar(orientation='horizontal')
    #     plt.show()


def main():
    # video_name = 'seaErraCaesarea.avi'
    # extract_video_images(video_name)
    images_dir = '../data/D5/Raw'
    # raw_to_png_orig(images_dir)
    # raw_to_png(images_dir)
    # depthmaps_dir = '../data/D5/depthMaps_2020_04_16/png'
    # depthmaps_dir = '../data/CaesareaSet/enhanced/train/sparse_depth'
    # tif_to_png(depthmaps_dir)
    # depthmaps_png = '../data/D5/depthMaps_2020_04_16/png_resized_new_val/uint16'
    # gt_to_sparse(depthmaps_png)
    depthmaps_png = '../data/D5/depthMaps_2020_04_16/resized_sparse_500_png_val/uint16'
    colorize_depth(depthmaps_png)
    # save_depth_as_uint16(depthmaps_png)
    images_dir = '../data/D5/Raw/png_resized'


if __name__ == '__main__':
    main()
