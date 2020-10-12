import glob
import os
import random
import scipy

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import rawpy
from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata

from dataloaders import transforms
from vis_utils import depth_colorize


def crop_image(image):
    h, w, c = image.shape
    orig_im = cv2.imread(
        '../data/depth_selection/val_selection_cropped/image/2011_09_26_drive_0002_sync_image_0000000005_image_02.png')
    h_orig, w_orig, c = orig_im.shape
    x = round((h - h_orig) / 2)
    y = round((w - w_orig) / 2)
    crop_img = image[x:x + h_orig, y:y + w_orig]
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


def save_depth_as_uint16(maps_dir):
    output_dir = os.path.join(maps_dir, "uint16")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for image in glob.glob(os.path.join(maps_dir, "*.tif")):
        png_im = np.array(Image.open(image))  # .convert('L'))
        img = (png_im * 256).astype('uint16')
        file_name = os.path.join(output_dir, image.split('/')[-1].split('.')[0] + ".png")
        cv2.imwrite(file_name, img)


def calc_depth_hist(main_dir):
    options = ['train', 'val', 'test']
    fig, axs = plt.subplots(3, 1)
    fig.suptitle('Depth histograms - "Nachsholim" data-set')
    j = 0
    for o in options:
        hist = 0
        total_pixels = 0
        gt_dir = os.path.join(main_dir, o)
        gt_images = sorted(os.listdir(gt_dir))
        for i in range(0, len(gt_images)):
            if os.path.isdir(os.path.join(gt_dir, gt_images[i])):
                continue
            gt_im = np.array(Image.open(os.path.join(gt_dir, gt_images[i])))
            gt_im = gt_im.astype(np.float) / 256.
            gt_im = gt_im[gt_im > 0.0]
            gt_im = gt_im[gt_im <= 15.0]
            total_pixels += len(gt_im)
            hist += np.bincount(gt_im.astype('int64').ravel(), minlength=16)
        axs[j].plot(hist / total_pixels * 100, label=o + ' set')
        axs[j].legend(loc='upper right')
        if j == 2:
            axs[j].set(xlabel='depth [m]')
        if j == 1:
            axs[j].set(ylabel='pixels [%]')
        axs[j].grid(True)
        j += 1
    plt.show()


def truncate_depth_maps(maps_dir, value):
    output_dir = os.path.join(maps_dir, "truncate")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for image in glob.glob(os.path.join(maps_dir, "*.png")):
        png_im = np.array(Image.open(image))
        png_im = png_im.astype(np.float) / 256.
        png_im[png_im > value] = 0.0
        img = (png_im * 256).astype('uint16')
        file_name = os.path.join(output_dir, image.split('/')[-1])
        cv2.imwrite(file_name, img)


def gt_to_sparse(maps_dir, n_samples):
    output_dir_sparse = os.path.join(maps_dir, "sparse_" + str(n_samples))
    if not os.path.isdir(output_dir_sparse):
        os.mkdir(output_dir_sparse)
    output_dir_linear_interp = os.path.join(maps_dir, "interp_" + str(n_samples))
    if not os.path.isdir(output_dir_linear_interp):
        os.mkdir(output_dir_linear_interp)
    for image in glob.glob(os.path.join(maps_dir, "*.png")):
        png_im = np.array(Image.open(image)).astype("uint16")
        new_depth = np.zeros(png_im.shape).astype("uint16")
        y_idx, x_idx = np.where(png_im > 0)  # list of all the indices with pixel value 1
        if x_idx.size < n_samples:
            continue
        chosen_pixels = random.sample(range(0, x_idx.size), k=n_samples)  # k=int(x_idx.size * 0.1)
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

        # calc linear interpolation images
        nx, ny = png_im.shape[1], png_im.shape[0]
        X, Y = np.meshgrid(np.arange(0, nx, 1), np.arange(0, ny, 1))
        samples = png_im[np.array(iy), np.array(ix)]
        interpolated_im = griddata((np.array(iy), np.array(ix)), samples, (Y, X), method='linear')
        interpolated_im_u = np.array(interpolated_im).astype("uint16")
        image_name_interp = os.path.join(output_dir_linear_interp, image.split('/')[-1].split('.')[0] + "_interp.png")
        cv2.imwrite(image_name_interp, interpolated_im_u)


def calc_errors(gt_dir, interp_dir):
    RMSE = 0
    MAE = 0
    Avg_pred = 0
    Avg_target = 0
    count = 0
    o_height, o_width = 512, 800
    transform_geometric = transforms.Compose([
        transforms.BottomCrop((o_height, o_width))])

    gt_images = sorted(os.listdir(gt_dir))
    interp_images = sorted(os.listdir(interp_dir))
    for i in range(0, len(gt_images)):
        depth_gt = np.array(Image.open(os.path.join(gt_dir, gt_images[i])))
        depth_gt = depth_gt.astype(np.float) / 256.
        depth_gt = transform_geometric(depth_gt)
        depth_gt = np.expand_dims(depth_gt, -1)
        depth_interp = np.array(Image.open(os.path.join(interp_dir, interp_images[i])))
        depth_interp = depth_interp.astype(np.float) / 256.
        depth_interp = transform_geometric(depth_interp)
        depth_interp = np.expand_dims(depth_interp, -1)

        # depth_gt[depth_gt > np.percentile(depth_gt, 90)] = 0
        valid_mask = depth_gt > 0.1
        # convert from meters to mm
        target_mm = 1e3 * depth_gt[valid_mask]
        output_mm = 1e3 * depth_interp[valid_mask]
        RMSE += np.sqrt(np.mean((output_mm - target_mm) ** 2))
        MAE += float(np.mean(np.abs(output_mm - target_mm)))
        Avg_pred += np.mean(output_mm)
        Avg_target += np.mean(target_mm)
        count += 1
    print("RMSE: ", RMSE / count)
    print("MAE: ", MAE / count)
    print("Average depth pred: ", Avg_pred / count)
    print("Average depth target: ", Avg_target / count)


def colorize_depth():
    cmap = plt.cm.jet

    img_list = []
    diff_list = []
    rgb_list = []
    pred_with_sparse_list = []
    gt_dir = '../data/Nachsholim/depth_lft/uint16/test/truncate'
    # gt_dir = '../data/SouthCarolinaCave/depthMaps/uint16/test/truncate'
    gt_images = sorted(os.listdir(gt_dir))
    sparse_dir = '../data/Nachsholim/depth_lft/sparse/test/truncate'
    # sparse_dir = '../data/SouthCarolinaCave/depthMaps/sparse/test/truncate'
    sparse_images = sorted(os.listdir(sparse_dir))
    linear_interp_dir = '../data/Nachsholim/depth_lft/interp/test/truncate'
    # linear_interp_dir = '../data/SouthCarolinaCave/depthMaps/interp/test/truncate'
    interp_images = sorted(os.listdir(linear_interp_dir))
    pred_rgb_dir = '../data/Nachsholim/results/supervised/rgb_unenhanced'
    # pred_rgb_dir = '../data/SouthCarolinaCave/results/supervised/rgb'
    pred_rgb_images = sorted(os.listdir(pred_rgb_dir))
    pred_d_dir = '../data/Nachsholim/results/supervised/d'
    # pred_d_dir = '../data/SouthCarolinaCave/results/supervised/d'
    pred_d_images = sorted(os.listdir(pred_d_dir))
    pred_gd_dir = '../data/Nachsholim/results/supervised/gd_unenhanced'
    # pred_gd_dir = '../data/SouthCarolinaCave/results/supervised/gd'
    pred_gd_images = sorted(os.listdir(pred_gd_dir))
    pred_rgbd_dir = '../data/Nachsholim/results/supervised/rgbd_unenhanced'
    # pred_rgbd_dir = '../data/SouthCarolinaCave/results/supervised/rgbd'
    pred_rgbd_images = sorted(os.listdir(pred_rgbd_dir))
    pred_rgbd_photo_dir = '../data/Nachsholim/results/supervised/rgbd_photo_unenhanced'
    # pred_rgbd_photo_dir = '../data/SouthCarolinaCave/results/supervised/rgbd_photo'
    pred_rgbd_photo_images = sorted(os.listdir(pred_rgbd_photo_dir))
    rgb_dir = '../data/Nachsholim/rgb_unenhanced/test'
    # rgb_dir = '../data/SouthCarolinaCave/cave_seaerra_lft_to1500/png/test'
    rgb_images = sorted(os.listdir(rgb_dir))

    oheight, owidth = 512, 800  # 448, 832  #
    transform_geometric = transforms.Compose([
        transforms.BottomCrop((oheight, owidth))])

    skip = 200  # 80
    # top_percent = 90
    for i in range(1):  # range(0, len(gt_images)):
        i += 1
        gt_im = np.array(Image.open(os.path.join(gt_dir, gt_images[i * skip])))
        gt_im = gt_im.astype(np.float) / 256.
        gt_im = transform_geometric(gt_im)
        # gt_im[gt_im > np.percentile(gt_im, top_percent)] = 0.0
        sparse_im = np.array(Image.open(os.path.join(sparse_dir, sparse_images[i * skip])))
        sparse_im = sparse_im.astype(np.float) / 256.
        sparse_im = transform_geometric(sparse_im)
        sparse_dilate = cv2.dilate(sparse_im, np.ones((6, 6), np.uint8), iterations=1)
        # sparse_im[sparse_im > np.percentile(gt_im, top_percent)] = 0.0
        pred_rgb_im = np.array(Image.open(os.path.join(pred_rgb_dir, pred_rgb_images[i * skip])))
        pred_rgb_im = pred_rgb_im.astype(np.float) / 256.
        diff_im_rgb = gt_im - pred_rgb_im
        # pred_rgb_im[pred_rgb_im > np.percentile(gt_im, top_percent)] = 0.0
        pred_d_im = np.array(Image.open(os.path.join(pred_d_dir, pred_d_images[i * skip])))
        pred_d_im = pred_d_im.astype(np.float) / 256.
        diff_im_d = gt_im - pred_d_im
        pred_d_sparse = pred_d_im
        pred_d_sparse[sparse_dilate > 0] = 0
        # pred_d_im[pred_d_im > np.percentile(gt_im, top_percent)] = 0.0
        pred_gd_im = np.array(Image.open(os.path.join(pred_gd_dir, pred_gd_images[i * skip])))
        pred_gd_im = pred_gd_im.astype(np.float) / 256.
        diff_im_gd = gt_im - pred_gd_im
        pred_gd_sparse = pred_gd_im
        pred_gd_sparse[sparse_dilate > 0] = 0
        # pred_gd_im[pred_gd_im > np.percentile(gt_im, top_percent)] = 0.0
        pred_rgbd_im = np.array(Image.open(os.path.join(pred_rgbd_dir, pred_rgbd_images[i * skip])))
        pred_rgbd_im = pred_rgbd_im.astype(np.float) / 256.
        diff_im_rgbd = gt_im - pred_rgbd_im
        pred_rgbd_sparse = pred_rgbd_im
        pred_rgbd_sparse[sparse_dilate > 0] = 0
        # pred_rgbd_im[pred_rgbd_im > np.percentile(gt_im, top_percent)] = 0.0
        pred_rgbd_photo_im = np.array(Image.open(os.path.join(pred_rgbd_photo_dir, pred_rgbd_photo_images[i * skip])))
        pred_rgbd_photo_im = pred_rgbd_photo_im.astype(np.float) / 256.
        diff_im_rgbd_photo = gt_im - pred_rgbd_photo_im
        pred_rgbd_photo_sparse = pred_rgbd_photo_im
        pred_rgbd_photo_sparse[sparse_dilate > 0] = 0
        # pred_rgbd_photo_im[pred_rgbd_photo_im > np.percentile(gt_im, top_percent)] = 0.0
        interp_im = np.array(Image.open(os.path.join(linear_interp_dir, interp_images[i * skip])))
        interp_im = interp_im.astype(np.float) / 256.
        interp_im = transform_geometric(interp_im)
        diff_im_interp = gt_im - interp_im
        interp_sparse = interp_im
        interp_sparse[sparse_dilate > 0] = 0
        # interp_im[interp_im > np.percentile(gt_im, top_percent)] = 0.0

        rgb_im = np.array(Image.open(os.path.join(rgb_dir, rgb_images[i * skip])))
        rgb_im = transform_geometric(rgb_im)

        depth_im = np.concatenate((sparse_dilate, pred_rgb_im, pred_d_sparse, pred_gd_sparse, pred_rgbd_sparse,
                                   pred_rgbd_photo_sparse, interp_sparse, gt_im), axis=0)
        # depth_im = np.concatenate((sparse_dilate, pred_rgbd_sparse, interp_sparse, gt_im), axis=0)
        img_list.append(depth_im)

        sparse_dilate_norm = (sparse_dilate - np.min(depth_im)) / (np.max(depth_im) - np.min(depth_im))
        sparse_dilate_colored = (255 * cmap(sparse_dilate_norm)[:, :, :3]).astype('uint8')
        # sparse_dilate_colored = depth_colorize(sparse_dilate)
        pred_rgb_norm = (pred_rgb_im - np.min(depth_im)) / (np.max(depth_im) - np.min(depth_im))
        pred_rgb_im_colored = (255 * cmap(pred_rgb_norm)[:, :, :3]).astype('uint8')
        # pred_rgb_im_colored = depth_colorize(pred_rgb_im)
        # pred_rgbd_im_colored = depth_colorize(pred_rgbd_im)
        pred_rgbd_norm = (pred_rgbd_im - np.min(depth_im)) / (np.max(depth_im) - np.min(depth_im))
        pred_rgbd_im_colored = (255 * cmap(pred_rgbd_norm)[:, :, :3]).astype('uint8')
        (pred_rgbd_im_colored[:, :, 0])[sparse_dilate > 0] = 255
        (pred_rgbd_im_colored[:, :, 1])[sparse_dilate > 0] = 255
        (pred_rgbd_im_colored[:, :, 2])[sparse_dilate > 0] = 255
        # pred_rgbd_photo_im_colored = depth_colorize(pred_rgbd_photo_im)
        pred_rgbd_photo_norm = (pred_rgbd_photo_im - np.min(depth_im)) / (np.max(depth_im) - np.min(depth_im))
        pred_rgbd_photo_im_colored = (255 * cmap(pred_rgbd_photo_norm)[:, :, :3]).astype('uint8')
        (pred_rgbd_photo_im_colored[:, :, 0])[sparse_dilate > 0] = 255
        (pred_rgbd_photo_im_colored[:, :, 1])[sparse_dilate > 0] = 255
        (pred_rgbd_photo_im_colored[:, :, 2])[sparse_dilate > 0] = 255
        # pred_d_im_colored = depth_colorize(pred_d_im)
        pred_d_norm = (pred_d_im - np.min(depth_im)) / (np.max(depth_im) - np.min(depth_im))
        pred_d_im_colored = (255 * cmap(pred_d_norm)[:, :, :3]).astype('uint8')
        (pred_d_im_colored[:, :, 0])[sparse_dilate > 0] = 255
        (pred_d_im_colored[:, :, 1])[sparse_dilate > 0] = 255
        (pred_d_im_colored[:, :, 2])[sparse_dilate > 0] = 255
        # pred_gd_im_colored = depth_colorize(pred_d_im)
        pred_gd_norm = (pred_gd_im - np.min(depth_im)) / (np.max(depth_im) - np.min(depth_im))
        pred_gd_im_colored = (255 * cmap(pred_gd_norm)[:, :, :3]).astype('uint8')
        (pred_gd_im_colored[:, :, 0])[sparse_dilate > 0] = 255
        (pred_gd_im_colored[:, :, 1])[sparse_dilate > 0] = 255
        (pred_gd_im_colored[:, :, 2])[sparse_dilate > 0] = 255
        # interp_im_colored = depth_colorize(interp_im)
        interp_norm = (interp_im - np.min(depth_im)) / (np.max(depth_im) - np.min(depth_im))
        interp_im_colored = (255 * cmap(interp_norm)[:, :, :3]).astype('uint8')
        (interp_im_colored[:, :, 0])[sparse_dilate > 0] = 255
        (interp_im_colored[:, :, 1])[sparse_dilate > 0] = 255
        (interp_im_colored[:, :, 2])[sparse_dilate > 0] = 255
        # gt_colored = depth_colorize()
        gt_norm = (gt_im - np.min(depth_im)) / (np.max(depth_im) - np.min(depth_im))
        gt_colored = (255 * cmap(gt_norm)[:, :, :3]).astype('uint8')

        pred_with_sparse_1 = np.concatenate((gt_colored, sparse_dilate_colored, sparse_dilate_colored), axis=1)
        pred_with_sparse_2 = np.concatenate((pred_d_im_colored, pred_rgb_im_colored, interp_im_colored), axis=1)
        pred_with_sparse_3 = np.concatenate((pred_rgbd_photo_im_colored, pred_rgbd_im_colored, pred_gd_im_colored),
                                            axis=1)
        pred_with_sparse = np.concatenate((pred_with_sparse_1, pred_with_sparse_2, pred_with_sparse_3), axis=0)
        pred_with_sparse_list.append(pred_with_sparse)

        diff_im = np.concatenate((diff_im_rgb, diff_im_d, diff_im_rgbd, diff_im_interp), axis=0)
        diff_list.append(diff_im)

        rgb_list.append(rgb_im)

    depth_tot = np.hstack(img_list)
    plt.figure(5)
    ax = plt.gca()
    im = ax.imshow(depth_tot, cmap="jet")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="7%", pad="2%")
    plt.colorbar(im, cax=cax, orientation='horizontal', label='depth [m]')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

    pred_with_sparse_tot = np.hstack(pred_with_sparse_list)
    plt.figure(2)
    ax4 = plt.gca()
    ax4.imshow(pred_with_sparse_tot)
    divider = make_axes_locatable(ax4)
    cax4 = divider.append_axes("bottom", size="7%", pad="2%")
    plt.colorbar(im, cax=cax4, orientation='horizontal', label='depth [m]')
    ax4.set_xticks([])
    ax4.set_yticks([])
    plt.show()

    diff_tot = np.hstack(diff_list)
    plt.figure(3)
    ax2 = plt.gca()
    im2 = ax2.imshow(diff_tot, cmap="jet")
    divider = make_axes_locatable(ax2)
    cax2 = divider.append_axes("bottom", size="7%", pad="2%")
    plt.colorbar(im2, cax=cax2, orientation='horizontal', label='depth [m]')
    ax2.set_xticks([])
    ax2.set_yticks([])
    plt.show()

    rgb_tot = np.hstack(rgb_list)
    plt.figure(4)
    ax3 = plt.gca()
    ax3.imshow(rgb_tot)
    plt.show()


def rename_files():
    dir_name = '../data/Nachsholim/rgb_unenhanced/sampled_done'
    count = 3
    # for f in os.listdir(dir_name):
    # sub_dir = os.path.join(dir_name, f)
    # if os.path.isdir(sub_dir) and 'train' not in sub_dir:
    for image in sorted(glob.glob(os.path.join(dir_name, "*.tif"))):
        image_num = image.split('/')[-1].split('.')[0].split('_')[1]
        print(image_num)
        im = Image.open(image)
        image_name = os.path.join(dir_name, "new", "input_l_" + image_num + ".png")
        imageio.imsave(image_name, im)


def compare_rgb_images(dir1, dir2):
    cmap = plt.cm.jet
    count_d = 0
    count_gd = 0
    display_d = False
    display_gd = False
    dir1_images = sorted(os.listdir(dir1))
    dir2_images = sorted(os.listdir(dir2))
    input_images = sorted(glob.glob(os.path.join('../data/Nachsholim/rgb_seaErra/test', "*.png")))
    sparse_images = sorted(glob.glob(os.path.join('../data/Nachsholim/depth_lft/sparse/test/truncate', "*.png")))
    gt_images = sorted(glob.glob(os.path.join('../data/Nachsholim/depth_lft/uint16/test/truncate', "*.png")))
    rgb_d_list = []
    rgb_gd_list = []
    diff_d_list = []
    diff_gd_list = []
    o_height, o_width = 512, 800
    transform_geometric = transforms.Compose([
        transforms.BottomCrop((o_height, o_width))])
    for i in range(len(dir1_images)):
        rmse1_str = dir1_images[i].split('_')[-1].split('.')
        rmse1 = float(rmse1_str[0] + '.' + rmse1_str[1])
        rmse2_str = dir2_images[i].split('_')[-1].split('.')
        rmse2 = float(rmse2_str[0] + '.' + rmse2_str[1])
        diff = abs(rmse2 - rmse1)
        if rmse1 < rmse2:
            count_d += 1
            if diff > 40 and len(rgb_d_list) < 8 and (count_d % 50) == 0:
                display_d = True
                output_win = np.array(Image.open(os.path.join(dir1, dir1_images[i]))).astype(np.float) / 256.
                output_lose = np.array(Image.open(os.path.join(dir2, dir2_images[i]))).astype(np.float) / 256.
        else:
            count_gd += 1
            if diff > 40 and len(rgb_gd_list) < 8 and (count_gd % 2) == 0:
                display_gd = True
                output_win = np.array(Image.open(os.path.join(dir2, dir2_images[i]))).astype(np.float) / 256.
                output_lose = np.array(Image.open(os.path.join(dir1, dir1_images[i]))).astype(np.float) / 256.

        if display_d or display_gd:
            input = transform_geometric(np.array(Image.open(input_images[i])))
            sparse = transform_geometric(np.array(Image.open(sparse_images[i]))).astype(np.float) / 256.
            sparse_dilate = cv2.dilate(sparse, np.ones((6, 6), np.uint8), iterations=1)
            gt = transform_geometric(np.array(Image.open(gt_images[i]))).astype(np.float) / 256.
            row = np.concatenate((sparse_dilate, output_win, output_lose, gt), axis=1)
            sparse_dilate_norm = (sparse_dilate - np.min(row)) / (np.max(row) - np.min(row))
            sparse_dilate_colored = (255 * cmap(sparse_dilate_norm)[:, :, :3]).astype('uint8')
            output_win_norm = (output_win - np.min(row)) / (np.max(row) - np.min(row))
            output_win_colored = (255 * cmap(output_win_norm)[:, :, :3]).astype('uint8')
            output_lose_norm = (output_lose - np.min(row)) / (np.max(row) - np.min(row))
            output_lose_colored = (255 * cmap(output_lose_norm)[:, :, :3]).astype('uint8')
            gt_norm = (gt - np.min(row)) / (np.max(row) - np.min(row))
            gt_colored = (255 * cmap(gt_norm)[:, :, :3]).astype('uint8')
            diff_im = output_win - output_lose
            diff_im[gt == 0] = 0
            row_colored = np.concatenate((input, sparse_dilate_colored, output_win_colored, output_lose_colored,
                                          gt_colored), axis=1)
            if display_d:
                rgb_d_list.append(row_colored)
                diff_d_list.append(diff_im)
                display_d = False
            else:
                rgb_gd_list.append(row_colored)
                diff_gd_list.append(diff_im)
                display_gd = False

    print(count_d)
    print(count_gd)

    rgb_d_tot = np.vstack(rgb_d_list)
    plt.figure(6)
    ax6 = plt.gca()
    ax6.imshow(rgb_d_tot)
    ax6.set_xticks([])
    ax6.set_yticks([])
    plt.show()

    rgb_gd_tot = np.vstack(rgb_gd_list)
    plt.figure(7)
    ax7 = plt.gca()
    ax7.imshow(rgb_gd_tot)
    ax7.set_xticks([])
    ax7.set_yticks([])
    plt.show()

    diff_d_tot = np.vstack(diff_d_list)
    plt.figure(8)
    ax8 = plt.gca()
    im8 = ax8.imshow(diff_d_tot, cmap="jet")
    divider = make_axes_locatable(ax8)
    cax2 = divider.append_axes("bottom", size="7%", pad="2%")
    plt.colorbar(im8, cax=cax2, orientation='horizontal', label='depth [m]')
    ax8.set_xticks([])
    ax8.set_yticks([])
    plt.show()

    diff_gd_tot = np.vstack(diff_gd_list)
    plt.figure(9)
    ax9 = plt.gca()
    im9 = ax9.imshow(diff_gd_tot, cmap="jet")
    divider = make_axes_locatable(ax9)
    cax3 = divider.append_axes("bottom", size="7%", pad="2%")
    plt.colorbar(im9, cax=cax3, orientation='horizontal', label='depth [m]')
    ax9.set_xticks([])
    ax9.set_yticks([])
    plt.show()


def main():
    # depthmaps_dir = '../data/D5/depthMaps_2020_04_16/png'
    # depthmaps_dir = '../data/SouthCarolinaCave/cave_seaerra_lft_to1500/'
    # tif_to_png(depthmaps_dir)

    # depthmaps_dir = '../data/Nachsholim/depth_lft/tif/test'
    # save_depth_as_uint16(depthmaps_dir)

    # main_dir = '../data/Nachsholim/depth_lft/uint16'
    # calc_depth_hist(main_dir)

    # depthmaps_dir = '../data/Nachsholim/depth_lft/uint16/test'
    # value = 7.0
    # truncate_depth_maps(depthmaps_dir, value)

    # depthmaps_png = '../data/Nachsholim/depth_lft/uint16/test/truncate'
    # n_samples = 500
    # gt_to_sparse(depthmaps_png, n_samples)

    # gt_dir = '../data/SouthCarolinaCave/depthMaps/uint16/test/truncate'
    # interp_dir = '../data/SouthCarolinaCave/depthMaps/interp/test/truncate'
    # calc_errors(gt_dir, interp_dir)

    rename_files()

    # colorize_depth()

    output_d = '../pretrained_models/supervised/nachsholim/test_results/mode=dense.data=nachsholim.input=d.resnet18' \
               '.epochs35.criterion=l2.lr=0.0001.bs=2.wd=0.pretrained=True.jitter=0.1.time=2020-10-06@12' \
               '-58_test_with_rmse/val_output'
    output_gd = '../pretrained_models/supervised/nachsholim/test_results/mode=dense.data=nachsholim.input=gd.resnet18' \
                '.epochs35.criterion=l2.lr=0.0001.bs=2.wd=0.pretrained=True.jitter=0.1.time=2020-10-06@13' \
                '-14_seaErra_test_with_rmse/val_output'
    # compare_rgb_images(output_d, output_gd)


if __name__ == '__main__':
    main()

# def make_train_val_sets(input_dir, depth_dir):
#     input_val_dir = os.path.join(input_dir, '../val')
#     if not os.path.isdir(input_val_dir):
#         os.mkdir(input_val_dir)
#     depth_val_dir = os.path.join(depth_dir, '../val')
#     if not os.path.isdir(depth_val_dir):
#         os.mkdir(depth_val_dir)
#     input_images = sorted(glob.glob(os.path.join(input_dir, "*.png")))
#     depth_images = sorted(glob.glob(os.path.join(depth_dir, "*.png")))
#     n_samples = len(os.listdir(input_dir))
#     n_val = round(n_samples*0.2)
#     selected_val = random.sample(range(1, n_samples), k=n_val)
#     for i in selected_val:
#         shutil.move(input_images[i], input_val_dir)
#         shutil.move(depth_images[i], depth_val_dir)

# video_name = 'seaErraCaesarea.avi'
# extract_video_images(video_name)
# images_dir = '../data/D5/Raw'
# raw_to_png_orig(images_dir)
# raw_to_png(images_dir)

# def raw_to_png(images_dir):
#     output_dir = os.path.join(images_dir, "png_resized_new")
#     if not os.path.isdir(output_dir):
#         os.mkdir(output_dir)
#     for image in glob.glob(os.path.join(images_dir, "*NEF")):
#         with rawpy.imread(image) as raw:
#             rgb = raw.postprocess()
#             orig_im = cv2.imread('../data/CaesareaSet/enhanced/input/seaErra_in_01914.png')
#             h_orig, w_orig, c = orig_im.shape
#             scale = w_orig / rgb.shape[1]
#             # scale = 0.1648
#             width = int(rgb.shape[1] * scale)
#             height = int(rgb.shape[0] * scale)
#             dim = (width, height)
#             resized_rgb = cv2.resize(rgb, dim, interpolation=cv2.INTER_AREA)
#             # cropped_image = crop_image(resized_rgb)
#             image_name_cropped = os.path.join(output_dir, image.split('/')[-1].split('.')[0] + ".png")
#             print("saving image: ", image_name_cropped)
#             imageio.imsave(image_name_cropped, resized_rgb)
