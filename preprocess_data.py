import glob
import os
import random
import shutil

import cv2
import imageio
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
import rawpy
import scipy
from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn import datasets
from dataloaders import transforms


o_height, o_width = 416, 736 # 352, 640 # 416, 736 # 512, 800
transform_geometric = transforms.Compose([
    transforms.CenterCrop((o_height, o_width))])


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
    for image in sorted(glob.glob(os.path.join(maps_dir, "*.tif"))):
        # for converting Tif to Png from terminal, run:
        # gdal_translate -of PNG <<current_image>> <<new_image_name>>
        im = Image.open(image)
        image_name = os.path.join(output_dir, image.split('/')[-1].split('.')[0] + ".png")
        imageio.imsave(image_name, im)
        # png_im = np.array(Image.open(image_name)).astype("uint16")
        # orig_im = cv2.imread(
        #     '../data/depth_selection/val_selection_cropped/image/2011_09_26_drive_0002_sync_image_0000000005_image_02.png')
        # h_orig, w_orig, c = orig_im.shape
        # scale = w_orig / png_im.shape[1]
        # width = int(png_im.shape[1] * scale)
        # height = int(png_im.shape[0] * scale)
        # dim = (width, height)
        # resized_map = cv2.resize(png_im, dim, interpolation=cv2.INTER_AREA)
        # resized_name = os.path.join(output_dir_resized, image.split('/')[-1].split('.')[0] + "_resized.png")
        # imageio.imsave(resized_name, resized_map)


def resize_images(images_dir, factor):
    output_dir = os.path.join(images_dir, "resized")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for image in glob.glob(os.path.join(images_dir, "*.png")):
        im = np.array(Image.open(image))  # .astype("uint16")
        resized_map = cv2.resize(im, (0, 0), fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        resized_name = os.path.join(output_dir, image.split('/')[-1].split('.')[0] + "_resized.png")
        imageio.imsave(resized_name, resized_map)


def save_depth_as_uint16(maps_dir):
    output_dir = os.path.join(maps_dir, "uint16")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for image in sorted(glob.glob(os.path.join(maps_dir, "*.png"))):
        png_im = np.array(Image.open(image)).astype(np.float) / 1000.  # .convert('L'))
        img = (png_im * 256).astype('uint16')
        # image_name = image.split('/')[-1].split('.')[0]
        image_name = image.split('/')[-1].split('.')[0] + '.' +  image.split('/')[-1].split('.')[1]
        file_name = os.path.join(output_dir, image_name + ".png")
        cv2.imwrite(file_name, img)


def calc_depth_hist(main_dir):
    options = ['train', 'val', 'test']
    fig, axs = plt.subplots(3, 1)
    fig2, axs2 = plt.subplots(1, 1)
    fig.suptitle('Depth histograms - "Nachsholim" data-set')
    fig2.suptitle('Depth histograms - "Nachsholim" data-set')
    j = 0
    all_hist = 0
    all_pixels = 0
    max_gt = 0
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
            if max(gt_im) > max_gt:
                max_gt = max(gt_im)
            gt_im = gt_im[gt_im <= 50.0]
            total_pixels += len(gt_im)
            all_pixels += len(gt_im)
            hist += np.bincount(gt_im.astype('int64').ravel(), minlength=51)
            all_hist += np.bincount(gt_im.astype('int64').ravel(), minlength=51)
        axs[j].plot(hist / total_pixels * 100, label=o + ' set')
        axs[j].legend(loc='upper right')
        if j == 2:
            axs[j].set(xlabel='depth [m]')
        if j == 1:
            axs[j].set(ylabel='pixels [%]')
        axs[j].grid(True)
        j += 1
    axs2.plot(all_hist / all_pixels * 100, label="total set")
    axs2.legend(loc='upper right')
    axs2.set(xlabel='depth [m]')
    axs2.set(ylabel='pixels [%]')
    axs2.grid(True)
    fig2.set_size_inches(6, 5)
    plt.show()


def truncate_depth_maps(maps_dir, value):
    subdir_name = "truncate_" + str(value)
    output_dir = os.path.join(maps_dir, subdir_name)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for image in sorted(glob.glob(os.path.join(maps_dir, "*.png"))):
        png_im = np.array(Image.open(image))
        png_im = png_im.astype(np.float) / 256.
        png_im[png_im > value] = 0.0
        img = (png_im * 256).astype('uint16')
        file_name = os.path.join(output_dir, image.split('/')[-1])
        cv2.imwrite(file_name, img)


def gt_to_sparse(maps_dir, n_samples, interp):
    output_dir_sparse = os.path.join(maps_dir, "sparse_" + str(n_samples))
    if not os.path.isdir(output_dir_sparse):
        os.mkdir(output_dir_sparse)
    output_dir_linear_interp = os.path.join(maps_dir, "interp_" + str(n_samples))
    if not os.path.isdir(output_dir_linear_interp):
        os.mkdir(output_dir_linear_interp)
    for image in glob.glob(os.path.join(maps_dir, "*.png")):
        png_im = np.array(Image.open(image)).astype("uint16")
        new_depth = np.zeros(png_im.shape).astype("uint16")
        y_idx, x_idx = np.where(png_im >= 0)  # list of all the indices with pixel value 1
        if x_idx.size < n_samples:
            print(image)
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
        if interp:
            nx, ny = png_im.shape[1], png_im.shape[0]
            X, Y = np.meshgrid(np.arange(0, nx, 1), np.arange(0, ny, 1))
            samples = png_im[np.array(iy), np.array(ix)]
            interpolated_im = griddata((np.array(iy), np.array(ix)), samples, (Y, X), method='linear')
            interpolated_im_u = np.array(interpolated_im).astype("uint16")
            image_name_interp = os.path.join(output_dir_linear_interp, image.split('/')[-1].split('.')[0] + "_interp.png")
            cv2.imwrite(image_name_interp, interpolated_im_u)


def gt_to_sparse_based_slam(gt_dir, slam_dir):
    output_dir = os.path.join(gt_dir, "manual_slam")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    gt_images = sorted(glob.glob(os.path.join(gt_dir, "*.png")))
    slam_images = sorted(glob.glob(os.path.join(slam_dir, "*.png")))
    for i in range(0, len(gt_images)):
        # first = int(gt_images[i].split('/')[-1].split('_')[-1].split('.')[0])
        # second = int(rgb_images[i].split('/')[-1].split('_')[-1].split('.')[0])
        # if first != second:
        #     print(first)
        gt_im = np.array(Image.open(gt_images[i])).astype("uint16")
        slam_im = np.array(Image.open(slam_images[i])).astype("uint16")
        gt_im[slam_im == 0] = 0
        image_name = os.path.join(output_dir, 'sparse_based_slam_' + gt_images[i].split('/')[-1].split('_')[-1])
        cv2.imwrite(image_name, gt_im)


def sparse_to_interp(sparse_dir):
    output_dir_linear_interp = os.path.join(sparse_dir, "interp")
    if not os.path.isdir(output_dir_linear_interp):
        os.mkdir(output_dir_linear_interp)
    for image in sorted(glob.glob(os.path.join(sparse_dir, "*.png"))):
        png_im = np.array(Image.open(image)).astype(np.float) / 256.
        png_im = transform_geometric(png_im)
        png_im[png_im > 6] = 0
        iy, ix = np.where(png_im > 0)
        # iy, ix = np.where(np.logical_and(png_im > 0, png_im <= 15 * 256))
        if len(iy) <= 2:
            print(image.split('/')[-1].split('.')[0])
            continue
        nx, ny = png_im.shape[1], png_im.shape[0]
        X, Y = np.meshgrid(np.arange(0, nx, 1), np.arange(0, ny, 1))
        samples = png_im[np.array(iy), np.array(ix)]
        interpolated_im = griddata((np.array(iy), np.array(ix)), samples, (Y, X), method='linear', fill_value='nan')
        interpolated_im_u = np.array(interpolated_im * 256).astype("uint16")
        image_name_interp = os.path.join(output_dir_linear_interp, image.split('/')[-1].split('.')[0] + "_interp.png")
        # present_depth_map(interpolated_im_u)
        cv2.imwrite(image_name_interp, interpolated_im_u)


def calc_errors(gt_dir, interp_dir, output_dir, max_depth):
    RMSE = 0
    RMSE_3 = 0
    RMSE_6 = 0
    RMSE_3_interp = 0
    RMSE_6_interp = 0
    RMSE_3_count = 0
    RMSE_6_count = 0
    diff_3_count = 0
    diff_6_count = 0
    diff_count = 0
    var_count = 0
    diff_3_count_interp = 0
    diff_6_count_interp = 0
    diff_count_interp = 0
    diff_count_var = 0
    MAE = 0
    Avg_pred = 0
    Avg_target = 0
    absrel = 0
    pearson = 0
    silog = 0
    count = 0
    count_3 = 0

    gt_images = sorted(glob.glob(os.path.join(gt_dir, "*.png")))
    interp_images = sorted(glob.glob(os.path.join(interp_dir, "*.png")))
    output_images = sorted(glob.glob(os.path.join(output_dir, "*.png")))
    output_var_images = sorted(glob.glob(os.path.join(output_dir, '..', 'val_output_var', "*.tif")))
    valid_all = 0
    # valid_3 = 0
    # diff_3_bin = 0
    # valid_6 = 0
    rmse_3 = []
    rmse_6 = []

    # valid_3_var = 0
    # valid_6_var = 0
    RMSE_3_var = 0
    RMSE_6_var = 0
    diff_3_count_var = 0
    diff_6_count_var = 0
    up_cover = 0

    min_gt = max_depth
    # fig, axs = plt.subplots(1, 2)

    j=0
    for i in range(0, len(output_images)):
        # if i+j != int(output_images[i].split('_')[-3]):
        #     j+=1
            # continue
        depth_gt = np.array(Image.open(gt_images[i+j]))
        depth_gt = depth_gt.astype(np.float) / 256.
        depth_gt = transform_geometric(depth_gt)
        depth_gt[depth_gt > max_depth] = 0
        # gt_im = depth_gt
        depth_gt = np.expand_dims(depth_gt, -1)
        depth_interp = np.array(Image.open(interp_images[i]))
        depth_interp = depth_interp.astype(np.float) / 256.
        depth_interp = transform_geometric(depth_interp)
        # interp_im = depth_interp
        depth_interp = np.expand_dims(depth_interp, -1)
        depth_output = np.array(Image.open(output_images[i]))
        depth_output = depth_output.astype(np.float) / 256.
        depth_output = transform_geometric(depth_output)
        # out_im = depth_output
        depth_output = np.expand_dims(depth_output, -1)
        depth_var = np.array(Image.open(output_var_images[i])).astype(np.float)
        depth_var = np.sqrt(np.exp(depth_var))
        depth_var = transform_geometric(depth_var)
        depth_var = np.expand_dims(depth_var, -1)
        # row = np.concatenate((gt_im, interp_im, out_im), axis=1)
        # out_im[gt_im == 0] = 0
        # row = np.concatenate((row, out_im), axis=1)
        # gt_im[interp_im == 0] = 0
        # interp_im[gt_im == 0] = 0
        # out_im[gt_im == 0] = 0

        depth_gt[depth_interp == 0] = 0

        valid_pixels = len(depth_gt[depth_gt > 0])
        valid_all += valid_pixels
        # row = np.concatenate((row, interp_im, out_im), axis=1)

        # plt.figure(1)
        # plt.title('     original gt |   output interp   |   output RGBd         output RGBd masked | intersection gt-interp | output RGBd masked')
        # ax1 = plt.gca()
        # im1 = ax1.imshow(row, cmap="jet")
        # divider = make_axes_locatable(ax6)
        # cax2 = divider.append_axes("bottom", size="7%", pad="2%")
        # plt.colorbar(im1, cax=cax2, orientation='horizontal', label='depth [m]')
        # ax1.set_xticks([])
        # ax1.set_yticks([])
        # plt.show()
        # if np.min(depth_gt[depth_gt > 0]) < min_gt:
        #     min_gt = np.min(depth_gt[depth_gt > 0])
        valid_mask = depth_gt > 0.5
        if valid_mask.any():
            valid_mask1 = depth_gt <=3
            valid_mask2 = depth_gt > 3
            valid_mask_var_1 = depth_var < 0.1*depth_output
            valid_mask_var_2 = depth_var < 0.1*depth_output
            valid_mask_var = depth_var < 0.1*depth_output
            # convert from meters to mm
            target_mm = 1e3 * depth_gt[valid_mask]
            output_mm = 1e3 * depth_output[valid_mask]
            diff = output_mm - target_mm
            diff_count += len(diff[diff > 0.1 * target_mm]) * 100 / len(diff)
            var_count += len(depth_var[depth_var > 0.5]) * 100 / (512*800)
            interp_mm = 1e3 * depth_interp[valid_mask]
            diff_interp = interp_mm - target_mm
            diff_count_interp += len(diff_interp[diff_interp > 0.1 * target_mm]) * 100 / len(diff_interp)
            target_mm_var = 1e3 * depth_gt[valid_mask*valid_mask_var]
            output_mm_var = 1e3 * depth_output[valid_mask*valid_mask_var]
            output_var_coverage = depth_output[valid_mask_var]
            gt_coverage = depth_gt[valid_mask]
            up_cover += 100* (len(output_var_coverage)-len(gt_coverage))/len(gt_coverage)
            diff_var = output_mm_var - target_mm_var
            diff_count_var += len(diff_var[diff_var > 0.1 * target_mm_var]) * 100 / len(diff)

            target_mm_3 = 1e3 * depth_gt[valid_mask * valid_mask1]
            output_mm_3 = 1e3 * depth_output[valid_mask * valid_mask1]
            interp_mm_3 = 1e3 * depth_interp[valid_mask * valid_mask1]
            target_mm_3_var = 1e3 * depth_gt[valid_mask * valid_mask1 * valid_mask_var_1]
            output_mm_3_var = 1e3 * depth_output[valid_mask * valid_mask1 * valid_mask_var_1]

            # up to 3 meters

            diff_3 = output_mm_3 - target_mm_3
            diff_3_interp = interp_mm_3 - target_mm_3
            # valid_3 += len(diff_3)
            diff_3_count += len(diff_3[diff_3>100])  * 100 / len(diff_3)
            diff_3_count_interp += len(diff_3_interp[diff_3_interp > 100]) * 100 / len(diff_3_interp)
            # to count by pixels
            # diff_3_bin += np.bincount(diff_3[(diff_3>100) & (diff_3 <=1500)].astype('int64').ravel(), minlength=1501)
            # rmse_3.append(diff_3 ** 2)
            rmse_3 = float(np.sqrt(np.mean(diff_3 ** 2)))
            rmse_3_interp = float(np.sqrt(np.mean(diff_3_interp ** 2)))
            RMSE_3 += rmse_3
            RMSE_3_interp += rmse_3_interp

            count_3 += 1

            diff_3_var = output_mm_3_var - target_mm_3_var
            # valid_3_var += len(diff_3_var)
            diff_3_count_var += len(diff_3_var[diff_3_var>100])  * 100 / len(diff_3)
            RMSE_3_var += float(np.sqrt(np.mean(diff_3_var ** 2)))


            target_mm_6 = 1e3 * depth_gt[valid_mask2]
            output_mm_6 = 1e3 * depth_output[valid_mask2]
            interp_mm_6 = 1e3 * depth_interp[valid_mask2]
            target_mm_6_var = 1e3 * depth_gt[valid_mask2 * valid_mask_var_2]
            output_mm_6_var = 1e3 * depth_output[valid_mask2 * valid_mask_var_2]
            diff_6 = output_mm_6 - target_mm_6
            diff_6_interp = interp_mm_6 - target_mm_6
            # valid_6 += len(diff_6)
            diff_6_count += len(diff_6[diff_6>300]) * 100 / len(diff_6)
            diff_6_count_interp += len(diff_6_interp[diff_6_interp > 300]) * 100 / len(diff_6_interp)
            rmse_6 = float(np.sqrt(np.mean(diff_6 ** 2)))
            rmse_6_interp = float(np.sqrt(np.mean(diff_6_interp ** 2)))
            RMSE_6 += rmse_6
            RMSE_6_interp += rmse_6_interp

            diff_6_var = output_mm_6_var - target_mm_6_var
            # valid_6_var += len(diff_6_var)
            diff_6_count_var += len(diff_6_var[diff_6_var > 300])  * 100 / len(diff_6)
            RMSE_6_var += float(np.sqrt(np.mean(diff_6_var ** 2)))

            target_mm = 1e3 * depth_gt[valid_mask]
            output_mm = 1e3 * depth_output[valid_mask]
            interp_mm = 1e3 * depth_interp[valid_mask]
            diff = output_mm - target_mm
            diff_interp = interp_mm - target_mm
            # diff_30 = diff[diff > 300]

            RMSE += float(np.sqrt(np.mean(diff ** 2)))  # * valid_pixels
            RMSE += float(np.sqrt(np.mean(diff_interp ** 2)))

            MAE += float(np.mean(np.abs(diff)))
            # MAE_30 += float(np.mean(np.abs(diff_30))) if len(diff_30) > 0 else 0
            Avg_pred += np.mean(output_mm)
            Avg_target += np.mean(target_mm)
            absrel += float(np.mean(np.abs(diff) / target_mm))
            pearson += scipy.stats.pearsonr(output_mm, target_mm)[0]
            err_log = np.log(target_mm) - np.log(output_mm)
            normalized_squared_log = np.mean(err_log ** 2)
            log_mean = np.mean(err_log)
            silog += normalized_squared_log - log_mean * log_mean
            count += 1
    print("RMSE: ", RMSE / count)
    print("RMSE 3m: ", RMSE_3 / count)
    # axs[1].plot(diff_3_bin*100 / valid_3, label="total set")
    print("RMSE 6m: ", RMSE_6 / count)
    print("Diff over 10: ", diff_3_count /count) # for counting pixels * 100 / valid_3)
    print("Diff over 30: ", diff_6_count / count)
    print("Diff over 10 percent: ", diff_count / count) # for counting pixels * 100 / valid_6)
    print("up_cover: ", up_cover / count)

    print('after filtering variance')
    print("RMSE 3m: ", RMSE_3_var / count)
    # axs[1].plot(diff_3_bin*100 / valid_3, label="total set")
    print("RMSE 6m: ", RMSE_6_var / count)
    print("Diff over 10: ", diff_3_count_var /count) # for counting pixels * 100 / valid_3)
    print("Diff over 30: ", diff_6_count_var / count)
    print("Diff over 10 percent: ", diff_count_var / count)
    # linear interp
    print("RMSE 3m: ", RMSE_3_interp / count)
    # axs[1].plot(diff_3_bin*100 / valid_3, label="total set")
    print("RMSE 6m: ", RMSE_6_interp / count)
    print("Diff over 10: ", diff_3_count_interp /count) # for counting pixels * 100 / valid_3)
    print("Diff over 30: ", diff_6_count_interp / count) # for counting pixels * 100 / valid_6)
    print("Diff over 10 percent: ", diff_count_interp / count)

    print("MAE: ", MAE / count)
    # print("Average depth pred: ", Avg_pred / count)
    # print("Average depth target: ", Avg_target / count)
    print("absrel: ", absrel / count)
    # print("pearson: ", pearson / count)
    print("silog: ", silog / count)


def colorize_depth():
    cmap = plt.cm.jet
    img_list = []
    diff_list = []
    rgb_list = []
    pred_with_sparse_list = []
    gt_dir = '../data/Nachsholim/depth_lft/uint16/test/truncate'
    gt_images = sorted(os.listdir(gt_dir))
    sparse_dir = '../data/Nachsholim/depth_lft/sparse/test'
    sparse_images = sorted(os.listdir(sparse_dir))
    linear_interp_dir = '../data/Nachsholim/depth_lft/interp/test'
    interp_images = sorted(os.listdir(linear_interp_dir))
    pred_rgb_dir = '../pretrained_models/supervised/nachsholim/test_truncate_6/mode=dense.data=nachsholim.input=rgb.resnet18.epochs35.criterion=l2.lr=0.0001.bs=2.wd=0.pretrained=True.jitter=0.1.time=2020-10-26@10-24_seaErra_truncate6/val_output'
    pred_rgb_images = sorted(os.listdir(pred_rgb_dir))
    pred_d_dir = '../pretrained_models/supervised/nachsholim/test_truncate_6/mode=dense.data=nachsholim.input=d.resnet18.epochs35.criterion=l2.lr=0.0001.bs=2.wd=0.pretrained=True.jitter=0.1.time=2020-10-21@14-44_truncate6/val_output'
    pred_d_images = sorted(os.listdir(pred_d_dir))
    pred_gd_dir = '../pretrained_models/supervised/nachsholim/test_truncate_6/mode=dense.data=nachsholim.input=gd.resnet18.epochs35.criterion=l2.lr=0.0001.bs=2.wd=0.pretrained=True.jitter=0.1.time=2020-10-25@11-24_seaErra_truncate6/val_output'
    pred_gd_images = sorted(os.listdir(pred_gd_dir))
    pred_rgbd_dir = '../pretrained_models/supervised/nachsholim/test_truncate_6/mode=dense.data=nachsholim.input=rgbd.resnet18.epochs35.criterion=l2.lr=0.0001.bs=2.wd=0.pretrained=True.jitter=0.1.time=2020-10-21@15-10_seaErra_truncate6/val_output'
    pred_rgbd_images = sorted(os.listdir(pred_rgbd_dir))
    pred_rgbd_photo_dir = '../pretrained_models/supervised/nachsholim/test_truncate_6/mode=dense+photo.w1=0.1.w2=0.1.data=nachsholim.input=rgbd.resnet18.epochs40.criterion=l2.lr=0.0001.bs=2.wd=0.pretrained=True.jitter=0.1.time=2020-10-21@15-16_seaErra_truncate6/val_output'
    pred_rgbd_photo_images = sorted(os.listdir(pred_rgbd_photo_dir))
    rgb_dir = '../data/Nachsholim/rgb_seaErra/test'
    rgb_images = sorted(os.listdir(rgb_dir))

    skip = 200  # 80
    for i in range(1):  # range(0, len(gt_images)):
        i += 1
        gt_im = np.array(Image.open(os.path.join(gt_dir, gt_images[i * skip])))
        gt_im = transform_geometric(gt_im.astype(np.float) / 256.)
        sparse_im = np.array(Image.open(os.path.join(sparse_dir, sparse_images[i * skip])))
        sparse_im = transform_geometric(sparse_im.astype(np.float) / 256.)
        sparse_dilate = cv2.dilate(sparse_im, np.ones((3, 3), np.uint8), iterations=1)
        pred_rgb_im = np.array(Image.open(os.path.join(pred_rgb_dir, pred_rgb_images[i * skip])))
        pred_rgb_im = pred_rgb_im.astype(np.float) / 256.
        diff_im_rgb = gt_im - pred_rgb_im
        pred_d_im = np.array(Image.open(os.path.join(pred_d_dir, pred_d_images[i * skip])))
        pred_d_im = pred_d_im.astype(np.float) / 256.
        diff_im_d = gt_im - pred_d_im
        diff_im_d[gt_im == 0] = 0
        pred_d_sparse = pred_d_im
        pred_d_sparse[sparse_dilate > 0] = 0
        pred_gd_im = np.array(Image.open(os.path.join(pred_gd_dir, pred_gd_images[i * skip])))
        pred_gd_im = pred_gd_im.astype(np.float) / 256.
        diff_im_gd = gt_im - pred_gd_im
        diff_im_gd[gt_im == 0] = 0
        pred_gd_sparse = pred_gd_im
        pred_gd_sparse[sparse_dilate > 0] = 0
        pred_rgbd_im = np.array(Image.open(os.path.join(pred_rgbd_dir, pred_rgbd_images[i * skip])))
        pred_rgbd_im = pred_rgbd_im.astype(np.float) / 256.
        diff_im_rgbd = gt_im - pred_rgbd_im
        diff_im_rgbd[gt_im == 0] = 0
        pred_rgbd_sparse = pred_rgbd_im
        pred_rgbd_sparse[sparse_dilate > 0] = 0
        pred_rgbd_photo_im = np.array(Image.open(os.path.join(pred_rgbd_photo_dir, pred_rgbd_photo_images[i * skip])))
        pred_rgbd_photo_im = pred_rgbd_photo_im.astype(np.float) / 256.
        diff_im_rgbd_photo = gt_im - pred_rgbd_photo_im
        pred_rgbd_photo_sparse = pred_rgbd_photo_im
        pred_rgbd_photo_sparse[sparse_dilate > 0] = 0
        interp_im = np.array(Image.open(os.path.join(linear_interp_dir, interp_images[i * skip])))
        interp_im = transform_geometric(interp_im.astype(np.float) / 256.)
        diff_im_interp = gt_im - interp_im
        interp_sparse = interp_im
        interp_sparse[sparse_dilate > 0] = 0

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

        diff_im = np.concatenate((diff_im_d, diff_im_gd, diff_im_rgbd), axis=0)
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
    dir_name = '../data/Nachsholim/depth_lft/uint16_slam/train/truncate/filled_maps_all'
    for image in sorted(glob.glob(os.path.join(dir_name, "*.png"))):
        # image_num = image.split('/')[-1].split('.')[0].split('_')[1]
        # print(image_num)
        # im = Image.open(image)
        image_name = os.path.join(dir_name, image.split('/')[-1].split('.')[0] + "_filled" + ".png")
        os.rename(image, image_name)


def compare_rgb_images(dir1, dir2):
    cmap = plt.cm.jet
    count_d = 0
    count_gd = 0
    display_d = False
    display_gd = False
    dir1_images = sorted(os.listdir(dir1))
    dir2_images = sorted(os.listdir(dir2))
    input_images = sorted(glob.glob(os.path.join('../data/Nachsholim/rgb_unenhanced_slam/test', "*.png")))
    sparse_images = sorted(glob.glob(os.path.join('../data/Nachsholim/depth_lft/sparse_manual_slam/test/', "*.png")))
    interp_images = sorted(glob.glob(os.path.join('../data/Nachsholim/depth_lft/sparse_manual_slam/interp_test', "*.png")))
    gt_images = sorted(glob.glob(os.path.join('../data/Nachsholim/depth_lft/uint16_slam/test/truncate', "*.png")))
    rgb_d_list = []
    rgb_gd_list = []
    depth_d_list = []
    depth_gd_list = []
    diff_d_list = []
    diff_gd_list = []
    rmse1_list = []
    rmse2_list = []
    rmsed_win = []
    rmsed_lose = []
    rmsegd_win = []
    rmsegd_lose = []
    for i in range(len(dir1_images)):
        rmse1_str = dir1_images[i].split('_')[-1].split('.')
        rmse1 = float(rmse1_str[0] + '.' + rmse1_str[1])
        rmse2_str = dir2_images[i].split('_')[-1].split('.')
        rmse2 = float(rmse2_str[0] + '.' + rmse2_str[1])
        rmse1_list.append(rmse1)
        rmse2_list.append(rmse2)
        diff = abs(rmse2 - rmse1)
        if rmse1 < rmse2:
            count_d += 1
            if diff > 40 and len(rgb_d_list) < 6 and (count_d % 150) == 0:
                display_d = True
                output_win = np.array(Image.open(os.path.join(dir1, dir1_images[i]))).astype(np.float) / 256.
                output_lose = np.array(Image.open(os.path.join(dir2, dir2_images[i]))).astype(np.float) / 256.
                rmsed_win.append(rmse1)
                rmsegd_lose.append(rmse2)
        else:
            count_gd += 1
            if diff > 40 and len(rgb_gd_list) < 6 and (count_gd % 30) == 0:
                display_gd = True
                output_win = np.array(Image.open(os.path.join(dir2, dir2_images[i]))).astype(np.float) / 256.
                output_lose = np.array(Image.open(os.path.join(dir1, dir1_images[i]))).astype(np.float) / 256.
                rmsegd_win.append(rmse2)
                rmsed_lose.append(rmse1)

        if display_d or display_gd:
            input = transform_geometric(np.array(Image.open(input_images[i])))
            sparse = transform_geometric(np.array(Image.open(sparse_images[i]))).astype(np.float) / 256.
            sparse_dilate = cv2.dilate(sparse, np.ones((6, 6), np.uint8), iterations=1)
            # interp = transform_geometric(np.array(Image.open(interp_images[i]))).astype(np.float) / 256.
            gt = transform_geometric(np.array(Image.open(gt_images[i]))).astype(np.float) / 256.
            output_win_gt = np.array(Image.open(os.path.join(dir2, dir2_images[i]))).astype(np.float) / 256.
            output_win_gt[gt == 0] = 0
            output_lose_gt = np.array(Image.open(os.path.join(dir1, dir1_images[i]))).astype(np.float) / 256.
            output_lose_gt[gt == 0] = 0
            row = np.concatenate((sparse_dilate, output_win, output_win_gt, output_lose, output_lose_gt, gt), axis=1)

            # gt[sparse_dilate == 0] = 0
            # sparse_dilate[gt == 0] = 0
            diff_im = output_win_gt - output_lose_gt

            if display_d:
                rgb_d_list.append(input)
                depth_d_list.append(row)
                diff_d_list.append(diff_im)
                display_d = False
            else:
                rgb_gd_list.append(input)
                depth_gd_list.append(row)
                diff_gd_list.append(diff_im)
                display_gd = False

    print(count_d)
    print(count_gd)

    avg_rmse_d = np.mean(rmse1_list)
    avg_rmse_gd = np.mean(rmse2_list)
    var_rmse_d = np.std(rmse1_list)
    var_rmse_gd = np.std(rmse2_list)
    print("avg_rmse_d: ", avg_rmse_d)
    print("avg_rmse_gd: ", avg_rmse_gd)
    print("var_rmse_d: ", var_rmse_d)
    print("var_rmse_gd: ", var_rmse_gd)
    print('d wins: ', rmsed_win, rmsegd_lose)
    print('gd wins: ', rmsegd_win, rmsed_lose)

    # plt.figure(10)
    # ax10 = plt.gca()
    # ax10.plot(rmse2_list)
    # ax10.plot([avg_rmse_gd]*len(rmse2_list), linestyle='--', color='r')

    depth_d_tot = np.vstack(depth_d_list)
    depth_d_norm = (depth_d_tot - np.min(depth_d_tot)) / (np.max(depth_d_tot) - np.min(depth_d_tot))
    depth_d_colored = (255 * cmap(depth_d_norm)[:, :, :3]).astype('uint8')
    rgb_d_tot = np.vstack(rgb_d_list)
    d_tot = np.concatenate((rgb_d_tot, depth_d_colored), axis=1)
    plt.figure(6)
    ax6 = plt.gca()
    im6 = ax6.imshow(depth_d_tot, cmap="jet")
    divider = make_axes_locatable(ax6)
    cax2 = divider.append_axes("bottom", size="7%", pad="2%")
    plt.colorbar(im6, cax=cax2, orientation='horizontal', label='depth [m]')
    ax6.imshow(d_tot)
    ax6.set_xticks([])
    ax6.set_yticks([])
    plt.show()

    depth_gd_tot = np.vstack(depth_gd_list)
    depth_gd_norm = (depth_gd_tot - np.min(depth_gd_tot)) / (np.max(depth_gd_tot) - np.min(depth_gd_tot))
    depth_gd_colored = (255 * cmap(depth_gd_norm)[:, :, :3]).astype('uint8')
    rgb_gd_tot = np.vstack(rgb_gd_list)
    gd_tot = np.concatenate((rgb_gd_tot, depth_gd_colored), axis=1)
    plt.figure(7)
    ax7 = plt.gca()
    im7 = ax7.imshow(depth_gd_tot, cmap="jet")
    divider = make_axes_locatable(ax7)
    cax2 = divider.append_axes("bottom", size="7%", pad="2%")
    plt.colorbar(im7, cax=cax2, orientation='horizontal', label='depth [m]')
    ax7.imshow(gd_tot)
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


def scaled_slam(gt_dir, slam_dir):
    gt_images = sorted(glob.glob(os.path.join(gt_dir, "*.png")))
    slam_images = sorted(glob.glob(os.path.join(slam_dir, "*.png")))
    X = np.array([])
    y = np.array([])
    skip = 5
    for i in range(0, int(len(gt_images)/5)):
        gt_im = np.array(Image.open(gt_images[i*skip])).astype(np.float) / 256.
        slam_im = np.array(Image.open(slam_images[i*skip])).astype(np.float) / 256.
        valid_mask = (slam_im > 0) & (gt_im > 0) & (slam_im <= 6)
        gt_masked = gt_im[valid_mask]
        slam_masked = slam_im[valid_mask]
        X = np.concatenate([X, slam_masked])
        y = np.concatenate([y, gt_masked])
    regl = LinearRegression().fit(X.reshape(-1, 1), y, )
    reg = RANSACRegressor().fit(X.reshape(-1, 1), y)
    print(regl.coef_, regl.intercept_)
    print(reg.estimator_.coef_, reg.estimator_.intercept_)
    linear_pred = regl.predict(X.reshape(-1, 1))
    ransac_pred = reg.predict(X.reshape(-1, 1))
    plt.scatter(X, y, s=1, c='tab:blue')
    plt.plot(X, linear_pred, c='cyan')
    plt.plot(X, ransac_pred, c='red')
    plt.legend(['linear regression', 'RANSAC'], loc='upper right')
    plt.show()


def present_images():
    cmap = plt.cm.jet
    gt_list = []
    rgb_un_list = []
    rgb_sea_list = []
    sparse_list = []
    # gt_dir = '../data/Nachsholim/rearranged/gt/test'
    gt_dir = '../data/SQUID/gt_all'
    gt_images = sorted(glob.glob(os.path.join(gt_dir, "*.png")))
    # sparse_dir = '../data/Nachsholim/rearranged/sparse/test'
    sparse_dir = '../data/SQUID/sparse_all'
    sparse_images = sorted(glob.glob(os.path.join(sparse_dir, "*.png")))
    # rgb_unenhanced_dir = '../data/Nachsholim/rearranged/rgb/test'
    rgb_unenhanced_dir = '../data/SQUID/rgb_all/'
    rgb_un_images = sorted(glob.glob(os.path.join(rgb_unenhanced_dir, "*.png")))
    rgb_seaErra_dir = '../data/Nachsholim/rearranged/rgb_seaErra/test'
    rgb_sea_images = sorted(glob.glob(os.path.join(rgb_seaErra_dir, "*.png")))

    skip = 15
    for i in range(4):
        i += 1
        gt_im = np.array(Image.open(gt_images[i * skip]))
        gt_im = gt_im.astype(np.float) / 256.
        gt_im = transform_geometric(gt_im)
        gt_im[gt_im > 30] = 0
        gt_list.append(gt_im)
        sparse_im = np.array(Image.open(sparse_images[i * skip]))
        sparse_im = sparse_im.astype(np.float) / 256.
        sparse_im = transform_geometric(sparse_im)
        sparse_im[sparse_im > 30] = 0
        sparse_im = cv2.dilate(sparse_im, np.ones((8, 8), np.uint8), iterations=1)
        sparse_list.append(sparse_im)
        rgb_un_im = np.array(Image.open(rgb_un_images[i * skip]))
        rgb_un_im = transform_geometric(rgb_un_im)
        rgb_un_list.append(rgb_un_im)
        rgb_sea_im = np.array(Image.open(rgb_sea_images[i * skip]))
        rgb_sea_im = transform_geometric(rgb_sea_im)
        rgb_sea_list.append(rgb_sea_im)

    rgb_un_tot = np.hstack(rgb_un_list)
    plt.figure(1)
    ax1 = plt.gca()
    ax1.imshow(rgb_un_tot)
    ax1.set_xticks([])
    ax1.set_yticks([])
    plt.show()
    rgb_sea_tot = np.hstack(rgb_sea_list)
    plt.figure(2)
    ax2 = plt.gca()
    ax2.imshow(rgb_sea_tot)
    ax2.set_xticks([])
    ax2.set_yticks([])
    plt.show()

    gt_tot = np.hstack(gt_list)
    sparse_tot = np.hstack(sparse_list)
    depth_tot = np.concatenate((sparse_tot, gt_tot), axis=0)
    depth_norm = (depth_tot - np.min(depth_tot)) / (np.max(depth_tot) - np.min(depth_tot))
    depth_colored = (255 * cmap(depth_norm)[:, :, :3]).astype('uint8')
    tot = np.concatenate((rgb_un_tot, depth_colored), axis=0)
    plt.figure(3)
    ax3 = plt.gca()
    im = ax3.imshow(gt_tot, cmap="jet")
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("bottom", size="7%", pad="2%")
    plt.colorbar(im, cax=cax, orientation='horizontal', label='depth [m]')
    ax3.imshow(tot)
    ax3.set_xticks([])
    ax3.set_yticks([])
    plt.show()


def count_slam_samples(slam_dir):
    slam_samples = []
    for image in sorted(glob.glob(os.path.join(slam_dir, "*.png"))):
        png_im = np.array(Image.open(image)) / 256.
        png_im[png_im > 6] = 0
        samples = png_im > 0
        slam_samples.append(len(png_im[samples]))
    avg_samples = np.mean(slam_samples)
    var_samples = np.var(slam_samples)
    std_samples = np.std(slam_samples)
    plt.figure(11)
    ax11 = plt.gca()
    ax11.plot(slam_samples)
    ax11.plot([avg_samples] * len(slam_samples), linestyle='--', color='r')
    ax11.set(ylabel='#samples')
    plt.show()


def slam_x2point(slam_dir):
    output_dir = os.path.join(slam_dir, "points")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    count = 0
    for image in sorted(glob.glob(os.path.join(slam_dir, "*.png"))):
        file_name = os.path.join(output_dir, image.split('/')[-1])
        if os.path.isfile(file_name):
            continue
        png_im = np.array(Image.open(image)).astype("uint16")
        samples_old = png_im[png_im > 0]
        for i in range(1, png_im.shape[0]-1):
            for j in range(1, png_im.shape[1]-1):
                if png_im[i, j] > 0:
                    if sum([(png_im[i+1, j-1] == png_im[i, j]), (png_im[i-1, j-1] == png_im[i, j]),
                            (png_im[i+1, j+1] == png_im[i, j]), (png_im[i-1, j+1] == png_im[i, j])]) >= 3:
                        if png_im[i+1, j-1] == png_im[i, j]:
                            png_im[i+1, j-1] = 0
                        if png_im[i-1, j-1] == png_im[i, j]:
                            png_im[i-1, j-1] = 0
                        if png_im[i+1, j+1] == png_im[i, j]:
                            png_im[i+1, j+1] = 0
                        if png_im[i-1, j+1] == png_im[i, j]:
                            png_im[i-1, j+1] = 0
        samples_new = png_im[png_im > 0]
        if len(samples_new) < 10:
            count += 1
            print(image)
        cv2.imwrite(file_name, png_im)
    print(count)


def color_depth_corr(rgb_dir, depth_dir, sparse_dir):
    # if not os.path.isdir(os.path.join(rgb_dir, "dropped_pearson")):
    #     os.mkdir(os.path.join(rgb_dir, "dropped_pearson"))
    # if not os.path.isdir(os.path.join(depth_dir, "dropped_pearson")):
    #     os.mkdir(os.path.join(depth_dir, "dropped_pearson"))
    # if not os.path.isdir(os.path.join(sparse_dir, "dropped_pearson")):
    #     os.mkdir(os.path.join(sparse_dir, "dropped_pearson"))
    rgb_images = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    depth_images = sorted(glob.glob(os.path.join(depth_dir, "*.png")))
    sparse_images = sorted(glob.glob(os.path.join(sparse_dir, "*.png")))
    pearson_tot = 0
    count_1 = 0
    count = 0
    for i in range(len(rgb_images)):
        # if i < 1816:
        #     continue
        rgb_im = np.array(Image.open(rgb_images[i])) #.astype(np.float)
        # rgb_im[:, :, 0] = np.divide(rgb_im[:, :, 0] , np.max(rgb_im[:, :, 0]))
        # rgb_im[:, :, 1] = np.divide(rgb_im[:, :, 1] , np.max(rgb_im[:, :, 1]))
        # rgb_im[:, :, 2] = np.divide(rgb_im[:, :, 2] , np.max(rgb_im[:, :, 2]))
        rgb_b_r = np.maximum(rgb_im[:, :, 2], rgb_im[:, :, 1]) - rgb_im[:, :, 0]
        # rgb_b_r = 0.53214829 + 0.51309827* np.maximum(rgb_im[:, :, 2], rgb_im[:, :, 1]) - 0.91066194 * rgb_im[:, :, 0]
        depth_im = np.array(Image.open(depth_images[i])).astype(np.float) / 256.
        valid_mask = (depth_im > 0.0) & (depth_im <= 6.0)
        rgb_masked = rgb_b_r[valid_mask]
        depth_masked = depth_im[valid_mask]
        if  depth_masked.size < 2 or  rgb_masked.size < 2:
            count_1 += 1
            continue
        pearson = scipy.stats.pearsonr(rgb_masked, depth_masked)[0]
        pearson_tot += pearson

        # X = rgb_masked
        # y = depth_masked
        # regl = LinearRegression().fit(X.reshape(-1, 1), y, )
        # reg = RANSACRegressor().fit(X.reshape(-1, 1), y)
        # coef = regl.coef_ if regl.coef_ > reg.estimator_.coef_ else reg.estimator_.coef_
        # intercept = regl.intercept_ if regl.coef_ > reg.estimator_.coef_ else reg.estimator_.intercept_
        # file_name = os.path.join(depth_dir, "filled_maps_new", depth_images[i].split('/')[-1].split('.')[0] + "_filled.png")
        # file_name2 = os.path.join(depth_dir, "filled_maps_all_new", depth_images[i].split('/')[-1].split('.')[0] + "_filled.png")

        # # if os.path.isfile(file_name):
        # #     continue
        # if (pearson < 0.2):
        #     shutil.move(rgb_images[i], os.path.join(rgb_dir, "dropped_pearson"))
        #     shutil.move(depth_images[i], os.path.join(depth_dir, "dropped_pearson"))
        #     shutil.move(sparse_images[i], os.path.join(sparse_dir, "dropped_pearson"))
        #     print(rgb_images[i] + " {:.3f}".format(pearson))
        #     count += 1
        #     fig, axs = plt.subplots(1, 4)
        #
        #     axs[0].imshow(rgb_im)
        #     axs[0].set_xticks([])
        #     axs[0].set_yticks([])
        #     axs[0].set_title('original RGB')
        #
        #     # rgb_b_r[depth_im > 0] = 0
        #     im_rgb = axs[1].imshow(rgb_b_r, cmap="jet")
        #     divider = make_axes_locatable(axs[1])
        #     cax = divider.append_axes("bottom", size="7%", pad="2%")
        #     plt.colorbar(im_rgb, cax=cax, orientation='horizontal', label='value')
        #     axs[1].set_xticks([])
        #     axs[1].set_yticks([])
        #     axs[1].set_title('Max(B,G)-R')
        #
        #     depth_im[depth_im > 15] = 0
        #     im_depth_new = axs[2].imshow(depth_im, cmap="jet")
        #     divider3 = make_axes_locatable(axs[2])
        #     cax3 = divider3.append_axes("bottom", size="7%", pad="2%")
        #     plt.colorbar(im_depth_new, cax=cax3, orientation='horizontal', label='depth [m]')
        #     axs[2].set_xticks([])
        #     axs[2].set_yticks([])
        #     axs[2].set_title('Ground truth')
        # else:
        #     if coef > 0.055:
        #         if os.path.isfile(file_name):
        #             continue
        #         depth_im_fill = np.array(Image.open(depth_images[i])).astype(np.float) / 256.
        #         depth_im_new = fill_gt(depth_im, depth_im_fill, rgb_b_r, coef, intercept)
        #         if scipy.stats.pearsonr(rgb_b_r[depth_im_new > 0], depth_im_new[depth_im_new > 0])[0] < 0.9:
        #             continue
        #             # print(scipy.stats.pearsonr(rgb_b_r[depth_im_new > 0], depth_im_new[depth_im_new > 0])[0])
        #             shutil.copy(depth_images[i], os.path.join(depth_dir, "filled_maps_all_new"))
        #             os.rename(os.path.join(depth_dir, "filled_maps_all_new", depth_images[i].split('/')[-1]),
        #                       file_name2)
        #         else:
        #             # print(scipy.stats.pearsonr(rgb_b_r[depth_im_new > 0], depth_im_new[depth_im_new > 0])[0])
        #             cv2.imwrite(file_name, (depth_im_new * 256).astype('uint16'))
        #             cv2.imwrite(file_name2, (depth_im_new * 256).astype('uint16'))
        #             count += 1
        #
        #         # shutil.copy(rgb_images[i], os.path.join(rgb_dir, "filled_maps"))
        #         # shutil.copy(sparse_images[i], os.path.join(sparse_dir, "filled_maps"))
        #     else:
        #         shutil.copy(depth_images[i], os.path.join(depth_dir, "filled_maps_all_new"))
        #         os.rename(os.path.join(depth_dir, "filled_maps_all_new", depth_images[i].split('/')[-1]), file_name2)
        #         continue
        #
        #     # fig, axs = plt.subplots(1, 4)
        #     #
        #     # axs[0].imshow(rgb_im)
        #     # axs[0].set_xticks([])
        #     # axs[0].set_yticks([])
        #     # axs[0].set_title('original RGB')
        #     #
        #     # # rgb_b_r[depth_im > 0] = 0
        #     # im_rgb = axs[1].imshow(rgb_b_r, cmap="jet")
        #     # divider = make_axes_locatable(axs[1])
        #     # cax = divider.append_axes("bottom", size="7%", pad="2%")
        #     # plt.colorbar(im_rgb, cax=cax, orientation='horizontal', label='value')
        #     # axs[1].set_xticks([])
        #     # axs[1].set_yticks([])
        #     # axs[1].set_title('Max(B,G)-R')
        #     #
        #     # # depth_im_con = np.concatenate((depth_im, depth_im_new), axis=1)
        #     # # im_depth = axs[2].imshow(depth_im_con, cmap="jet")
        #     # # divider1 = make_axes_locatable(axs[2])
        #     # # cax1 = divider1.append_axes("bottom", size="7%", pad="2%")
        #     # # plt.colorbar(im_depth, cax=cax1, orientation='horizontal', label='depth [m]')
        #     # # axs[2].set_xticks([])
        #     # # axs[2].set_yticks([])
        #     # # axs[2].set_title('Ground truth')
        #     #
        #     # im_depth_new = axs[2].imshow(depth_im, cmap="jet")
        #     # divider3 = make_axes_locatable(axs[2])
        #     # cax3 = divider3.append_axes("bottom", size="7%", pad="2%")
        #     # plt.colorbar(im_depth_new, cax=cax3, orientation='horizontal', label='depth [m]')
        #     # axs[2].set_xticks([])
        #     # axs[2].set_yticks([])
        #     # axs[2].set_title('Ground truth')
        #     #
        #     # linear_pred = regl.predict(X.reshape(-1, 1))
        #     # ransac_pred = reg.predict(X.reshape(-1, 1))
        #     # axs[3].scatter(X, y, s=1, c='tab:blue')
        #     # axs[3].plot(X, linear_pred, c='cyan')
        #     # axs[3].plot(X, ransac_pred, c='red')
        #     # axs[3].legend(['linear regression', 'RANSAC'], loc='upper right')
        #     #
        #     # fig.suptitle(r'$\rho = ' + "{:.3f}$".format(pearson))
        #     # plt.show()

    # print(count)
    print(pearson_tot/(len(rgb_images)-count_1))


def present_depth_map(depth_im):
    plt.figure()
    ax = plt.gca()
    im = ax.imshow(depth_im, cmap="gray")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="7%", pad="2%")
    plt.colorbar(im, cax=cax, orientation='horizontal', label='depth [m]')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


def erase_circles(depth_dir):
    output_dir = os.path.join(depth_dir, "erased")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    depth_images = sorted(glob.glob(os.path.join(depth_dir, "*.png")))
    color = (0, 0, 0)
    thickness = -1
    radius = 120
    # center_coordinates = [(70, 900), (300, 450), (125, 650), (300, 300), (450, 200), (400, 650), (100, 440)]
    for i in range(len(depth_images)):
        depth_im = np.array(Image.open(depth_images[i])).astype("uint16")
        if len(depth_im[depth_im == 0]) / depth_im.size > 0.6:
            print(depth_images[i])
        else:
            x = int(depth_im.shape[0] / 2)
            y = int(depth_im.shape[1] / 2)
            center_coordinates = [(random.randint(radius, x), random.randint(radius, y)),
                                  (random.randint(radius, x), random.randint(y, y*2 - radius)),
                                  (random.randint(x, x * 2 - radius), random.randint(radius, y)),
                                  (random.randint(x, x * 2 - radius), random.randint(y, y*2 - radius))
                                  ]
            c = 0
            for center in center_coordinates:
                if depth_im[center] > 0:  # & (depth_im[center[0] + radius, center[1] + radius] > 0): #  & (c < 4):
                    depth_im = cv2.circle(depth_im, (center[1], center[0]), random.randint(50, radius), color, thickness)
                    c += 1
            # present_depth_map(depth_im / 256.)

        file_name = os.path.join(output_dir, depth_images[i].split('/')[-1].split('.')[0] + "_erased.png")
        cv2.imwrite(file_name, depth_im)


def erase_pixels(depth_dir):
    output_dir = os.path.join(depth_dir, "erased_pixels")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    depth_images = sorted(glob.glob(os.path.join(depth_dir, "*.png")))
    for i in range(len(depth_images)):
        depth_im = np.array(Image.open(depth_images[i])).astype(np.float)
        height, width = depth_im.shape
        if len(depth_im[depth_im == 0]) / depth_im.size > 0.6:
            print(depth_images[i])
        else:
            y_idx, x_idx = np.where(depth_im[50:height-50, 50:width-50] > 0)
            blob_num = random.randint(5, 10)
            for b in range(1, blob_num):
                blob_size = random.randint(10000, 40000)
                blob = []
                rand_idx = random.randint(0, x_idx.size)
                x = x_idx[rand_idx]
                y = y_idx[rand_idx]
                blob.append((y, x))
                depth_im[y, x] = 0
                while len(blob) < blob_size:
                    x = x + random.randint(-1, 1)
                    y = y + random.randint(-1, 1)
                    blob.append((y, x))
                    # depth_im[y, x] = 0
                ConvexHull(blob)
        present_depth_map(depth_im / 256.)
        present_depth_map((np.array(Image.open(depth_images[i])).astype(np.float) - depth_im) / 256.)

        # file_name = os.path.join(output_dir, depth_images[i].split('/')[-1].split('.')[0] + "_erased.png")
        # cv2.imwrite(file_name, depth_im)


def fill_gt(depth_im, depth_im_fill, rgb_b_r, coef, intercept):
    # present_depth_map(depth_im)
    iy, ix = np.where(depth_im > 0)
    nx, ny = depth_im.shape[1], depth_im.shape[0]
    X, Y = np.meshgrid(np.arange(0, nx, 1), np.arange(0, ny, 1))
    samples = depth_im[np.array(iy), np.array(ix)]
    interpolated_im = griddata((np.array(iy), np.array(ix)), samples, (Y, X), method='linear', fill_value=0.0)

    depth_im_temp = np.zeros(depth_im.shape).astype(np.float)
    for r in range(0, depth_im.shape[0] - 1):
        for c in range(0, depth_im.shape[1] - 1):
            depth_im_temp[r, c] = rgb_b_r[r, c] * coef + intercept
    depth_im_blured = cv2.GaussianBlur(depth_im_temp, (31, 31), 0)

    for r in range(0, depth_im_fill.shape[0] - 1):
        for c in range(0, depth_im_fill.shape[1] - 1):
            if depth_im[r, c] == 0:
                depth_im_fill[r, c] = depth_im_blured[r, c]

    depth_im_fill[depth_im_fill > 6] = 0
    depth_im_fill[depth_im_fill < 0] = 0

    for r in range(0, depth_im.shape[0] - 1):
        for c in range(0, depth_im.shape[1] - 1):
            if (depth_im[r, c] == 0) & (depth_im_fill[r, c] > 0) & (interpolated_im[r, c] > 0) & \
                    (interpolated_im[r, c] < 3) & (depth_im_blured[r, c] - interpolated_im[r, c] > 0.5):
                depth_im_fill[r, c] = interpolated_im[r, c]

    # present_depth_map(depth_im)
    return depth_im_fill


def compare_output(results_dirs, rgb_dir, sparse_dir, gt_dir, max_depth, save_images, with_var):
    cmap = plt.cm.jet
    output_images = {}
    var_images = {}
    for i in range(0, len(results_dirs)):
        output_images[i] = sorted(os.listdir(os.path.join(results_dirs[i], 'val_output')))
        if with_var:
            var_images[i] = sorted(os.listdir(os.path.join(results_dirs[i], 'val_output_var')))
    input_images = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    sparse_images = sorted(glob.glob(os.path.join(sparse_dir, "*.png")))
    gt_images = sorted(glob.glob(os.path.join(gt_dir, "*.png")))
    rgb_list = []
    depth_list = []
    diff_list = []
    var_list = []
    skip = 50
    start = 0
    output = {}
    var = {}
    output_gt = {}
    output_var = {}
    output_diff_3 = {}
    output_diff_6 = {}
    output_diff = {}
    var_percent = 0
    for i in range(4):
        for o in range(len(output_images)):
            output[o] = transform_geometric(np.array(Image.open(os.path.join(results_dirs[o], 'val_output',output_images[o][i*skip + start]))).astype(np.float) / 256.)
            output_gt[o] = transform_geometric(np.array(Image.open(os.path.join(results_dirs[o], 'val_output',output_images[o][i * skip + start]))).astype(
                np.float) / 256.)
            output_var[o] = transform_geometric(np.array(Image.open(os.path.join(results_dirs[o], 'val_output', output_images[o][i * skip + start]))).astype(
                np.float) / 256.)
            output_diff_3[o] = transform_geometric(np.array(Image.open(os.path.join(results_dirs[o], 'val_output',output_images[o][i * skip + start]))).astype(
                np.float) / 256.)
            output_diff_6[o] = transform_geometric(np.array(Image.open(os.path.join(results_dirs[o], 'val_output',output_images[o][i * skip + start]))).astype(
                np.float) / 256.)
            output_diff_6[o] = transform_geometric(np.array(Image.open(os.path.join(results_dirs[o], 'val_output', output_images[o][i * skip + start]))).astype(
                np.float) / 256.)
            if with_var:
                var_im = transform_geometric(np.array(Image.open(os.path.join(results_dirs[o], 'val_output_var' ,var_images[o][i * skip + start]))).astype(np.float))
                var[o] = var_im  # np.sqrt(np.exp(var_im))
        input = transform_geometric(np.array(Image.open(input_images[i*skip + start])))
        sparse = transform_geometric(np.array(Image.open(sparse_images[i*skip + start]))).astype(np.float) / 256.
        sparse[sparse > max_depth] = 0
        sparse_dilate = cv2.dilate(sparse, np.ones((4, 4), np.uint8), iterations=1)
        gt = transform_geometric(np.array(Image.open(gt_images[i*skip + start]))).astype(np.float) / 256.
        gt[gt > max_depth] = 0
        output_gt[0][gt == 0] = 0
        diff_im = output[0] - gt
        diff_im[gt == 0] = 0
        # output_diff_6[0][gt == 0] = 0
        output_diff_3[0][diff_im*100 > 10] = 0
        output_diff_3[0][gt > 3] = 0
        output_diff_6[0][diff_im*100 > 30] = 0
        output_diff_6[0][gt <= 3] = 0
        output_diff[0] = output_diff_3[0] + output_diff_6[0]
        output_diff_3[0][gt == 0] = 0

        if with_var:
            var_im = np.sqrt(np.exp(var[0]))
            # output_var[0][gt == 0] = output[0][gt == 0]
            output_var[0][var_im > 0.5] = 0
            var_percent+= 100* len(output_var[0][output_var[0]==0])/ (416 * 736)
        output_diff[0][diff_im > 0.1 * gt] = 0
        output_diff[0][gt == 0] = output[0][gt == 0]
        # output_diff[0][gt == 0] = 0
        # print(np.min(gt[diff_im > 30]))
        row = np.concatenate((sparse_dilate, output[0]), axis=1)  # , output_gt[0], output_diff[0]), axis=1
        row_diff = diff_im
        if with_var:
            row_var = var[0]

        for o in range(1, len(output)):
            output_gt[o][gt == 0] = 0
            diff_im = output[o] - gt
            diff_im[gt == 0] = 0
            output_diff_3[o][diff_im*100 > 10] = 0
            output_diff_3[o][gt > 3] = 0
            output_diff_6[o][diff_im*100 > 30] = 0
            output_diff_6[o][gt <= 3] = 0
            output_diff[o] = output_diff_3[o] + output_diff_6[o]
            output_diff_3[o][gt == 0] = 0
            if with_var:
                var_im = np.sqrt(np.exp(var[o]))
                output_var[o][var_im > 0.1*gt] = 0
                output_var[o][gt == 0] = output[o][gt ==0]
            output_diff[o][diff_im > 0.1 * gt] = 0
            output_diff[o][gt == 0] = output[o][gt == 0]
            # output_diff[o][gt == 0] = 0
            # print(np.min(gt[diff_im > 30]))
            row = np.concatenate((row, output[o]), axis=1)  # , output_gt[o], output_diff[o]), axis=1)
            diff_im = diff_im
            row_diff = np.concatenate((row_diff, diff_im), axis=0)
            if with_var:
                row_var = np.concatenate((row_var, var[o]), axis=1)
        row = np.concatenate((row, gt), axis=1)

        rgb_list.append(input)
        depth_list.append(row)
        diff_list.append(row_diff)
        if with_var:
            var_list.append(row_var)

    var_percent_avg = var_percent / len(gt_images)
    depth_tot = np.vstack(depth_list)
    # depth_tot[depth_tot > 7] = 0
    depth_norm = (depth_tot - np.min(depth_tot)) / (np.max(depth_tot) - np.min(depth_tot))
    depth_colored = (255 * cmap(depth_norm)[:, :, :3]).astype('uint8')
    rgb_tot = np.vstack(rgb_list)
    tot = np.concatenate((rgb_tot, depth_colored), axis=1)
    fig6 = plt.figure(6)
    plt.title('RGB | sparse | output | output (valid GT)  |  GT')
    ax6 = plt.gca()
    im6 = ax6.imshow(depth_tot, cmap="jet")
    divider = make_axes_locatable(ax6)
    cax2 = divider.append_axes("bottom", size="7%", pad="2%")
    plt.colorbar(im6, cax=cax2, orientation='horizontal', label='depth [m]')
    ax6.imshow(tot)
    ax6.set_xticks([])
    ax6.set_yticks([])
    plt.show()

    if with_var:
        var_tot = np.vstack(var_list)
        var_act = np.sqrt(np.exp(var_tot))
        # # diff_tot[diff_tot < 30] = 0
        # # diff_tot[diff_tot > 50] = 0
        fig8 = plt.figure(8)
        # plt.title('output original  |  output original + corr')
        ax8 = plt.gca()
        im8 = ax8.imshow(var_act, cmap="gray")
        divider = make_axes_locatable(ax8)
        cax2 = divider.append_axes("right", size="7%", pad="2%")
        plt.colorbar(im8, cax=cax2, orientation='vertical', label='std[cm]')
        ax8.imshow(var_tot, cmap="gray")
        ax8.set_xticks([])
        ax8.set_yticks([])
        plt.show()

    diff_tot = np.hstack(diff_list)
    # diff_tot[diff_tot < 30] = 0
    # diff_tot[diff_tot > 50] = 0
    fig9 = plt.figure(9)
    plt.title('output original  |  output original + corr')
    ax9 = plt.gca()
    im9 = ax9.imshow(diff_tot, cmap="jet")
    divider = make_axes_locatable(ax9)
    cax3 = divider.append_axes("right", size="7%", pad="2%")
    plt.colorbar(im9, cax=cax3, orientation='vertical', label='depth [m]')
    ax9.set_xticks([])
    ax9.set_yticks([])
    plt.show()

    if save_images:
        fig6.set_size_inches(10, 10)
        fig6.savefig(results_dirs[0] + '/../' + 'results.png')
        fig8.set_size_inches(5, 10)
        fig8.savefig(results_dirs[0] + '/../' + 'diff.png')

def print_graph(pred_dir, gt_dir):
    gt_im_valid = np.linspace(0.5, 3.0, num=500)
    gt_im_valid_6 = np.linspace(3.0, 6.0, num=500)
    gt_im_valid_t = np.linspace(3.0, 6.0, num=500)
    pred = np.linspace(0.5, 3.0, num=500) +  np.random.uniform(low=-0.5, high=0.5, size=(500,))
    pred_6 = np.linspace(3.0 , 6.0, num=500) + np.random.uniform(low=-0.5, high=0.5, size=(500,))
    pred_t = np.linspace(3.0, 6.0, num=500) + np.random.uniform(low=-0.5, high=0.5, size=(500,))
    D_mel = 0.1
    D_mel_6 = 0.2
    D_mel_t = 0.4
    MAD = sorted(pred - gt_im_valid)
    MAD_6 = sorted(pred_6 - gt_im_valid_6)
    MAD_t = sorted(pred_t - gt_im_valid_t)
    tukey = [1 if np.abs(mad) > D_mel else 1 - np.power(1-(np.abs(mad)/D_mel)**2, 3) for mad in MAD]
    tukey6 = [1 if np.abs(mad) > D_mel_6 else 1 - np.power(1 - (np.abs(mad) / D_mel_6) ** 2, 3) for mad in MAD_6]
    tukeyt = [1 if np.abs(mad) > D_mel_t else 1 - np.power(1 - (np.abs(mad) / D_mel_t) ** 2, 3) for mad in MAD_t]

    plt.figure(1)
    plt.plot(MAD, tukey)
    plt.plot(MAD_6, tukey6)
    plt.plot(MAD_t, tukeyt)
    plt.xlabel('Depth error [m]')
    plt.ylabel('Tukey loss')
    plt.show()
    plt.legend([r'$\tau = 0.1m$', r'$\tau = 0.2m$', r'$\tau = 0.4m$'], loc='lower right')

    t=3
    k=6
    x = np.linspace(0.5, 6.0, num=500)
    my_sigmoid = 0.1 + 0.2 *(1 / (1 + np.exp(k*(t-x))))

    plt.figure(2)
    plt.plot(x, my_sigmoid)
    plt.xlabel('GT [m]')
    plt.ylabel(r'$\tau$ [m]')
    plt.show()

    pred = np.linspace(0.5 , 6.0, num=500) + np.random.uniform(low=-0.5, high=0.5, size=(500,))
    # MAD = sorted(pred - x)
    MAD = np.linspace(-0.5 , 0.5, num=500)
    # my_tukey = np.zeros(500)
    # for i in range(500):
    #     if np.abs(MAD[i]) <= my_sigmoid[i]:
    #         my_tukey[i] = 1 - np.power(1 - (np.abs(MAD[i]) / my_sigmoid[i]) ** 2, 3)
    #     else:
    #         my_tukey[i] = 1
    tau_05 = 0.1 + 0.2 *(1 / (1 + np.exp(k*(t-1))))
    tukey_05 = [1 if np.abs(mad) > tau_05 else 1 - np.power(1 - (np.abs(mad) / tau_05) ** 2, 3) for mad in MAD]
    tau_1 = 0.1 + 0.2 * (1 / (1 + np.exp(k * (t - 2.5))))
    tukey_1 = [1 if np.abs(mad) > tau_1 else 1 - np.power(1 - (np.abs(mad) / tau_1) ** 2, 3) for mad in MAD]
    tau_2 = 0.1 + 0.2 * (1 / (1 + np.exp(k * (t - 3))))
    tukey_2 = [1 if np.abs(mad) > tau_2 else 1 - np.power(1 - (np.abs(mad) / tau_2) ** 2, 3) for mad in MAD]
    tau_4 = 0.1 + 0.2 * (1 / (1 + np.exp(k * (t -3.5))))
    tukey_4 = [1 if np.abs(mad) > tau_4 else 1 - np.power(1 - (np.abs(mad) / tau_4) ** 2, 3) for mad in MAD]
    tau_5 = 0.1 + 0.2 * (1 / (1 + np.exp(k * (t - 5))))
    tukey_5 = [1 if np.abs(mad) > tau_5 else 1 - np.power(1 - (np.abs(mad) / tau_5) ** 2, 3) for mad in MAD]
    plt.figure(3)
    ax = plt.axes(projection='3d')
    yline = x
    xline = MAD
    # zline = my_tukey
    ax.plot3D(xline, np.linspace(1, 1, num=500), tukey_05)
    ax.plot3D(xline, np.linspace(2.5, 2.5, num=500), tukey_1)
    ax.plot3D(xline, np.linspace(3, 3, num=500), tukey_2)
    ax.plot3D(xline, np.linspace(3.5, 3.5, num=500), tukey_4)
    ax.plot3D(xline, np.linspace(5, 5, num=500), tukey_5)
    ax.set_ylabel('GT [m]')
    ax.set_xlabel('Depth error [m]')
    ax.set_zlabel('Tukey Loss')
    ax.legend(['gt=1m', 'gt=2.5m','gt=3m', 'gt=3.5m','gt=5m'], loc='lower left')
    my_tukey = [1 if np.abs(mad) > D_mel else 1 - np.power(1 - (np.abs(mad) / D_mel) ** 2, 3) for mad in MAD]


    pred = np.concatenate((np.linspace(0.5 , 3.0, num=50) + 0.1, np.linspace(3.0 , 6.0, num=50)+ 0.3), axis = 0)
    # pred = np.concatenate((np.linspace(0.5 , 3.0, num=50) + np.random.uniform(low=0.1, high=1.0, size=(50,)),
    #                        np.linspace(3.0 , 6.0, num=50)+ np.random.uniform(low=0.3, high=3.0, size=(50,))), axis = 0)

    rel = np.abs(pred - gt_im_valid) / gt_im_valid
    squared_rel = ((pred - gt_im_valid) / gt_im_valid) ** 2
    rel_squared = np.abs(pred - gt_im_valid) / gt_im_valid ** 2
    rel_exp = np.abs(pred - gt_im_valid) / np.exp(gt_im_valid)
    plt.figure(1)
    plt.plot(gt_im_valid, rel)
    plt.plot(gt_im_valid, squared_rel)
    plt.plot(gt_im_valid, rel_squared)
    plt.plot(gt_im_valid, rel_exp)
    plt.xlabel('gt')
    plt.ylabel('loss')
    plt.title('Error is 1[m]')
    plt.show()
    plt.legend(['abs_rel', 'squared_rel', 'rel_squared', 'rel_exp'])

    pred = np.linspace(0.5, 6.0, num=100) + 2
    rel = np.abs(pred - gt_im_valid) / gt_im_valid
    squared_rel = ((pred - gt_im_valid) / gt_im_valid) ** 2
    rel_squared = np.abs(pred - gt_im_valid) / gt_im_valid ** 2
    rel_exp = np.abs(pred - gt_im_valid) / np.exp(gt_im_valid)
    plt.figure(2)
    plt.plot(gt_im_valid, rel)
    plt.plot(gt_im_valid, squared_rel)
    plt.plot(gt_im_valid, rel_squared)
    plt.plot(gt_im_valid, rel_exp)
    plt.xlabel('gt')
    plt.ylabel('loss')
    plt.title('Error is 2[m]')
    plt.show()
    plt.legend(['abs_rel', 'squared_rel', 'rel_squared', 'rel_exp'])

    pred = np.linspace(0.5, 6.0, num=100) + 0.1
    rel = np.abs(pred - gt_im_valid) / gt_im_valid
    squared_rel = ((pred - gt_im_valid) / gt_im_valid) ** 2
    rel_squared = np.abs(pred - gt_im_valid) / gt_im_valid ** 2
    rel_exp = np.abs(pred - gt_im_valid) / np.exp(gt_im_valid)
    plt.figure(3)
    plt.plot(gt_im_valid, rel)
    plt.plot(gt_im_valid, squared_rel)
    plt.plot(gt_im_valid, rel_squared)
    plt.plot(gt_im_valid, rel_exp)
    plt.xlabel('gt')
    plt.ylabel('loss')
    plt.title('Error is 0.1[m]')
    plt.show()
    plt.legend(['abs_rel', 'squared_rel', 'rel_squared', 'rel_exp'])


    plt.plot(gt_im_valid, rel_squared)


    pred_images = sorted(glob.glob(os.path.join(pred_dir, "*.png")))
    gt_images = sorted(glob.glob(os.path.join(gt_dir, "*.png")))
    for i in range(len(gt_images)):
        gt_im = transform_geometric(np.array(Image.open(gt_images[i])).astype(np.float) / 256.)
        gt_im_3 = transform_geometric(np.array(Image.open(gt_images[i])).astype(np.float) / 256.)
        gt_im[gt_im > 6] = 0
        gt_im_3[gt_im_3 > 3] = 0
        gt_im_valid = gt_im[gt_im > 0]
        gt_im_valid_3 = gt_im_3[gt_im_3 > 0]
        pred_im = transform_geometric(np.array(Image.open(pred_images[i])).astype(np.float) / 256.)
        pred_im_valid = pred_im[gt_im > 0]
        pred_im_valid_3 = pred_im[gt_im_3 > 0]
        rel_exp = np.abs(pred_im_valid - gt_im_valid) / np.exp(gt_im_valid)
        rel = np.abs(pred_im_valid - gt_im_valid) / gt_im_valid
        rel_squared = ((pred_im_valid - gt_im_valid) / gt_im_valid) ** 2
        fun1 = 1*((pred_im_valid_3 - gt_im_valid_3) > 0.1)
        plt.figure(1)
        plt.plot(pred_im_valid, rel_exp)
        plt.figure(2)
        plt.plot(pred_im_valid, rel)
        plt.figure(3)
        plt.plot(pred_im_valid, rel_squared)
        plt.xlabel('pred')
        plt.ylabel('fun')
        plt.show()


def remove_cans(dir, mode):
    cans_dir = os.path.join(dir , 'cans')
    if not os.path.isdir(cans_dir):
        os.mkdir(cans_dir)
    rgb_dir = os.path.join('../data/Nachsholim/rearranged/rgb', mode, 'cans')
    dir_images = sorted(glob.glob(os.path.join(dir, "*.png")))
    rgb_images = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    i = 0
    for j in range(len(dir_images)):
        image = dir_images[j]
        im_num = image.split('/')[-1].split('.')[0].split('_')[2]
        rgb_num = rgb_images[i].split('/')[-1].split('.')[0].split('_')[2]
        if rgb_num == im_num:
            shutil.move(image, cans_dir)
            i += 1
               #  break

def remove_less_than_3(mode):
    rgb_dir = os.path.join('../data/Nachsholim/rearranged/rgb', mode)
    out_dir = os.path.join(rgb_dir , 'less_3')
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    gt_dir = os.path.join('../data/Nachsholim/rearranged/gt', mode)
    out_dir = os.path.join(gt_dir , 'less_3')
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    sparse_dir = os.path.join('../data/Nachsholim/rearranged/sparse', mode)
    out_dir = os.path.join(sparse_dir , 'less_3')
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    gt_images = sorted(glob.glob(os.path.join(gt_dir, "*.png")))
    rgb_images = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    sparse_images = sorted(glob.glob(os.path.join(sparse_dir, "*.png")))
    count = 0
    for j in range(len(gt_images)):
        image = transform_geometric(np.array(Image.open(gt_images[j])).astype(float) / 256.)
        if np.min(image[image > 0]) > 3:
            count += 1
            shutil.move(gt_images[j], os.path.join(gt_dir , 'less_3'))
            shutil.move(rgb_images[j], os.path.join(rgb_dir, 'less_3'))
            shutil.move(sparse_images[j], os.path.join(sparse_dir, 'less_3'))
    print(count)


def main():

    images_dir = '../data/Nachsholim/rearranged/rgb_seaErra/'
    mode = 'train'
    # remove_cans(os.path.join(images_dir, mode), mode)
    #
    # rgb_dir = '../data/Nachsholim/rearranged/rgb/train/cans'
    # rgb_images = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    # for image in rgb_images:
    #     img =cv2.imread(image)
    #     plt.figure(5)
    #     ax = plt.gca()
    #     im = ax.imshow(img)
    #     plt.show()

    # remove_less_than_3('train')
    # remove_less_than_3('val')
    # remove_less_than_3('test')


    pred = '../pretrained_models/supervised/nachsholim_manual_slam/rearranged/test/' \
           'mode=dense.data=nachsholim.input=gd.resnet18.epochs20.criterion=REL.lr=0.0001.bs=2.wd=0.pretrained=True.jitter=0.1.time=2021-04-07@11-34_rel_exp/val_output'
    target = '../data/Nachsholim/rearranged/gt/test'
    print_graph(pred, target)

    images_dir = '../data/SQUID/rgb/rgt/png'
    # tif_to_png(images_dir)
    factor = 0.3
    # resize_images(images_dir, factor)

    depth_maps_dir = '../data/SouthCarolinaCemetery/sparse/'
    # save_depth_as_uint16(depth_maps_dir)

    main_dir = '../data/Nachsholim/rearranged/gt/'
    # calc_depth_hist(main_dir)

    # depth_maps_dir = '../data/Nachsholim/depth_lft/uint16/test'
    # value = 15.0
    # truncate_depth_maps(depth_maps_dir, value)

    depth_maps_png = '../data/SQUID/gt_all'
    n_samples = 500
    interp = False
    # gt_to_sparse(depth_maps_png, n_samples, interp)

    gt_dir = '../data/Nachsholim/rearranged/gt/train/'
    slam_dir = '../data/Nachsholim/rearranged/slam/train/points'
    # gt_to_sparse_based_slam(gt_dir, slam_dir)
    # scaled_slam(gt_dir, slam_dir)

    sparse_dir = '../data/Nachsholim/rearranged/sparse/test/'
    # sparse_to_interp(sparse_dir)

    gt_dir = '../data/Nachsholim/rearranged/gt/test'
    interp_dir = '../data/Nachsholim/rearranged/sparse/test/interp_6m'
    output_dir = '../pretrained_models/supervised/nachsholim_manual_slam/rearranged/test/' \
                 'mode=dense.data=nachsholim.input=gd.resnet18.epochs20.criterion=Tukey.lr=0.0001.bs=2.wd=0.pretrained=True.jitter=0.1.rank_metric=diff_thresh.time=2021-08-04@10-21_with_var_test/val_output'
    # output_dir = '../pretrained_models/supervised/nachsholim_manual_slam/rearranged/test/' \
    #              'mode=dense.data=nachsholim.input=d.resnet18.epochs20.criterion=Tukey.lr=0.0001.bs=2.wd=0.pretrained=True.jitter=0.1.rank_metric=diff_thresh.train_var=True.time=2021-09-22@17-27/val_output'
    max_depth = 6
    # calc_errors(gt_dir, output_dir, output_dir, max_depth)

    # rename_files()

    # colorize_depth()

    output_dense = '../pretrained_models/supervised/nachsholim_manual_slam/test/mode=dense.data=nachsholim.input=rgbd.resnet18.epochs30.criterion=l2.lr=0.0001.bs=2.wd=0.pretrained=True.jitter=0.1.time=2021-01-28@15-22_regular_centercrop/val_output'
    output_dense_corr = '../pretrained_models/supervised/nachsholim_manual_slam/test/mode=dense+corr.w1=0.5.w2=0.1.data=nachsholim.input=rgbd.resnet18.epochs30.criterion=l2.lr=0.0001.bs=2.wd=0.pretrained=True.jitter=0.1.time=2021-01-31@16-19_original_centercrop/val_output'
    # compare_rgb_images(output_dense, output_dense_corr)

    # present_images()

    slam_dir = '../data/Nachsholim/rearranged/sparse/train'
    # count_slam_samples(slam_dir)

    slam_dir = '../data/Nachsholim/rearranged/slam/test/'
    # slam_x2point(slam_dir)

    rgb_dir = '../data/Nachsholim/rearranged/rgb/train/'
    gt_dir = '../data/Nachsholim/rearranged/gt/train'
    sparse_dir = '../data/Nachsholim/rearranged/sparse/train/'
    # color_depth_corr(rgb_dir, gt_dir, sparse_dir)

    # count = 0
    # for image_name in sorted(glob.glob(os.path.join(gt_dir, "*.png"))):
    #     count += 1
    #     if count < 500:
    #         continue
    # # image_name ='../data/Nachsholim/depth_lft/uint16_slam/train/truncate/filled_maps/dist_l_000457_filled.png'
    #     depth_im = np.array(Image.open(image_name)).astype(np.float) / 256.
    #     present_depth_map(depth_im)

    rgb_dir = '../data/Nachsholim/rgb/test'
    sparse_dir = '../data/Nachsholim/sparse/test/'
    gt_dir = '../data/Nachsholim/gt/test'
    # out_interp = '../data/Nachsholim/rearranged/sparse/test/interp_6m'
    # out1 = '../pretrained_models/supervised/nachsholim_manual_slam/rearranged/test/' \
    #        'mode=dense.data=nachsholim.input=gd.resnet18.epochs20.criterion=Tukey.lr=0.0001.bs=2.wd=0.pretrained=True.jitter=0.1.rank_metric=diff_thresh.time=2021-07-19@16-37_test_best/val_output'
    out1 = '../pretrained_models/supervised/nachsholim_manual_slam/rearranged/test/' \
           'mode=dense.data=nachsholim.input=gd.resnet18.epochs20.criterion=Tukey.lr=0.0001.bs=2.wd=0.pretrained=True.jitter=0.1.rank_metric=diff_thresh.time=2021-08-04@10-21_with_var_test/'
    out2 = '../pretrained_models/supervised/nachsholim_manual_slam/rearranged/test/' \
           'mode=dense+corr.w1=0.5.w2=0.5.data=nachsholim.input=rgbd.resnet18.epochs20.criterion=Tukey.lr=0.0001.bs=2.wd=0.pretrained=True.jitter=0.1.rank_metric=diff_thresh.train_var=True.time=2021-11-16@17-23/'
    out3 = '../pretrained_models/self-supervised/nachsholim_manual_slam/rearranged/test/' \
           'mode=sparse+corr.w1=0.5.w2=0.5.data=nachsholim.input=rgbd.resnet18.epochs20.criterion=l2.lr=0.0001.bs=2.wd=0.pretrained=True.jitter=0.1.rank_metric=rmse.train_var=True.time=2021-09-26@09-16/'
    max_depth = 6
    save_images = False
    with_var = True
    compare_output([out1, out2], rgb_dir, sparse_dir, gt_dir, max_depth, save_images, with_var)  #  output_including_filled, output_only_filled])

    gt_dir = '../data/Nachsholim/depth_lft/uint16_slam/train/truncate'
    # erase_circles(gt_dir)
    # erase_pixels(gt_dir)

    # image_name = '../pretrained_models/supervised/nachsholim_manual_slam/rearranged/test/mode=dense.data=nachsholim.input=gd.resnet18.epochs20.criterion=REL.lr=0.0001.bs=2.wd=0.pretrained=True.jitter=0.1.time=2021-04-19@16-07_rel_varlog/val_output_var/epoch_10_0000000004.png'
    image_name = '../results/mode=dense.data=nachsholim.input=gd.resnet18.epochs20.criterion=Tukey.lr=0.0001.bs=2.wd=0.pretrained=True.jitter=0.1.rank_metric=diff_3.time=2021-07-04@18-47_and_var_test/val_output_var/epoch_7_0000000000.tif'
    # image_name = '../data/SouthCarolinaCemetery/sparse/uint16/d_1535224912.623127.png'
    # image_name = '../pretrained_models/supervised/nachsholim_manual_slam/rearranged/test/mode=dense.data=nachsholim.input=gd.resnet18.epochs20.criterion=REL.lr=0.0001.bs=2.wd=0.pretrained=True.jitter=0.1.rank_metric=rel_exp.time=2021-06-17@10-51_var+rel_test/val_output_var/epoch_13_0000000018.png'
    depth_im = np.array(Image.open(image_name)).astype(np.float) # / 256.
    present_depth_map(np.sqrt(np.exp(depth_im)))
    # depth_im = cv2.dilate(depth_im, np.ones((3, 3), np.uint8), iterations=1)
    # depth_im[depth_im > 4.0] = 0
    # present_depth_map(depth_im)

    dir1 = '../data/Nachsholim/rearranged/rgb_seaErra/train/'
    dir2 = '../data/Nachsholim/rearranged/rgb/train/'
    dir1_images = sorted(glob.glob(os.path.join(dir1, "*.png")))
    dir2_images = sorted(glob.glob(os.path.join(dir2, "*.png")))
    if len(dir1_images) != len(dir2_images):
        print("Produced different sizes for datasets")
    for i in range(0, len(dir1_images)):
        image_num1 = dir1_images[i].split('/')[-1].split('.')[0].split('_')[2]
        image_num2 = dir2_images[i].split('/')[-1].split('.')[0].split('_')[2]
        if image_num1 != image_num2:
            print("Unmatched files were found")
            print(image_num1)



    dir1 = '../data/Nachsholim/rearranged/rgb_seaErra/val/'
    dir2 = '../data/Nachsholim/rearranged/rgb_seaErra/test_to_train/'
    num = 28024 # 22801  # 28024
    dir1_images = sorted(glob.glob(os.path.join(dir1, "*.png")))
    dir2_images = sorted(glob.glob(os.path.join(dir2, "*.png")))
    for i in range(0, len(dir2_images)):
        image_num2 = dir2_images[i].split('/')[-1].split('.')[0].split('_')[2]
        # image = np.array(Image.open(dir2_images[i])) #.astype("uint16")
        # image = cv2.imread(dir2_images[i])
        num = num + 3
        new_num = str(num).zfill(6)
        image_new_name = os.path.join(dir2, 'input_l_' + new_num + '.png')
        os.rename(dir2_images[i], image_new_name)
        # cv2.imwrite(image_new_name, image)
        # for j in range(i, len(dir1_images)):
        #     image_num1 = '0' + dir1_images[j].split('/')[-1].split('.')[0].split('_')[1]
        #     if image_num1 == image_num2:
        #         break
        #     else:
        #         print(image_num1)
        #         shutil.copy(dir1_images[j], os.path.join(dir1, "filled_maps_new"))
        #         break


if __name__ == '__main__':
    main()
