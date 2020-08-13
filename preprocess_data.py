import shutil

import cv2
import os
import rawpy
import imageio
import glob

from PIL import Image
from scipy.interpolate import griddata
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
        chosen_pixels = random.sample(range(0, x_idx.size), k=min(x_idx.size, n_samples))  # k=int(x_idx.size * 0.1)
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

        # # colorize image
        # depth_norm = (interpolated_im_u - np.min(interpolated_im_u)) / \
        #              (np.max(interpolated_im_u) - np.min(interpolated_im_u))
        # cmap = plt.cm.jet
        # depth_color = 255 * cmap(depth_norm)[:, :, :3]  # H, W, C
        # interp_color = depth_color.astype('uint8')
        # image_to_write = cv2.cvtColor(interp_color, cv2.COLOR_RGB2BGR)
        # image_name_color = os.path.join(output_dir_sparse, image.split('/')[-1].split('.')[0] + "_interp_col.png")
        # cv2.imwrite(image_name_color, image_to_write)


def save_depth_as_uint16(maps_dir):
    output_dir = os.path.join(maps_dir, "uint16")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for image in glob.glob(os.path.join(maps_dir, "*.tif")):
        png_im = np.array(Image.open(image))  # .convert('L'))
        img = (png_im * 256).astype('uint16')
        file_name = os.path.join(output_dir, image.split('/')[-1].split('.')[0] + ".png")
        cv2.imwrite(file_name, img)


def colorize_depth():
    img_list = []
    diff_list = []
    rgb_list = []
    gt_dir = '../data/SouthCarolinaCave/depthMaps/uint16/test'
    gt_images = sorted(os.listdir(gt_dir))
    sparse_dir = '../data/SouthCarolinaCave/depthMaps/sparse/test'
    sparse_images = sorted(os.listdir(sparse_dir))
    pred_d_dir = '../data/SouthCarolinaCave/results/d'
    pred_d_images = sorted(os.listdir(pred_d_dir))
    pred_rgb_dir = '../data/SouthCarolinaCave/results/rgb'
    pred_rgb_images = sorted(os.listdir(pred_rgb_dir))
    pred_rgbd_dir = '../data/SouthCarolinaCave/results/rgbd'
    pred_rgbd_images = sorted(os.listdir(pred_rgbd_dir))
    linear_interp_dir = '../data/SouthCarolinaCave/depthMaps/interp/test'
    interp_images = sorted(os.listdir(linear_interp_dir))
    rgb_dir = '../data/SouthCarolinaCave/cave_seaerra_lft_to1500/png/test'
    rgb_images = sorted(os.listdir(rgb_dir))

    oheight, owidth = 512, 800  # 448, 832  #
    transform_geometric = transforms.Compose([
        transforms.BottomCrop((oheight, owidth))])

    skip = 80
    top_percent = 90
    for i in range(3):  # range(0, len(gt_images)):
        gt_im = np.array(Image.open(os.path.join(gt_dir, gt_images[i*skip])))
        gt_im = gt_im.astype(np.float) / 256.
        gt_im = transform_geometric(gt_im)
        gt_im[gt_im > np.percentile(gt_im, top_percent)] = 0.0
        sparse_im = np.array(Image.open(os.path.join(sparse_dir, sparse_images[i*skip])))
        sparse_im = sparse_im.astype(np.float) / 256.
        sparse_im = transform_geometric(sparse_im)
        sparse_im[sparse_im > np.percentile(gt_im, top_percent)] = 0.0
        pred_d_im = np.array(Image.open(os.path.join(pred_d_dir, pred_d_images[i*skip])))
        pred_d_im = pred_d_im.astype(np.float) / 256.
        pred_d_im[pred_d_im > np.percentile(gt_im, top_percent)] = 0.0
        pred_rgb_im = np.array(Image.open(os.path.join(pred_rgb_dir, pred_rgb_images[i*skip])))
        pred_rgb_im = pred_rgb_im.astype(np.float) / 256.
        pred_rgb_im[pred_rgb_im > np.percentile(gt_im, top_percent)] = 0.0
        pred_rgbd_im = np.array(Image.open(os.path.join(pred_rgbd_dir, pred_rgbd_images[i*skip])))
        pred_rgbd_im = pred_rgbd_im.astype(np.float) / 256.
        pred_rgbd_im[pred_rgbd_im > np.percentile(gt_im, top_percent)] = 0.0
        interp_im = np.array(Image.open(os.path.join(linear_interp_dir, interp_images[i*skip])))
        interp_im = interp_im.astype(np.float) / 256.
        interp_im = transform_geometric(interp_im)
        interp_im[interp_im > np.percentile(gt_im, top_percent)] = 0.0
        rgb_im = np.array(Image.open(os.path.join(rgb_dir, rgb_images[i*skip])))
        rgb_im = transform_geometric(rgb_im)

        depth_im = np.concatenate((sparse_im, pred_d_im, pred_rgb_im, pred_rgbd_im, gt_im, interp_im), axis=0)
        img_list.append(depth_im)

        diff_im_d = gt_im - pred_d_im
        diff_im_rgb = gt_im - pred_rgb_im
        diff_im_rgbd = gt_im - pred_rgbd_im
        diff_im_interp = gt_im - interp_im
        # diff_im = np.concatenate((pred_d_im, gt_im, diff_im_d), axis=0)
        diff_im = np.concatenate((diff_im_d, diff_im_rgb, diff_im_rgbd, diff_im_interp), axis=0)
        diff_list.append(diff_im)

        rgb_list.append(rgb_im)

    depth_tot = np.hstack(img_list)
    ax = plt.gca()
    im = ax.imshow(depth_tot, cmap="jet")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="7%", pad="2%")
    plt.colorbar(im, cax=cax, orientation='horizontal', label='depth [m]')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

    diff_tot = np.hstack(diff_list)
    ax2 = plt.gca()
    im2 = ax2.imshow(diff_tot, cmap="jet")
    divider = make_axes_locatable(ax2)
    cax2 = divider.append_axes("bottom", size="7%", pad="2%")
    plt.colorbar(im2, cax=cax2, orientation='horizontal', label='depth [m]')
    ax2.set_xticks([])
    ax2.set_yticks([])
    plt.show()

    # rgb_tot = np.hstack(rgb_list)
    # ax = plt.gca()
    # im = ax.imshow(rgb_tot)


def calc_errors(gt_dir, interp_dir):
    RMSE = 0
    MAE = 0
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

        depth_gt[depth_gt > np.percentile(depth_gt, 90)] = 0
        valid_mask = depth_gt > 0.1
        # convert from meters to mm
        target_mm = 1e3 * depth_gt[valid_mask]
        output_mm = 1e3 * depth_interp[valid_mask]
        RMSE += np.sqrt(np.mean((output_mm - target_mm) ** 2))
        MAE += float(np.mean(np.abs(output_mm - target_mm)))
        count += 1
    print("RMSE: ", RMSE/count)
    print("MAE: ", MAE/count)


def make_train_val_sets(input_dir, depth_dir):
    input_val_dir = os.path.join(input_dir, '../val')
    if not os.path.isdir(input_val_dir):
        os.mkdir(input_val_dir)
    depth_val_dir = os.path.join(depth_dir, '../val')
    if not os.path.isdir(depth_val_dir):
        os.mkdir(depth_val_dir)
    input_images = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    depth_images = sorted(glob.glob(os.path.join(depth_dir, "*.png")))
    n_samples = len(os.listdir(input_dir))
    n_val = round(n_samples*0.2)
    selected_val = random.sample(range(1, n_samples), k=n_val)
    for i in selected_val:
        shutil.move(input_images[i], input_val_dir)
        shutil.move(depth_images[i], depth_val_dir)


def rename_files():
    dir_name = '../data/SouthCarolinaCave/cave_seaerra_lft_to1500/png/'
    for f in os.listdir(dir_name):
        if os.path.isfile(os.path.join(dir_name, f)) and ('input' in f):
            splits = f.split('_')
            new_name = splits[0] + '_SeaErra' + splits[2]
            os.rename(os.path.join(dir_name, f), os.path.join(dir_name, new_name))


def main():
    # video_name = 'seaErraCaesarea.avi'
    # extract_video_images(video_name)
    # images_dir = '../data/D5/Raw'
    # raw_to_png_orig(images_dir)
    # raw_to_png(images_dir)

    # depthmaps_dir = '../data/D5/depthMaps_2020_04_16/png'
    # depthmaps_dir = '../data/SouthCarolinaCave/cave_seaerra_lft_to1500/'
    # tif_to_png(depthmaps_dir)

    # depthmaps_dir = '../data/D5/depthMaps_2020_04_16/tif'
    # save_depth_as_uint16(depthmaps_dir)

    # depthmaps_png = '../data/SouthCarolinaCave/depthMaps/uint16/val'
    # n_samples = 500
    # gt_to_sparse(depthmaps_png, n_samples)

    # gt_dir = '../data/SouthCarolinaCave/depthMaps/uint16/test'
    # interp_dir = '../data/SouthCarolinaCave/depthMaps/interp/test'
    # calc_errors(gt_dir, interp_dir)

    # rename_files()

    colorize_depth()


if __name__ == '__main__':
    main()
