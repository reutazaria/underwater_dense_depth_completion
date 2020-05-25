import cv2
import os
import rawpy
import imageio
import glob
from PIL import Image
# import matplotlib as mpl
# import matplotlib.pyplot as plt
import numpy as np
import random
import tifffile as tiff


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
    output_dir = os.path.join(images_dir, "png_cropped")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for image in glob.glob(os.path.join(images_dir, "*NEF")):
        with rawpy.imread(image) as raw:
            rgb = raw.postprocess()
            scale = 0.1648
            width = int(rgb.shape[1] * scale)
            height = int(rgb.shape[0] * scale)
            dim = (width, height)
            resized_rgb = cv2.resize(rgb, dim, interpolation=cv2.INTER_AREA)
            cropped_image = crop_image(resized_rgb)
            image_name_cropped = os.path.join(output_dir, image.split('/')[-1].split('.')[0] + ".png")
            print("saving image: ", image_name_cropped)
            imageio.imsave(image_name_cropped, cropped_image)


def tif_to_png(maps_dir):
    output_dir = os.path.join(maps_dir, "png")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    output_dir_resized = os.path.join(maps_dir, "png_resized")
    if not os.path.isdir(output_dir_resized):
        os.mkdir(output_dir_resized)
    output_dir_cropped = os.path.join(maps_dir, "cropped_png")
    if not os.path.isdir(output_dir_cropped):
        os.mkdir(output_dir_cropped)
    # output_dir_colored = os.path.join(maps_dir, "colored_png")
    # if not os.path.isdir(output_dir_colored):
    #     os.mkdir(output_dir_colored)
    for image in glob.glob(os.path.join(maps_dir, "tif")):
        # for converting Tif to Png from terminal, run:
        # gdal_translate -of PNG <<current_image>> <<new_image_name>>
        im = Image.open(image)
        image_name = os.path.join(output_dir, image.split('/')[-1].split('.')[0] + ".png")
        imageio.imsave(image_name, im)
        png_im = np.array(Image.open(image_name)).astype("uint16")
        # png_im = Image.open(image_name)
        orig_im = cv2.imread(
            '../data/depth_selection/val_selection_cropped/image/2011_09_26_drive_0002_sync_image_0000000005_image_02.png')
        h_orig, w_orig, c = orig_im.shape
        scale = w_orig / png_im.shape[1]
        width = int(png_im.shape[1] * scale)
        height = int(png_im.shape[0] * scale)
        dim = (width, height)
        resized_map = cv2.resize(png_im, dim, interpolation=cv2.INTER_AREA)
        resized_name = os.path.join(output_dir_resized, image.split('/')[-1].split('.')[0] + "_resized.png")
        imageio.imsave(resized_name, resized_map)
        # cropped_image = crop_image(resized_rgb)
        # resized_map = im.resize(dim, Image.ANTIALIAS)

        x = round((height - h_orig) / 2)
        y = round((width - w_orig) / 2)
        cropped_image = resized_map[x:x + h_orig, y:y + w_orig]
        image_name_cropped = os.path.join(output_dir_cropped, image.split('/')[-1].split('.')[0] + "_cropped.png")
        print("saving image: ", image_name_cropped)
        imageio.imsave(image_name_cropped, cropped_image)

        # convert grayscale to RGB
        # gray_im = cv2.imread(image_name_cropped, 0)
        # cm = plt.get_cmap('gist_rainbow')
        # colored_image = cm(gray_im)
        # colored_image_name = os.path.join(output_dir_colored, image.split('/')[-1].split('.')[0] + "_colored.png")
        # Image.fromarray((colored_image[:, :, :3] * 255).astype(np.uint8)).save(colored_image_name)


def gt_to_sparse(maps_dir):
    output_dir_cropped = os.path.join(maps_dir, "cropped_sparse_10000_png")
    if not os.path.isdir(output_dir_cropped):
        os.mkdir(output_dir_cropped)
    for image in glob.glob(os.path.join(maps_dir, "*.png")):
        png_im = np.array(Image.open(image)).astype("uint16")
        new_depth = np.zeros(png_im.shape)
        y_idx, x_idx = np.where(png_im > 0)  # list of all the indices with pixel value 1
        chosen_pixels = random.sample(range(0, x_idx.size), k=10000)  # k=int(x_idx.size * 0.1)
        for i in range(0, len(chosen_pixels)):
            rand_idx = chosen_pixels[i]  # randomly choose any element in the x_idx list
            x = x_idx[rand_idx]
            y = y_idx[rand_idx]
            new_depth[y, x] = png_im[y, x]
        image_name_cropped = os.path.join(output_dir_cropped, image.split('/')[-1].split('.')[0] + "_sparse.png")
        print("saving image: ", image_name_cropped)
        imageio.imsave(image_name_cropped, new_depth)



def main():
    # video_name = 'seaErraCaesarea.avi'
    # extract_video_images(video_name)
    images_dir = '../data/D5/Raw'
    # raw_to_png_orig(images_dir)
    # raw_to_png(images_dir)
    # depthmaps_dir = '../data/D5/depthMaps_2020_04_16'
    # depthmaps_dir = '../data/depth_selection/val_selection_cropped/sparse_depth_maps'
    # tif_to_png(depthmaps_dir)
    depthmaps_png = '../data/D5/depthMaps_2020_04_16/cropped_png'
    gt_to_sparse(depthmaps_png)
    images_dir = '../data/D5/Raw/png_resized'


if __name__ == '__main__':
    main()
