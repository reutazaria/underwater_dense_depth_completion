import cv2
import os
import rawpy
import imageio
import glob
from PIL import Image
# import matplotlib as mpl
# import matplotlib.pyplot as plt
import numpy as np
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
    output_dir = os.path.join('..', video_name.split('.')[0] + '_cropped_images_up')
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    while success:
        image_name = os.path.join(output_dir, video_name.split('.')[0] + "_image_%d.png" % count)
        crop_img = crop_image(image)
        cv2.imwrite(image_name, crop_img)
        # save frame as PNG file
        # if count == 1999:
        #     return
        success, image = video_cap.read()
        print('Read a new frame: ', image_name)
        count += 1


def raw_to_png(images_dir):
    output_dir = os.path.join(images_dir, "png_cropped")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for image in glob.glob(os.path.join(images_dir, "*NEF")):
        with rawpy.imread(image) as raw:
            rgb = raw.postprocess()
            scale = 0.2
            width = int(rgb.shape[1] * scale)
            height = int(rgb.shape[0] * scale)
            dim = (width, height)
            resized_rgb = cv2.resize(rgb, dim, interpolation=cv2.INTER_AREA)
            cropped_image = crop_image(resized_rgb)
            # image_name = os.path.join(output_dir, image.split('.')[0] + ".png")
            # imageio.imsave(image_name, rgb)
            image_name_cropped = os.path.join(output_dir, image.split('/')[-1].split('.')[0] + "_cropped.png")
            print("saving image: ", image_name_cropped)
            imageio.imsave(image_name_cropped, cropped_image)


def tif_to_png(maps_dir):
    output_dir = os.path.join(maps_dir, "png")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    output_dir_cropped = os.path.join(maps_dir, "cropped_png")
    if not os.path.isdir(output_dir_cropped):
        os.mkdir(output_dir_cropped)
    # output_dir_colored = os.path.join(maps_dir, "colored_png")
    # if not os.path.isdir(output_dir_colored):
    #     os.mkdir(output_dir_colored)
    for image in glob.glob(os.path.join(maps_dir, "*tif")):
        # for converting Tif to Png from terminal, run:
        # gdal_translate -of PNG <<current_image>> <<new_image_name>>
        im = Image.open(image)
        image_name = os.path.join(output_dir, image.split('/')[-1].split('.')[0] + ".png")
        imageio.imsave(image_name, im)
        png_im = np.array(Image.open(image_name)).astype("uint16")
        # png_im = Image.open(image_name)
        scale = 0.2
        width = int(png_im.shape[1] * scale)
        height = int(png_im.shape[0] * scale)
        dim = (width, height)
        resized_map = cv2.resize(png_im, dim, interpolation=cv2.INTER_AREA)
        # cropped_image = crop_image(resized_rgb)
        # resized_map = im.resize(dim, Image.ANTIALIAS)
        orig_im = cv2.imread(
            '../data/depth_selection/val_selection_cropped/image/2011_09_26_drive_0002_sync_image_0000000005_image_02.png')
        h_orig, w_orig, c = orig_im.shape
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


def main():
    video_name = 'seaErraCaesarea.avi'
    # extract_video_images(video_name)
    images_dir = '../data/D5/Raw'
    # raw_to_png(images_dir)
    depthmaps_dir = '../data/D5/depthMaps'
    tif_to_png(depthmaps_dir)


if __name__ == '__main__':
    main()
