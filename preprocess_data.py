import cv2
import os
import rawpy
import imageio
import glob


def crop_image(image):
    h, w, c = image.shape
    orig_im = cv2.imread(
        '../data/depth_selection/val_selection_cropped/image/2011_09_26_drive_0002_sync_image_0000000005_image_02.png')
    h_orig, w_orig, c = orig_im.shape
    x = round((h - h_orig) / 2)
    y = round((w - w_orig) / 2)
    crop_img = image[x - 100:x + h_orig - 100, y:y + w_orig]
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
            image_name_cropped = os.path.join(output_dir, image.split('/')[-1].split('.')[0] + "_cropped.png")
            # imageio.imsave(image_name, rgb)
            imageio.imsave(image_name_cropped, cropped_image)


def main():
    video_name = 'seaErraCaesarea.avi'
    # extract_video_images(video_name)
    images_dir = '../data/D5/Raw'
    raw_to_png(images_dir)


if __name__ == '__main__':
    main()
