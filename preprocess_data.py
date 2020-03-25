import cv2
import os


def extract_video_images(video_name):
    video_path = os.path.join('..', video_name)
    video_cap = cv2.VideoCapture(video_path)
    success, image = video_cap.read()
    h, w, c = image.shape
    orig_im = cv2.imread('../data/depth_selection/val_selection_cropped/image/2011_09_26_drive_0002_sync_image_0000000005_image_02.png')
    h_orig, w_orig, c = orig_im.shape
    x = round((h-h_orig)/2)
    y = round((w-w_orig)/2)
    count = 0
    output_dir = os.path.join('..', video_name.split('.')[0] + '_cropped_images')
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    while success:
        image_name = os.path.join(output_dir, video_name.split('.')[0] + "_image_%d.png" % count)
        crop_img = image[x:x + h_orig, y:y + w_orig]
        cv2.imwrite(image_name, crop_img)
        # save frame as PNG file
        if count == 1999:
            return
        success, image = video_cap.read()
        print('Read a new frame: ', image_name)
        count += 1


def main():
    video_name = 'origCaesarea.MOV'
    extract_video_images(video_name)


if __name__ == '__main__':
    main()
