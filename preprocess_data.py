import cv2
import os


def extract_video_images(video_name):
    video_path = os.path.join('..', video_name)
    video_cap = cv2.VideoCapture(video_path)
    success, image = video_cap.read()
    count = 0
    output_dir = os.path.join('..', video_name.split('.')[0] + '_images')
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    while success:
        image_name = os.path.join(output_dir, video_name.split('.')[0] + "_image_%d.png" % count)
        cv2.imwrite(image_name, image)  # save frame as PNG file
        success, image = video_cap.read()
        print('Read a new frame: ', image_name)
        count += 1


def main():
    video_name = 'seaErraCaesarea.avi'
    extract_video_images(video_name)


if __name__ == '__main__':
    main()
