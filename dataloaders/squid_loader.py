import os
import os.path
import glob
import numpy as np
from numpy import linalg as LA
from random import choice
from PIL import Image
import torch.utils.data as data
import cv2
from dataloaders import transforms
from dataloaders.pose_estimator import get_pose_pnp
import yaml


iheight, iwidth = 548, 821  # raw image size
oheight, owidth = 512, 800
# crop_dims = [(352, 640), (256, 480), (192, 352), (416, 736)]
# theight, twidth = crop_dims[np.random.randint(0, len(crop_dims) - 1)]


def load_calib():
    f_name = "dataloaders/nachsholim_cal_resize_lft.yaml"
    with open(f_name) as f:
        parameters = yaml.load(f, Loader=yaml.FullLoader)

    # distortion_coef = [k1, k2, p1, p2, k3]
    fx = parameters['Camera.fx']
    fy = parameters['Camera.fy']
    cx = parameters['Camera.cx']
    cy = parameters['Camera.cy']

    # note: we will take the center crop of the images during augmentation
    # that changes the optical centers, but not focal lengths
    d_width = (iwidth - owidth) / 2
    d_height = (iheight - oheight) / 2
    cx = cx - d_width  # from width = 840 to 832, with a 4-pixel cut on both sides
    cy = cy - d_height  # from width = 560 to 448, with a 56-pixel cut on both sides

    K = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]

    return K


def get_paths_and_transform(split, args):
    assert (args.use_d or args.use_rgb
            or args.use_g), 'no proper input selected'

    if split == "train":
        transform = train_transform
        glob_d = os.path.join(
            args.data_folder,
            'Nachsholim/depth_lft/sparse_manual_slam/train/points/*.png')
        glob_gt = os.path.join(
            args.data_folder,
            'Nachsholim/depth_lft/uint16_slam/train/truncate/*.png')
        glob_rgb = os.path.join(
            args.data_folder,
            'Nachsholim/rgb_unenhanced_slam/train/*.png')

    elif split == "val":
        if args.val == "full":
            transform = val_transform
            glob_d = os.path.join(
                args.data_folder,
                'SQUID/sparse_500/lft/*.png')
            glob_gt = os.path.join(
                args.data_folder,
                'SQUID/gt/lft/uint16/resized/*.png')
            glob_rgb = os.path.join(
                args.data_folder,
                'SQUID/rgb/lft/png/resized/*.png')

        elif args.val == "select":
            transform = val_transform
            glob_d = os.path.join(
                args.data_folder,
                'Nachsholim/depth_lft/sparse_manual_slam/val/points/*.png')
            glob_gt = os.path.join(
                args.data_folder,
                'Nachsholim/depth_lft/uint16_slam/val/truncate/*.png')
            glob_rgb = os.path.join(
                args.data_folder,
                'Nachsholim/rgb_unenhanced_slam/val/*.png')

    elif split == "test_completion":
        transform = no_transform
        glob_d = os.path.join(
            args.data_folder,
            "depth_selection/test_depth_completion_anonymous/velodyne_raw/*.png"
        )
        glob_gt = None  # "test_depth_completion_anonymous/"
        glob_rgb = os.path.join(
            args.data_folder,
            "depth_selection/test_depth_completion_anonymous/image/*.png")
    elif split == "test_prediction":
        transform = no_transform
        glob_d = None
        glob_gt = None  # "test_depth_completion_anonymous/"
        glob_rgb = os.path.join(
            args.data_folder,
            "depth_selection/test_depth_prediction_anonymous/image/*.png")
    else:
        raise ValueError("Unrecognized split " + str(split))

    if glob_gt is not None:
        # train or val-full or val-select
        paths_d = sorted(glob.glob(glob_d))
        paths_gt = sorted(glob.glob(glob_gt))
        paths_rgb = sorted(glob.glob(glob_rgb))
    else:
        # test only has d or rgb
        paths_rgb = sorted(glob.glob(glob_rgb))
        paths_gt = [None] * len(paths_rgb)
        if split == "test_prediction":
            paths_d = [None] * len(
                paths_rgb)  # test_prediction has no sparse depth
        else:
            paths_d = sorted(glob.glob(glob_d))

    if len(paths_d) == 0 and len(paths_rgb) == 0 and len(paths_gt) == 0:
        raise (RuntimeError("Found 0 images under {}".format(glob_gt)))
    if len(paths_d) == 0 and args.use_d:
        raise (RuntimeError("Requested sparse depth but none was found"))
    if len(paths_rgb) == 0 and args.use_rgb:
        raise (RuntimeError("Requested rgb images but none was found"))
    if len(paths_rgb) == 0 and args.use_g:
        raise (RuntimeError("Requested gray images but no rgb was found"))
    if len(paths_rgb) != len(paths_d) or len(paths_rgb) != len(paths_gt):
        raise (RuntimeError("Produced different sizes for datasets"))

    paths = {"rgb": paths_rgb, "d": paths_d, "gt": paths_gt}
    return paths, transform


def rgb_read(filename):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    rgb_png = np.array(img_file, dtype='uint8') # in the range [0,255]
    # rgb_png = np.array(img_file, dtype=float) / 255.0 # scale pixels to the range [0,1]
    # rgb_png[:, :, 0] = np.divide(rgb_png[:, :, 0], np.max(rgb_png[:, :, 0]))
    # rgb_png[:, :, 1] = np.divide(rgb_png[:, :, 1], np.max(rgb_png[:, :, 1]))
    # rgb_png[:, :, 2] = np.divide(rgb_png[:, :, 2], np.max(rgb_png[:, :, 2]))
    img_file.close()
    return rgb_png


def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    depth_png = np.array(img_file, dtype=int)
    img_file.close()
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert np.max(depth_png) > 255, \
        "np.max(depth_png)={}, path={}".format(np.max(depth_png), filename)

    depth = depth_png.astype(np.float) / 256.
    depth[depth > 15] = 0
    depth = np.expand_dims(depth, -1)
    return depth


def drop_depth_measurements(depth, prob_keep):
    mask = np.random.binomial(1, prob_keep, depth.shape)
    depth *= mask
    return depth


def train_transform(rgb, sparse, target, rgb_near, args):
    # s = np.random.uniform(1.0, 1.5)  # random scaling
    # angle = np.random.uniform(-5.0, 5.0)  # random rotation degrees
    do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip

    transform_geometric = transforms.Compose([
        # transforms.Rotate(angle),
        # transforms.Resize(0.5),
        # transforms.BottomCrop((oheight, owidth)),
        transforms.CenterCrop((oheight, owidth)),
        transforms.HorizontalFlip(do_flip)
    ])
    if sparse is not None:
        sparse = transform_geometric(sparse)
    target = transform_geometric(target)
    if rgb is not None:
        brightness = np.random.uniform(max(0, 1 - args.jitter), 1 + args.jitter)
        contrast = np.random.uniform(max(0, 1 - args.jitter), 1 + args.jitter)
        saturation = np.random.uniform(max(0, 1 - args.jitter), 1 + args.jitter)
        transform_rgb = transforms.Compose([
            transforms.ColorJitter(brightness, contrast, saturation, 0),
            transform_geometric])
        rgb = transform_geometric(rgb)
        # rgb = transform_rgb(rgb)
        if rgb_near is not None:
            rgb_near = transform_geometric(rgb_near)
            # rgb_near = transform_rgb(rgb_near)
    # sparse = drop_depth_measurements(sparse, 0.9)

    return rgb, sparse, target, rgb_near


def val_transform(rgb, sparse, target, rgb_near, args):
    transform = transforms.Compose([
        transforms.CenterCrop((oheight, owidth)),
        # transforms.BottomCrop((oheight, owidth)),
    ])
    if rgb is not None:
        rgb = transform(rgb)
    if sparse is not None:
        sparse = transform(sparse)
    if target is not None:
        target = transform(target)
    if rgb_near is not None:
        rgb_near = transform(rgb_near)
    return rgb, sparse, target, rgb_near


def no_transform(rgb, sparse, target, rgb_near, args):
    return rgb, sparse, target, rgb_near


to_tensor = transforms.ToTensor()
to_float_tensor = lambda x: to_tensor(x).float()


def handle_gray(rgb, args):
    if rgb is None:
        return None, None
    if not args.use_g:
        return rgb, None
    else:
        img = np.array(Image.fromarray(rgb).convert('L'))
        # img = np.maximum(rgb[:, :, 2], rgb[:, :, 1]) - rgb[:, :, 0]
        img = np.expand_dims(img, -1)
        if not args.use_rgb:
            rgb_ret = None
        else:
            rgb_ret = rgb
        return rgb_ret, img


def get_rgb_near(path, args):
    assert path is not None, "path is None"

    def extract_frame_id(filename):
        head, tail = os.path.split(filename)
        number_string = tail[0:tail.find('.')].split('_')[-1]
        number = int(number_string)
        return head, number

    def get_nearby_filename(filename, new_id):
        head, tail = os.path.split(filename)
        number_string = tail[0:tail.find('.')].split('_')[-1]
        image_head = tail[0:tail.find(number_string)]
        new_filename = os.path.join(head, image_head + str(new_id).zfill(6) + '.png')
        return new_filename

    head, number = extract_frame_id(path)
    count = 0
    max_frame_diff = 3
    candidates = [
        (i - max_frame_diff)*3 for i in range(max_frame_diff * 2 + 1)
        if i - max_frame_diff != 0
    ]
    while True:
        random_offset = choice(candidates)
        path_near = get_nearby_filename(path, number + random_offset)
        if os.path.exists(path_near):
            break
        assert count < 20, "cannot find a nearby frame in 20 trials for {}".format(path_rgb_tgt)

    return rgb_read(path_near)


class SQUIDDepth(data.Dataset):
    """A data loader for the SQUID dataset
    """

    def __init__(self, split, args):
        self.args = args
        self.split = split
        paths, transform = get_paths_and_transform(split, args)
        self.paths = paths
        self.transform = transform
        self.K = load_calib()
        self.threshold_translation = 0.1

    def __getraw__(self, index):
        rgb = rgb_read(self.paths['rgb'][index]) if \
            (self.paths['rgb'][index] is not None and (self.args.use_rgb or self.args.use_g)) else None
        sparse = depth_read(self.paths['d'][index]) if \
            (self.paths['d'][index] is not None and self.args.use_d) else None
        target = depth_read(self.paths['gt'][index]) if \
            self.paths['gt'][index] is not None else None
        rgb_near = get_rgb_near(self.paths['rgb'][index], self.args) if \
            self.split == 'train' and self.args.use_pose else None
        return rgb, sparse, target, rgb_near

    def __getitem__(self, index):
        rgb, sparse, target, rgb_near = self.__getraw__(index)
        rgb, sparse, target, rgb_near = self.transform(rgb, sparse, target,
                                                       rgb_near, self.args)
        r_mat, t_vec = None, None
        if self.split == 'train' and self.args.use_pose:
            success, r_vec, t_vec = get_pose_pnp(rgb, rgb_near, sparse, self.K)
            # discard if translation is too small
            success = success and LA.norm(t_vec) > self.threshold_translation
            if success:
                r_mat, _ = cv2.Rodrigues(r_vec)
            else:
                # return the same image and no motion when PnP fails
                rgb_near = rgb
                t_vec = np.zeros((3, 1))
                r_mat = np.eye(3)

        rgb, gray = handle_gray(rgb, self.args)
        candidates = {"rgb": rgb, "d": sparse, "gt": target, \
                      "g": gray, "r_mat": r_mat, "t_vec": t_vec, "rgb_near": rgb_near}
        items = {
            key: to_float_tensor(val)
            for key, val in candidates.items() if val is not None
        }

        return items

    def __len__(self):
        return len(self.paths['gt'])