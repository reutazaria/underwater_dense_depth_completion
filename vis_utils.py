import os
if not ("DISPLAY" in os.environ):
    import matplotlib as mpl
    mpl.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2

cmap = plt.cm.jet


def depth_colorize(depth):
    depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    depth_color = 255 * cmap(depth)[:, :, :3]  # H, W, C
    return depth_color.astype('uint8')


def var_colorize(var):
    var = (var - np.min(var)) / (np.max(var) - np.min(var))
    var_color = 255 * plt.cm.gray(var)[:, :, :3]  # H, W, C
    return var_color.astype('uint8')


def merge_into_col(ele, pred, log_var):
    def preprocess_depth(x):
        y = np.squeeze(x.data.cpu().numpy())
        return y
        # return depth_colorize(y)

    # if is gray, transforms to rgb
    img_list = []
    if 'rgb' in ele:
        rgb = np.squeeze(ele['rgb'][0, ...].data.cpu().numpy())
        rgb = np.transpose(rgb, (1, 2, 0))
        img_list.append(rgb)
    elif 'g' in ele:
        g = np.squeeze(ele['g'][0, ...].data.cpu().numpy())
        g = np.array(Image.fromarray(g).convert('RGB'))
        img_list.append(g)
    depth_im = preprocess_depth(pred[0, ...])
    # add depth images
    if 'gt' in ele:
        gt_im = preprocess_depth(ele['gt'][0, ...])
        out_gt = preprocess_depth(pred[0, ...])
        out_gt[gt_im == 0] = 0
        depth_im = np.concatenate((depth_im, out_gt, gt_im), axis=0)
        # diff_im = gt_im - preprocess_depth(pred[0, ...])
        # diff_im[gt_im == 0] = 0
        # depth_im = np.concatenate((depth_im, diff_im), axis=0)
    if 'd' in ele:
        d_im = preprocess_depth(ele['d'][0, ...])
        d_im_dilate = cv2.dilate(d_im, np.ones((3, 3), np.uint8), iterations=1)
        depth_im = np.concatenate((d_im_dilate, depth_im), axis=0)
    img_list.append(depth_colorize(depth_im))
    if log_var:
        uncertainty_map = preprocess_depth(log_var[0, ...])
        img_list.append(var_colorize(uncertainty_map))

    # if 'd' in ele:
    #     img_list.append(preprocess_depth(ele['d'][0, ...]))
    # img_list.append(preprocess_depth(pred[0, ...]))
    # if 'gt' in ele:
    #     img_list.append(preprocess_depth(ele['gt'][0, ...]))
    #     # im_diff = ele['gt'][0, ...] - pred[0, ...]
    #     # img_list.append(preprocess_depth(im_diff[0, ...]))

    img_merge = np.vstack(img_list)
    # img_merge = np.hstack(img_list)
    return img_merge.astype('uint8')


def add_col(img_merge, row):
    return np.hstack([img_merge, row])


def save_image(img_merge, filename):
    image_to_write = cv2.cvtColor(img_merge, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, image_to_write)


def save_var_image(img, filename):
    image_to_write = np.exp(img).astype(np.float)
    cv2.imwrite(filename, image_to_write)


def save_depth_as_uint16png(img, filename):
    img = (img * 256).astype('uint16')
    cv2.imwrite(filename, img)


if ("DISPLAY" in os.environ):
    f, axarr = plt.subplots(4, 1)
    plt.tight_layout()
    plt.ion()


def display_warping(rgb_tgt, pred_tgt, warped):
    def preprocess(rgb_tgt, pred_tgt, warped):
        rgb_tgt = 255 * np.transpose(np.squeeze(rgb_tgt.data.cpu().numpy()),
                                     (1, 2, 0))  # H, W, C
        # depth = np.squeeze(depth.cpu().numpy())
        # depth = depth_colorize(depth)

        # convert to log-scale
        pred_tgt = np.squeeze(pred_tgt.data.cpu().numpy())
        # pred_tgt[pred_tgt<=0] = 0.9 # remove negative predictions
        # pred_tgt = np.log10(pred_tgt)

        pred_tgt = depth_colorize(pred_tgt)

        warped = 255 * np.transpose(np.squeeze(warped.data.cpu().numpy()),
                                    (1, 2, 0))  # H, W, C
        recon_err = np.absolute(
            warped.astype('float') - rgb_tgt.astype('float')) * (warped > 0)
        recon_err = recon_err[:, :, 0] + recon_err[:, :, 1] + recon_err[:, :, 2]
        recon_err = depth_colorize(recon_err)
        return rgb_tgt.astype('uint8'), warped.astype(
            'uint8'), recon_err, pred_tgt

    rgb_tgt, warped, recon_err, pred_tgt = preprocess(rgb_tgt, pred_tgt,
                                                      warped)

    # 1st column
    column = 0
    axarr[0].imshow(rgb_tgt)
    axarr[0].axis('off')
    axarr[0].axis('equal')
    # axarr[0, column].set_title('rgb_tgt')

    axarr[1].imshow(warped)
    axarr[1].axis('off')
    axarr[1].axis('equal')
    # axarr[1, column].set_title('warped')

    axarr[2].imshow(recon_err, 'hot')
    axarr[2].axis('off')
    axarr[2].axis('equal')
    # axarr[2, column].set_title('recon_err error')

    axarr[3].imshow(pred_tgt, 'hot')
    axarr[3].axis('off')
    axarr[3].axis('equal')
    # axarr[3, column].set_title('pred_tgt')

    # plt.show()
    plt.pause(0.001)
