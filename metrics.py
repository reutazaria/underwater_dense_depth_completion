import torch
import math
import numpy as np

lg_e_10 = math.log(10)


def log10(x):
    """Convert a new tensor with the base-10 logarithm of the elements of x. """
    return torch.log(x) / lg_e_10


class Result(object):
    def __init__(self):
        self.irmse = 0
        self.imae = 0
        self.mse = 0
        self.rmse = 0
        self.mae = 0
        self.absrel = 0
        self.squared_rel = 0
        self.rel_squared = 0
        self.rel_exp = 0
        self.diff_thresh = 0
        self.rmse_3 = 0
        self.diff_3 = 0
        self.rmse_6 = 0
        self.diff_6 = 0
        self.lg10 = 0
        self.delta1 = 0
        self.delta2 = 0
        self.delta3 = 0
        self.data_time = 0
        self.gpu_time = 0
        self.silog = 0  # Scale invariant logarithmic error [log(m)*100]
        self.avg_target = 0
        self.avg_pred = 0
        self.pearson = 0
        self.pearson_gb = 0
        self.loss = 0
        self.depth_loss = 0
        self.smooth_loss = 0
        self.photometric_loss = 0
        self.variance_loss = 0

    def set_to_worst(self):
        self.irmse = np.inf
        self.imae = np.inf
        self.mse = np.inf
        self.rmse = np.inf
        self.mae = np.inf
        self.absrel = np.inf
        self.squared_rel = np.inf
        self.rel_squared = np.inf
        self.rel_exp = np.inf
        self.diff_thresh = np.inf
        self.diff_3 = np.inf
        self.diff_6 = np.inf
        self.lg10 = np.inf
        self.silog = np.inf
        self.delta1 = 0
        self.delta2 = 0
        self.delta3 = 0
        self.data_time = 0
        self.gpu_time = 0

    def update(self, irmse, imae, mse, rmse, mae, absrel, squared_rel, rel_squared, rel_exp, diff_thresh, rmse_3,
               diff_3, rmse_6, diff_6, lg10, delta1, delta2, delta3, gpu_time, data_time, silog, avg_target, avg_pred,
               pearson, pearson_gb, loss=0, depth=0, smooth=0, photometric=0, variance=0):
        self.irmse = irmse
        self.imae = imae
        self.mse = mse
        self.rmse = rmse
        self.mae = mae
        self.absrel = absrel
        self.squared_rel = squared_rel
        self.rel_squared = rel_squared
        self.rel_exp = rel_exp
        self.diff_thresh = diff_thresh
        self.rmse_3 = rmse_3
        self.diff_3 = diff_3
        self.rmse_6 = rmse_6
        self.diff_6 = diff_6
        self.lg10 = lg10
        self.delta1 = delta1
        self.delta2 = delta2
        self.delta3 = delta3
        self.data_time = data_time
        self.gpu_time = gpu_time
        self.silog = silog
        self.avg_target = avg_target
        self.avg_pred = avg_pred
        self.pearson = pearson
        self.pearson_gb = pearson_gb
        self.loss = loss
        self.depth_loss = depth
        self.smooth_loss = smooth
        self.photometric_loss = photometric
        self.variance_loss = variance

    def evaluate(self, output, target, rgb, loss=0, depth=0, smooth=0, photometric=0, variance=0):

        valid_mask = target > 0.5
        valid_mask_3 = target <= 3
        valid_mask_6 = target > 3

        # convert from meters to mm
        output_mm = 1e3 * output[valid_mask]
        target_mm = 1e3 * target[valid_mask]

        output_mm_3 = 1e3 * output[valid_mask * valid_mask_3]
        target_mm_3 = 1e3 * target[valid_mask * valid_mask_3]
        diff_3 = output_mm_3 - target_mm_3

        output_mm_6 = 1e3 * output[valid_mask * valid_mask_6]
        target_mm_6 = 1e3 * target[valid_mask * valid_mask_6]
        diff_6 = output_mm_6 - target_mm_6

        abs_diff = (output_mm - target_mm).abs()

        self.avg_target = float(target_mm.mean())
        self.avg_pred = float(output_mm.mean())
        self.mse = float((torch.pow(abs_diff, 2)).mean())
        self.rmse = math.sqrt(self.mse)
        self.mae = float(abs_diff.mean())
        self.lg10 = float((log10(output_mm) - log10(target_mm)).abs().mean())
        self.absrel = float((abs_diff / target_mm).mean())
        # self.squared_rel = float(((abs_diff / target_mm)**2).mean())
        self.squared_rel = float(1e3 * (((output[valid_mask] - target[valid_mask]).abs() / target[valid_mask]) ** 2).mean())
        # self.rel_squared = float((abs_diff / target_mm ** 2).mean())
        self.rel_squared = float(1e3 * ((output[valid_mask] - target[valid_mask]).abs() / target[valid_mask]**2).mean())
        # self.rel_exp = float((abs_diff / target_mm.exp()).mean())
        self.rel_exp = 1e3*float(((output[valid_mask] - target[valid_mask]).abs() / target[valid_mask].exp()).mean())
        if len(target_mm_3) > 0:
            self.diff_3 = len(diff_3[diff_3 > 100])/len(diff_3) * 100
            self.rmse_3 = math.sqrt(float((torch.pow(diff_3, 2)).mean()))
        else:
            self.diff_3 = 0
            self.rmse_3 = 0

        if len(target_mm_6) > 0:
            self.diff_6 = len(diff_6[diff_6 > 300])/len(diff_6) * 100
            self.rmse_6 = math.sqrt(float((torch.pow(diff_6, 2)).mean()))
        else:
            self.diff_6 = 0
            self.rmse_6 = 0

        self.diff_thresh = self.diff_3 + self.diff_6

        if rgb is not None:
            gb = torch.max(rgb[:, 2, :, :], rgb[:, 1, :, :]) - rgb[:, 0, :, :]
            gb = gb.unsqueeze(1)
            vx = gb - gb.mean()
            vy = output - output.mean()
            self.pearson_gb = float((vx * vy).sum() / ((vx ** 2).sum().sqrt() * (vy ** 2).sum().sqrt()))
        else:
            self.pearson_gb = 0

        vx = target_mm - target_mm.mean()
        vy = output_mm - output_mm.mean()
        self.pearson = float((vx * vy).sum() / ((vx ** 2).sum().sqrt() * (vy ** 2).sum().sqrt()))

        maxRatio = torch.max(output_mm / target_mm, target_mm / output_mm)
        self.delta1 = float((maxRatio < 1.25).float().mean())
        self.delta2 = float((maxRatio < 1.25**2).float().mean())
        self.delta3 = float((maxRatio < 1.25**3).float().mean())
        self.data_time = 0
        self.gpu_time = 0

        # silog uses meters
        err_log = torch.log(target[valid_mask]) - torch.log(output[valid_mask])
        normalized_squared_log = (err_log**2).mean()
        log_mean = err_log.mean()
        self.silog = normalized_squared_log - log_mean * log_mean  # * 100

        # convert from meters to km
        inv_output_km = (1e-3 * output[valid_mask])**(-1)
        inv_target_km = (1e-3 * target[valid_mask])**(-1)
        abs_inv_diff = (inv_output_km - inv_target_km).abs()
        self.irmse = math.sqrt((torch.pow(abs_inv_diff, 2)).mean())
        self.imae = float(abs_inv_diff.mean())

        self.loss = float(loss)
        self.depth_loss = float(depth)
        self.smooth_loss = float(smooth)
        self.photometric_loss = float(photometric)
        self.variance_loss = float(variance)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0.0
        self.sum_irmse = 0
        self.sum_imae = 0
        self.sum_mse = 0
        self.sum_rmse = 0
        self.sum_mae = 0
        self.sum_absrel = 0
        self.sum_squared_rel = 0
        self.sum_rel_squared = 0
        self.sum_rel_exp = 0
        self.sum_diff_thresh = 0
        self.sum_rmse_3 = 0
        self.sum_diff_3 = 0
        self.sum_rmse_6 = 0
        self.sum_diff_6 = 0
        self.sum_lg10 = 0
        self.sum_delta1 = 0
        self.sum_delta2 = 0
        self.sum_delta3 = 0
        self.sum_data_time = 0
        self.sum_gpu_time = 0
        self.sum_silog = 0
        self.sum_avg_target = 0
        self.sum_avg_pred = 0
        self.sum_pearson = 0
        self.sum_pearson_gb = 0
        self.sum_loss = 0
        self.sum_depth_loss = 0
        self.sum_smooth_loss = 0
        self.sum_photometric_loss = 0
        self.sum_variance_loss = 0

    def update(self, result, gpu_time, data_time, n=1):
        self.count += n
        self.sum_irmse += n * result.irmse
        self.sum_imae += n * result.imae
        self.sum_mse += n * result.mse
        self.sum_rmse += n * result.rmse
        self.sum_mae += n * result.mae
        self.sum_absrel += n * result.absrel
        self.sum_squared_rel += n * result.squared_rel
        self.sum_rel_squared += n * result.rel_squared
        self.sum_rel_exp += n * result.rel_exp
        self.sum_diff_thresh += n * result.diff_thresh
        self.sum_rmse_3 += n * result.rmse_3
        self.sum_diff_3 += n * result.diff_3
        self.sum_rmse_6 += n * result.rmse_6
        self.sum_diff_6 += n * result.diff_6
        self.sum_lg10 += n * result.lg10
        self.sum_delta1 += n * result.delta1
        self.sum_delta2 += n * result.delta2
        self.sum_delta3 += n * result.delta3
        self.sum_data_time += n * data_time
        self.sum_gpu_time += n * gpu_time
        self.sum_silog += n * result.silog
        self.sum_avg_target += n * result.avg_target
        self.sum_avg_pred += n * result.avg_pred
        self.sum_pearson += n * result.pearson
        self.sum_pearson_gb += n * result.pearson_gb
        self.sum_loss += n * result.loss
        self.sum_depth_loss += n * result.depth_loss
        self.sum_smooth_loss += n * result.smooth_loss
        self.sum_photometric_loss += n * result.photometric_loss
        self.sum_variance_loss += n * result.variance_loss

    def average(self):
        avg = Result()
        if self.count > 0:
            avg.update(
                self.sum_irmse / self.count, self.sum_imae / self.count,
                self.sum_mse / self.count, self.sum_rmse / self.count,
                self.sum_mae / self.count, self.sum_absrel / self.count,
                self.sum_squared_rel / self.count, self.sum_rel_squared / self.count, self.sum_rel_exp / self.count,
                self.sum_diff_thresh / self.count, self.sum_rmse_3 / self.count, self.sum_diff_3 / self.count,
                self.sum_rmse_6 / self.count, self.sum_diff_6 / self.count, self.sum_lg10 / self.count,
                self.sum_delta1 / self.count, self.sum_delta2 / self.count,
                self.sum_delta3 / self.count, self.sum_gpu_time / self.count,
                self.sum_data_time / self.count, self.sum_silog / self.count,
                self.sum_avg_target / self.count, self.sum_avg_pred / self.count,
                self.sum_pearson / self.count, self.sum_pearson_gb / self.count,
                self.sum_loss / self.count, self.sum_depth_loss / self.count,
                self.sum_smooth_loss / self.count, self.sum_photometric_loss / self.count,
                self.sum_variance_loss / self.count)
        return avg
