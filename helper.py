import os
import time
import shutil
import torch
import csv
import vis_utils
from metrics import Result

fieldnames = [
    'epoch', 'rmse', 'mae', 'loss', 'depth_loss', 'smooth_loss', 'photometric_loss', 'variance_loss', 'irmse', 'imae', 'mse', 'absrel',
    'rel_exp', 'diff_thresh', 'rmse_3', 'diff_3', 'rmse_6', 'diff_6', 'lg10', 'silog', 'squared_rel', 'rel_squared',
    'delta1', 'delta2', 'delta3', 'data_time', 'gpu_time', 'avg_target', 'avg_pred', 'pearson', 'pearson_gb'
]


class logger:
    def __init__(self, args, prepare=True):
        self.args = args
        output_directory = get_folder_name(args)
        self.output_directory = output_directory
        self.best_result = Result()
        self.best_result.set_to_worst()

        if not prepare:
            return
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        self.train_csv = os.path.join(output_directory, 'train.csv')
        self.val_csv = os.path.join(output_directory, 'val.csv')
        self.best_txt = os.path.join(output_directory, 'best.txt')

        # backup the source code
        if args.resume == '':
            print("=> creating source code backup ...")
            backup_directory = os.path.join(output_directory, "code_backup")
            self.backup_directory = backup_directory
            backup_source_code(backup_directory)
            # create new csv files with only header
            with open(self.train_csv, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
            with open(self.val_csv, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
            print("=> finished creating source code backup.")

    def conditional_print(self, split, i, epoch, lr, n_set, blk_avg_meter,
                          avg_meter):
        if (i + 1) % self.args.print_freq == 0:
            avg = avg_meter.average()
            blk_avg = blk_avg_meter.average()
            print('=> output: {}'.format(self.output_directory))
            print(
                '{split} Epoch: {0} [{1}/{2}]\tlr={lr} '
                't_Data={blk_avg.data_time:.3f}({average.data_time:.3f}) '
                't_GPU={blk_avg.gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                'RMSE={blk_avg.rmse:.2f}({average.rmse:.2f}) '
                'MAE={blk_avg.mae:.2f}({average.mae:.2f}) '
                'iRMSE={blk_avg.irmse:.2f}({average.irmse:.2f}) '
                'iMAE={blk_avg.imae:.2f}({average.imae:.2f})\n\t'
                'silog={blk_avg.silog:.2f}({average.silog:.2f}) '
                'squared_rel={blk_avg.squared_rel:.2f}({average.squared_rel:.2f}) '
                'rel_squared={blk_avg.rel_squared:.2f}({average.rel_squared:.2f}) '
                'Delta1={blk_avg.delta1:.3f}({average.delta1:.3f})\n\t'
                'REL={blk_avg.absrel:.3f}({average.absrel:.3f}) '
                'rel_exp={blk_avg.rel_exp:.3f}({average.rel_exp:.3f}) '
                'diff_thresh={blk_avg.diff_thresh:.3f}({average.diff_thresh:.3f})\n\t'
                'rmse_3={blk_avg.rmse_3:.3f}({average.rmse_3:.3f}) '
                'diff_3={blk_avg.diff_3:.3f}({average.diff_3:.3f}) '
                'rmse_6={blk_avg.rmse_6:.3f}({average.rmse_6:.3f}) '
                'diff_6={blk_avg.diff_6:.3f}({average.diff_6:.3f})\n\t'
                'Lg10={blk_avg.lg10:.3f}({average.lg10:.3f}) '
                'Avg_target={blk_avg.avg_target:.3f}({average.avg_target:.3f}) '
                'Avg_pred={blk_avg.avg_pred:.3f}({average.avg_pred:.3f})\n\t'
                'Pearson={blk_avg.pearson:.3f}({average.pearson:.3f}) '
                'Pearson_gb={blk_avg.pearson_gb:.3f}({average.pearson_gb:.3f})\n\t'
                'Loss={blk_avg.loss:.3f}({average.loss:.3f}) '
                'Depth_loss={blk_avg.depth_loss:.3f}({average.depth_loss:.3f}) '
                'Smooth_loss={blk_avg.smooth_loss:.3f}({average.smooth_loss:.3f}) '
                'Photometric_loss={blk_avg.photometric_loss:.3f}({average.photometric_loss:.3f}) '
                'Variance_loss={blk_avg.variance_loss:.3f}({average.variance_loss:.3f}) '
                .format(epoch,
                        i + 1,
                        n_set,
                        lr=lr,
                        blk_avg=blk_avg,
                        average=avg,
                        split=split.capitalize()))
            blk_avg_meter.reset()

    def conditional_save_info(self, split, average_meter, epoch):
        avg = average_meter.average()
        if split == "train":
            csvfile_name = self.train_csv
        elif split == "val":
            csvfile_name = self.val_csv
        elif split == "eval":
            eval_filename = os.path.join(self.output_directory, 'eval.txt')
            self.save_single_txt(eval_filename, avg, epoch)
            return avg
        elif "test" in split:
            return avg
        else:
            raise ValueError("wrong split provided to logger")
        with open(csvfile_name, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({
                'epoch': epoch,
                'rmse': avg.rmse,
                'mae': avg.mae,
                'loss': avg.loss,
                'depth_loss': avg.depth_loss,
                'smooth_loss': avg.smooth_loss,
                'photometric_loss': avg.photometric_loss,
                'variance_loss': avg.variance_loss,
                'irmse': avg.irmse,
                'imae': avg.imae,
                'mse': avg.mse,
                'silog': avg.silog,
                'squared_rel': avg.squared_rel,
                'rel_squared': avg.rel_squared,
                'absrel': avg.absrel,
                'rel_exp': avg.rel_exp,
                'diff_thresh': avg.diff_thresh,
                'rmse_3': avg.rmse_3,
                'diff_3': avg.diff_3,
                'rmse_6': avg.rmse_6,
                'diff_6': avg.diff_6,
                'lg10': avg.lg10,
                'delta1': avg.delta1,
                'delta2': avg.delta2,
                'delta3': avg.delta3,
                'gpu_time': avg.gpu_time,
                'data_time': avg.data_time,
                'avg_target': avg.avg_target,
                'avg_pred': avg.avg_pred,
                'pearson': avg.pearson,
                'pearson_gb': avg.pearson_gb
            })
        return avg

    def save_single_txt(self, filename, result, epoch):
        with open(filename, 'w') as txtfile:
            txtfile.write(
                ("rank_metric={}\n" + "epoch={}\n" + "rmse={:.3f}\n" +
                 "mae={:.3f}\n" + "silog={:.3f}\n" + "squared_rel={:.3f}\n" + "rel_squared={:.3f}\n" +
                 "irmse={:.3f}\n" + "imae={:.3f}\n" + "mse={:.3f}\n" +
                 "absrel={:.3f}\n" + "rel_exp={:.3f}\n" + "diff_thresh={:.3f}\n" + "rmse_3={:.3f}\n" +
                 "diff_3={:.3f}\n" + "rmse_6={:.3f}\n" + "diff_6={:.3f}\n" + "lg10={:.3f}\n" + "delta1={:.3f}\n" +
                 "t_gpu={:.4f}\n" + "avg_depth_tar={:.3f}\n" + "avg_depth_pred={:.3f}\n"
                 "pearson={:.3f}\n" + "pearson_gb={:.3f}").format(self.args.rank_metric, epoch,
                                                                  result.rmse, result.mae, result.silog,
                                                                  result.squared_rel, result.rel_squared, result.irmse,
                                                                  result.imae, result.mse, result.absrel,
                                                                  result.rel_exp, result.diff_thresh, result.rmse_3,
                                                                  result.diff_3, result.rmse_6, result.diff_6,
                                                                  result.lg10, result.delta1, result.gpu_time,
                                                                  result.avg_target, result.avg_pred,
                                                                  result.pearson, result.pearson_gb))

    def save_best_txt(self, result, epoch):
        self.save_single_txt(self.best_txt, result, epoch)

    def _get_img_comparison_name(self, mode, epoch, is_best=False):
        if mode == 'eval':
            return self.output_directory + '/comparison_eval.png'
        if mode == 'val':
            if is_best:
                return self.output_directory + '/comparison_' + str(epoch) + '_best.png'
            else:
                return self.output_directory + '/comparison_' + str(epoch) + '.png'

    def conditional_save_img_comparison(self, mode, i, ele, pred, log_var, epoch, skip=100):
        # save 8 images for visualization
        if mode == 'val' or mode == 'eval':
            if i == 0:
                # self.img_merge = vis_utils.merge_into_col(ele, pred)
                self.img_merge = vis_utils.merge_into_col(ele, pred, log_var)
            elif i % skip == 0 and i < 4 * skip:
                # row = vis_utils.merge_into_col(ele, pred)
                row = vis_utils.merge_into_col(ele, pred, log_var)
                self.img_merge = vis_utils.add_col(self.img_merge, row)
            elif i == 4 * skip:
                filename = self._get_img_comparison_name(mode, epoch)
                # vis_utils.save_image(self.img_merge, filename)

    def save_img_comparison_as_best(self, mode, epoch):
        if mode == 'val':
            filename = self._get_img_comparison_name(mode, epoch, is_best=True)
            vis_utils.save_image(self.img_merge, filename)
            return self.img_merge

    def get_ranking_error(self, result):
        return getattr(result, self.args.rank_metric)

    def rank_conditional_save_best(self, mode, result, epoch, val):
        error = self.get_ranking_error(result)
        best_error = self.get_ranking_error(self.best_result)
        is_best = error < best_error
        if (is_best and mode == "val") or val == "full":
            self.old_best_result = self.best_result
            self.best_result = result
            self.save_best_txt(result, epoch)
        return is_best

    def conditional_save_pred(self, mode, i, pred, epoch, sample_i_rmse):
        if ("test" in mode or mode == "eval") or (mode == 'val' and self.args.save_pred):
            # save images for visualization/ testing
            image_folder = os.path.join(self.output_directory, mode + "_output")
            if not os.path.exists(image_folder):
                os.makedirs(image_folder)
            img = torch.squeeze(pred.data.cpu()).numpy()
            filename = os.path.join(image_folder, 'epoch_' + str(epoch) + '_{0:010d}'.format(i) +
                                    '_rmse_{:.3f}.png'.format(sample_i_rmse))
            vis_utils.save_depth_as_uint16png(img, filename)
            # img_color = vis_utils.depth_colorize(img)
            # image_name_color = os.path.join(image_folder, 'colorized' + '{0:010d}.png'.format(i))
            # vis_utils.save_image(img_color, image_name_color)

    def conditional_save_var(self, mode, i, var, epoch):
        if ("test" in mode or mode == "eval") or (mode == 'val' and self.args.save_pred):
            # save images for visualization/ testing
            image_folder = os.path.join(self.output_directory, mode + "_output_var")
            if not os.path.exists(image_folder):
                os.makedirs(image_folder)
            img = torch.squeeze(var.data.cpu()).numpy()
            filename = os.path.join(image_folder, 'epoch_' + str(epoch) + '_{0:010d}.tif'.format(i))
            vis_utils.save_var_image(img, filename)

    def conditional_summarize(self, mode, avg, is_best):
        print("\n*\nSummary of ", mode, "round")
        print(''
              'RMSE={average.rmse:.3f}\n'
              'MAE={average.mae:.3f}\n'
              'Loss={average.loss:.3f}\n'
              'Depth_loss={average.depth_loss:.3f}\n'
              'Smooth_loss={average.smooth_loss:.3f}\n'
              'Photometric_loss={average.photometric_loss:.3f}\n'
              'Variance_loss={average.variance_loss:.3f}\n'
              'iRMSE={average.irmse:.3f}\n'
              'iMAE={average.imae:.3f}\n'
              'squared_rel={average.squared_rel}\n'
              'rel_squared={average.rel_squared}\n'
              'silog={average.silog}\n'
              'Delta1={average.delta1:.3f}\n'
              'REL={average.absrel:.3f}\n'
              'rel_exp={average.rel_exp:.3f}\n'
              'diff_thresh={average.diff_thresh:.3f}\n'
              'rmse_3={average.rmse_3:.3f}\n'
              'diff_3={average.diff_3:.3f}\n'
              'rmse_6={average.rmse_6:.3f}\n'
              'diff_6={average.diff_6:.3f}\n'
              'Lg10={average.lg10:.3f}\n'
              't_GPU={time:.3f}\n'
              'AVG_target={average.avg_target:.3f}\n'
              'AVG_pred={average.avg_pred:.3f}\n'
              'Pearson={average.pearson:.3f}\n'
              'Pearson_gb={average.pearson_gb:.3f}\n'.format(average=avg, time=avg.gpu_time))
        if is_best and mode == "val":
            print("New best model by %s (was %.3f)" %
                  (self.args.rank_metric,
                   self.get_ranking_error(self.old_best_result)))
        elif mode == "val":
            print("(best %s is %.3f)" %
                  (self.args.rank_metric,
                   self.get_ranking_error(self.best_result)))
        print("*\n")


ignore_hidden = shutil.ignore_patterns(".", "..", ".git*", "*pycache*",
                                       "*build", "*.fuse*", "*_drive_*")


def backup_source_code(backup_directory):
    if os.path.exists(backup_directory):
        shutil.rmtree(backup_directory)
    shutil.copytree('.', backup_directory, ignore=ignore_hidden)


def adjust_learning_rate(lr_init, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 5 epochs"""
    lr = lr_init * (0.1 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(state, is_best, epoch, output_directory):
    checkpoint_filename = os.path.join(output_directory,
                                       'checkpoint-' + str(epoch) + '.pth.tar')
    torch.save(state, checkpoint_filename)
    if is_best:
        best_filename = os.path.join(output_directory, 'model_best.pth.tar')
        shutil.copyfile(checkpoint_filename, best_filename)
    if epoch > 0:
        prev_checkpoint_filename = os.path.join(
            output_directory, 'checkpoint-' + str(epoch - 1) + '.pth.tar')
        if os.path.exists(prev_checkpoint_filename):
            os.remove(prev_checkpoint_filename)


def get_folder_name(args):
    current_time = time.strftime('%Y-%m-%d@%H-%M')
    if args.use_pose or args.use_corr:
        prefix = "mode={}.w1={}.w2={}.".format(args.train_mode, args.w1, args.w2)
    else:
        prefix = "mode={}.".format(args.train_mode)
    return os.path.join(args.result, prefix +
                        'data={}.input={}.resnet{}.epochs{}.criterion={}.lr={}.bs={}.wd={}.pretrained={}.jitter={'
                        '}.rank_metric={}.time={}'.
                        format(args.data, args.input, args.layers, args.epochs, args.criterion,
                               args.lr, args.batch_size, args.weight_decay,
                               args.pretrained, args.jitter, args.rank_metric, current_time))


avgpool = torch.nn.AvgPool2d(kernel_size=2, stride=2).cuda()


def multiscale(img):
    img1 = avgpool(img)
    img2 = avgpool(img1)
    img3 = avgpool(img2)
    img4 = avgpool(img3)
    img5 = avgpool(img4)
    return img5, img4, img3, img2, img1
