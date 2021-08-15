import argparse
import os
import time
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import criteria
import helper
import warnings
from inverse_warp import Intrinsics, homography_from
from metrics import AverageMeter, Result
from model import DepthCompletionNet


warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser(description='Sparse-to-Dense')
parser.add_argument('-w',
                    '--workers',
                    default=4,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs',
                    default=11,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run (default: 11)')
parser.add_argument('--start-epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-c',
                    '--criterion',
                    metavar='LOSS',
                    default='l2',
                    choices=criteria.loss_names,
                    help='loss function: | '.join(criteria.loss_names) +
                         ' (default: l2)')
parser.add_argument('-b',
                    '--batch-size',
                    default=1,
                    type=int,
                    help='mini-batch size (default: 1)')
parser.add_argument('--lr',
                    '--learning-rate',
                    default=1e-5,
                    type=float,
                    metavar='LR',
                    help='initial learning rate (default 1e-5)')
parser.add_argument('--weight-decay',
                    '--wd',
                    default=0,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 0)')
parser.add_argument('--print-freq',
                    '-p',
                    default=50,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 50)')
parser.add_argument('--resume',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--data-folder',
                    default='../data',
                    type=str,
                    metavar='PATH',
                    help='data folder (default: none)')
parser.add_argument('-i',
                    '--input',
                    type=str,
                    default='gd',
                    choices=['d', 'rgb', 'rgbd', 'g', 'gd'],
                    help='input: | d | rgb | rgbd | g | gd')
parser.add_argument('-l',
                    '--layers',
                    type=int,
                    default=34,
                    help='use 16 for sparse_conv; use 18 or 34 for resnet')
parser.add_argument('--pretrained',
                    action="store_true",
                    help='use ImageNet pre-trained weights')
parser.add_argument('--val',
                    type=str,
                    default="select",
                    choices=["select", "full"],
                    help='full or select validation set')
parser.add_argument('--jitter',
                    type=float,
                    default=0.1,
                    help='color jitter for images')
parser.add_argument('--rank_metric',
                    type=str,
                    default='rmse',
                    choices=[m for m in dir(Result()) if not m.startswith('_')],
                    help='metrics for which best result is sbatch_datacted')
parser.add_argument('-m',
                    '--train-mode',
                    type=str,
                    default="dense",
                    choices=["dense", "sparse", "photo", "sparse+photo", "dense+photo", "dense+corr", "sparse+corr"],
                    help='dense | sparse | photo | sparse+photo | dense+photo | dense+corr | sparse+corr')
parser.add_argument('--data',
                    metavar='DATA',
                    default="nachsholim",
                    choices=['kitti', 'D5', 'nachsholim', 'cave', 'cemetery', 'squid'],
                    help='kitti | D5 | nachsholim | cave | cemetery | squid')
parser.add_argument('--save_pred',
                    action="store_true",
                    help='save prediction images')
parser.add_argument('-e', '--evaluate', default='', type=str, metavar='PATH')
parser.add_argument('-v', '--train_var_branch', default='', type=str, metavar='PATH')
parser.add_argument('-wv', '--without_var', action="store_true", help='train model without variance branch')
parser.add_argument('--cpu', action="store_true", help='run on cpu')

args = parser.parse_args()
args.result = os.path.join('..', 'results')
args.use_pose = ("photo" in args.train_mode)
args.use_corr = ("corr" in args.train_mode)
args.use_rgb = ('rgb' in args.input) or args.use_pose
args.use_d = 'd' in args.input
args.use_g = 'g' in args.input and 'rgb' not in args.input
if args.use_pose or args.use_corr:
    args.w1, args.w2 = 0.1, 0.1
else:
    args.w1, args.w2 = 0, 0
print(args)

cuda = torch.cuda.is_available() and not args.cpu
if cuda:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("=> using '{}' for computation.".format(device))

# define loss functions
if args.criterion == 'REL':
    depth_criterion = criteria.MaskedRELLoss()
elif args.criterion == 'Tukey':
    depth_criterion = criteria.MaskedTukeyLoss()
else:
    depth_criterion = criteria.MaskedMSELoss()
var_criterion = criteria.VarianceLoss()
photometric_criterion = criteria.PhotometricLoss()
smoothness_criterion = criteria.SmoothnessLoss()
correlation_criterion = criteria.PearsonCorrelationLoss()

if args.data == 'D5':
    from dataloaders.D5_loader import load_calib, oheight, owidth
    from dataloaders.D5_loader import D5Depth as Depth
elif args.data == 'nachsholim':
    from dataloaders.nachsholim_loader import load_calib, oheight, owidth
    from dataloaders.nachsholim_loader import NachsholimDepth as Depth
elif args.data == 'squid':
    from dataloaders.squid_loader import load_calib, oheight, owidth
    from dataloaders.squid_loader import SQUIDDepth as Depth
elif args.data == 'cave':
    from dataloaders.SC_Cave_loader import load_calib, oheight, owidth
    from dataloaders.SC_Cave_loader import CaveDepth as Depth
elif args.data == 'cemetery':
    from dataloaders.SC_Cemetery_loader import load_calib, oheight, owidth
    from dataloaders.SC_Cemetery_loader import CemeteryDepth as Depth
elif args.data == 'kitti':
    from dataloaders.kitti_loader import load_calib, oheight, owidth
    from dataloaders.kitti_loader import KittiDepth as Depth

if args.use_pose:
    # hard-coded camera intrinsics
    K = load_calib()
    fu, fv = float(K[0][0]), float(K[1][1])
    cu, cv = float(K[0][2]), float(K[1][2])
    data_intrinsics = Intrinsics(owidth, oheight, fu, fv, cu, cv)
    if cuda:
        data_intrinsics = data_intrinsics.cuda()


def iterate(mode, args, loader, model, optimizer, logger, epoch):
    # torch.cuda.empty_cache()
    block_average_meter = AverageMeter()
    average_meter = AverageMeter()
    meters = [block_average_meter, average_meter]

    # switch to appropriate mode
    assert mode in ["train", "val", "eval", "test_prediction", "test_completion"], "unsupported mode: {}".format(mode)
    if mode == 'train':
        model.train()
        lr = helper.adjust_learning_rate(args.lr, optimizer, epoch)
    else:
        model.eval()
        lr = 0

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    for i, batch_data in enumerate(loader):
        start = time.time()
        batch_data = {key: val.to(device) for key, val in batch_data.items() if val is not None}
        gt = batch_data['gt'] if mode != 'test_prediction' and mode != 'test_completion' else None
        rgb = batch_data['rgb'] if args.use_rgb else None
        data_time = time.time() - start

        start = time.time()
        pred, log_var = model(batch_data)
        gpu_time = time.time() - start

        loss, depth_loss, self_loss, smooth_loss, var_loss, mask = 0, 0, 0, 0, 0, None
        if mode == 'train':
            # Loss 1: the direct depth supervision from ground truth label
            # mask=1 indicates that a pixel does not ground truth labels
            if 'sparse' in args.train_mode:
                depth_loss = depth_criterion(pred, batch_data['d'])
                mask = (batch_data['d'] < 1e-3).float()
            elif 'dense' in args.train_mode:
                depth_loss = depth_criterion(pred, gt)
                mask = (gt < 1e-3).float()

            # Loss 2: the pearson correlation loss
            if args.use_corr:
                if rgb is None:
                    raise (RuntimeError("Requested rgb images for computing correlation loss but none was found"))
                gb = torch.max(batch_data['rgb'][:, 2, :, :], batch_data['rgb'][:, 1, :, :]) - batch_data['rgb'][:, 0, :, :]
                gb = gb.unsqueeze(1)
                self_loss = correlation_criterion(gb, pred)  # , gt)
            else:
                self_loss = 0

            # Loss 3: the depth smoothness loss
            smooth_loss = smoothness_criterion(pred) if args.w2 > 0 else 0

            # Loss 4: the variance loss
            var_loss = var_criterion(pred, log_var, gt) if not args.without_var else 0

            # Loss 5: the self-supervised photometric loss
            if args.use_pose:
                # create multi-scale pyramids
                pred_array = helper.multiscale(pred)
                rgb_curr_array = helper.multiscale(batch_data['rgb'])
                rgb_near_array = helper.multiscale(batch_data['rgb_near'])
                if mask is not None:
                    mask_array = helper.multiscale(mask)
                num_scales = len(pred_array)
                # compute photometric loss at multiple scales
                for scale in range(len(pred_array)):
                    pred_ = pred_array[scale]
                    rgb_curr_ = rgb_curr_array[scale]
                    rgb_near_ = rgb_near_array[scale]
                    mask_ = None
                    if mask is not None:
                        mask_ = mask_array[scale]
                    # compute the corresponding intrinsic parameters
                    height_, width_ = pred_.size(2), pred_.size(3)
                    intrinsics_ = data_intrinsics.scale(height_, width_)
                    # inverse warp from a nearby frame to the current frame
                    warped_ = homography_from(rgb_near_, pred_,
                                              batch_data['r_mat'],
                                              batch_data['t_vec'], intrinsics_)
                    self_loss += photometric_criterion(rgb_curr_, warped_, mask_) * (2 ** (scale - num_scales))

            # backprop
            loss = depth_loss + var_loss + args.w1 * self_loss + args.w2 * smooth_loss
            optimizer.zero_grad()
            # if args.train_var:
            #     var_loss.backward()
            # else:
            loss.backward()
            optimizer.step()

        # measure accuracy and record loss
        with torch.no_grad():
            mini_batch_size = next(iter(batch_data.values())).size(0)
            result = Result()
            if mode != 'test_prediction' and mode != 'test_completion':
                # var_loss = var_criterion(pred, log_var, gt)
                if rgb is not None:
                    result.evaluate(pred.data, gt.data, rgb.data, loss, depth_loss, smooth_loss, self_loss, var_loss)
                else:
                    result.evaluate(pred.data, gt.data, None, loss, depth_loss, smooth_loss, self_loss, var_loss)
                sample_i_rmse = result.rmse
            [m.update(result, gpu_time, data_time, mini_batch_size) for m in meters]
            logger.conditional_print(mode, i, epoch, lr, len(loader), block_average_meter, average_meter)
            if args.data == 'D5':
                skip = 7
            elif args.data == 'cave' or args.data == 'cemetery':
                skip = 70
            elif args.data == 'nachsholim':
                skip = 100
            elif args.data == 'squid':
                skip = 5
            elif args.data == 'kitti':
                skip = 100
            logger.conditional_save_pred(mode, i, pred, epoch, sample_i_rmse)
            logger.conditional_save_img_comparison(mode, i, batch_data, pred, log_var, epoch, skip)
            if args.train_var_branch or args.evaluate:
                logger.conditional_save_var(mode, i, log_var, epoch)

    end_event.record()
    torch.cuda.synchronize()  # Wait for the events to be recorded!
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(elapsed_time_ms)

    avg = logger.conditional_save_info(mode, average_meter, epoch)
    is_best = logger.rank_conditional_save_best(mode, avg, epoch, args.val)
    if (is_best and not (mode == "train")) or args.val == 'full':
        logger.save_img_comparison_as_best(mode, epoch)
    logger.conditional_summarize(mode, avg, is_best)

    return avg, is_best


def main():
    global args
    checkpoint = None
    is_eval = False
    if args.evaluate:
        args_new = args
        if os.path.isfile(args.evaluate):
            print("=> loading checkpoint '{}' ... ".format(args.evaluate), end='')
            checkpoint = torch.load(args.evaluate, map_location=device)
            args = checkpoint['args']
            args.data_folder = args_new.data_folder
            args.data = args_new.data
            args.val = args_new.val
            args.evaluate = args_new.evaluate
            args.save_pred = args_new.save_pred
            args.print_freq = args_new.print_freq
            # args.train_var_branch = False
            is_eval = True
            print("Completed.")
        else:
            print("No model found at '{}'".format(args.evaluate))
            return
    elif args.train_var_branch:
        args_new = args
        if os.path.isfile(args.train_var_branch):
            print("=> loading checkpoint '{}' ... ".format(args.train_var_branch), end='')
            checkpoint = torch.load(args.train_var_branch, map_location=device)
            args = checkpoint['args']
            args.train_var_branch = args_new.train_var_branch
            args.rank_metric = args_new.rank_metric
            print("Completed.")
        else:
            print("No model found at '{}'".format(args.train_var_branch))
            return
    elif args.resume:  # optionally resume from a checkpoint
        args_new = args
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}' ... ".format(args.resume), end='')
            checkpoint = torch.load(args.resume, map_location=device)
            args.start_epoch = checkpoint['epoch'] + 1
            args.data_folder = args_new.data_folder
            args.val = args_new.val
            print("Completed. Resuming from epoch {}.".format(checkpoint['epoch']))
        else:
            print("No checkpoint found at '{}'".format(args.resume))
            return

    print("=> creating model and optimizer ... ", end='')
    model = DepthCompletionNet(args).to(device)
    if args.without_var:
        print("=> training model without variance ... ", end='')
        ct = 0
        for child in model.children():
            if ct >= 13:
                for param in child.parameters():
                    param.requires_grad = False
            ct += 1
    model_named_params = [p for _, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(model_named_params, lr=args.lr, weight_decay=args.weight_decay)
    print('trainable parameters', sum(p.numel() for p in model.parameters() if p.requires_grad))

    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
    if args.train_var_branch or args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> checkpoint state loaded.")

    if args.train_var_branch:
        print("=> training variance branch... ", end='')
        ct = 0
        for child in model.children():
            if ct < 13:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True
            ct += 1
        print("=> variance parameters updated.")
        model_named_params = [p for _, p in model.named_parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(model_named_params, lr=args.lr, weight_decay=args.weight_decay)
        print('trainable parameters', sum(p.numel() for p in model.parameters() if p.requires_grad))
        print("completed.")

    model = torch.nn.DataParallel(model)

    # Data loading code
    print("=> creating data loaders ... ")
    if not is_eval:
        train_dataset = Depth('train', args)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=args.workers,
                                                   pin_memory=True,
                                                   sampler=None)
        print("\t==> train_loader size:{}".format(len(train_loader)))
    val_dataset = Depth('val', args)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True)  # set batch size to be 1 for validation
    print("\t==> val_loader size:{}".format(len(val_loader)))

    # create backups and results folder
    logger = helper.logger(args)
    if args.resume or is_eval:
        logger.best_result = checkpoint['best_result']
    print("=> logger created.")

    if is_eval:
        print("=> starting model evaluation ...")
        iterate("val", args, val_loader, model, None, logger, checkpoint['epoch'])
        return

    # main loop
    print("=> starting main loop ...")
    for epoch in range(args.start_epoch, args.epochs):
        print("=> starting training epoch {} ..".format(epoch))
        iterate("train", args, train_loader, model, optimizer, logger, epoch)  # train for one epoch
        result, is_best = iterate("val", args, val_loader, model, None, logger, epoch)  # evaluate on validation set
        helper.save_checkpoint({  # save checkpoint
            'epoch': epoch,
            'model': model.module.state_dict(),
            'best_result': logger.best_result,
            'optimizer': optimizer.state_dict(),
            'args': args,
        }, is_best, epoch, logger.output_directory)


if __name__ == '__main__':
    main()
