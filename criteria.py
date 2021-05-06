import torch
import torch.nn as nn

loss_names = ['l1', 'l2', 'REL']


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = pred - target
        diff = diff[valid_mask]
        # log_var = log_var[valid_mask]
        loss = diff ** 2
        self.loss = loss.mean()
        # self.loss = ((-log_var).exp() * diff ** 2 + log_var).mean()
        return self.loss


class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target, weight=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss


class MaskedRELLoss(nn.Module):
    def __init__(self):
        super(MaskedRELLoss, self).__init__()

    def forward(self, pred, target, weight=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        # log_var = log_var[valid_mask]
        rel = diff.abs() / target[valid_mask].exp()
        # self.loss = ((-log_var).exp() * rel + log_var).mean()
        self.loss = rel.mean()
        return self.loss


class VarianceLoss(nn.Module):
    def __init__(self):
        super(VarianceLoss, self).__init__()

    def forward(self, pred, log_var, target, weight=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        log_var = log_var[valid_mask]
        self.loss = ((-log_var).exp() * diff ** 2 + log_var).mean()
        return self.loss


class PhotometricLoss(nn.Module):
    def __init__(self):
        super(PhotometricLoss, self).__init__()

    def forward(self, target, recon, mask=None):

        assert recon.dim(
        ) == 4, "expected recon dimension to be 4, but instead got {}.".format(
            recon.dim())
        assert target.dim(
        ) == 4, "expected target dimension to be 4, but instead got {}.".format(
            target.dim())
        assert recon.size() == target.size(), "expected recon and target to have the same size, but got {} and {} instead" \
            .format(recon.size(), target.size())
        diff = (target - recon).abs()
        diff = torch.sum(diff, 1)  # sum along the color channel

        # compare only pixels that are not black
        valid_mask = (torch.sum(recon, 1) > 0).float() * (torch.sum(target, 1)
                                                          > 0).float()
        if mask is not None:
            valid_mask = valid_mask * torch.squeeze(mask).float()
        valid_mask = valid_mask.byte().detach()
        if valid_mask.numel() > 0:
            diff = diff[valid_mask]
            if diff.nelement() > 0:
                self.loss = diff.mean()
            else:
                print(
                    "warning: diff.nelement()==0 in PhotometricLoss (this is expected during early stage of training, try larger batch size)."
                )
                self.loss = 0
        else:
            print("warning: 0 valid pixel in PhotometricLoss")
            self.loss = 0
        return self.loss


class SmoothnessLoss(nn.Module):
    def __init__(self):
        super(SmoothnessLoss, self).__init__()

    def forward(self, depth):
        def second_derivative(x):
            assert x.dim(
            ) == 4, "expected 4-dimensional data, but instead got {}".format(
                x.dim())
            horizontal = 2 * x[:, :, 1:-1, 1:-1] - x[:, :, 1:-1, :-2] - x[:, :, 1:-1, 2:]
            vertical = 2 * x[:, :, 1:-1, 1:-1] - x[:, :, :-2, 1:-1] - x[:, :, 2:, 1:-1]
            der_2nd = horizontal.abs() + vertical.abs()
            return der_2nd.mean()

        self.loss = second_derivative(depth)
        return self.loss


class PearsonCorrelationLoss(nn.Module):
    def __init__(self):
        super(PearsonCorrelationLoss, self).__init__()

    def forward(self, gb, pred):
        # valid_mask = (target == 0).detach()
        # pred = pred[valid_mask]
        # gb = gb[valid_mask]
        vx = pred - pred.mean()
        vy = gb - gb.mean()
        pearson = (vx * vy).sum() / ((vx ** 2).sum().sqrt() * (vy ** 2).sum().sqrt())
        self.loss = 1 - pearson
        return self.loss
