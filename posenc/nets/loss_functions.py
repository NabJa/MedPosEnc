import torch
from einops.layers.torch import Rearrange
from monai.losses.ssim_loss import SSIMLoss


class VideoSSIMLoss(SSIMLoss):
    def __init__(self, data_range=1.0, win_size=16):
        super().__init__(spatial_dims=2, data_range=data_range, win_size=win_size)

        self.stack_video_frames = Rearrange("b f c h w -> (b f) c h w")

    def forward(self, pred, target):

        pred = self.stack_video_frames(pred)
        target = self.stack_video_frames(target)

        ssim_value = self.ssim_metric._compute_tensor(pred, target).view(-1, 1)
        loss: torch.Tensor = 1 - ssim_value

        return torch.mean(loss)  # the batch/frame average
