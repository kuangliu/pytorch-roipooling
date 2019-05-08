import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops

x = torch.randn(1, 1, 10, 10) * 10
print(x)
y1 = F.adaptive_max_pool2d(x, (5, 5))
print(y1)

roi_pool = ops.RoIPool(output_size=(5, 5), spatial_scale=1)
rois = torch.tensor([[0, 0, 0, 9, 9]], dtype=torch.float)
y2 = roi_pool(x, rois)
print(y2)

rois = [torch.tensor([[0, 0, 9, 9]], dtype=torch.float)]
y3 = roi_pool(x, rois)
print(y3)
