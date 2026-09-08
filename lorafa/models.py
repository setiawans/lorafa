import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

ARCHS = {
    "resnet20_4": {"num_blocks": (3, 3, 3), "widths": (64, 128, 256)},
    "resnet18": {"num_blocks": (2, 2, 2, 2), "widths": (64, 128, 256, 512)},
}

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))

class ResNetCIFAR(nn.Module):
    def __init__(self, num_blocks, widths, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, widths[0], 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(widths[0])
        self.in_planes = widths[0]
        stages = []
        for i, (n, w) in enumerate(zip(num_blocks, widths)):
            stages.append(self._make_stage(w, n, stride=1 if i == 0 else 2))
        self.stages = nn.Sequential(*stages)
        self.fc = nn.Linear(widths[-1], num_classes)
        self._init_weights()

    def _make_stage(self, planes, n, stride):
        blocks = []
        for s in [stride] + [1] * (n - 1):
            blocks.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*blocks)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.stages(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)

def build_model(arch: str, num_classes: int) -> ResNetCIFAR:
    if arch not in ARCHS:
        raise ValueError(f"unknown arch {arch!r}, choose from {list(ARCHS)}")
    return ResNetCIFAR(num_classes=num_classes, **ARCHS[arch])

def replace_head(model: ResNetCIFAR, num_classes: int) -> ResNetCIFAR:
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

class LoRAConv2d(nn.Module):
    def __init__(self, base: nn.Conv2d, rank: int, alpha: float, generator=None):
        super().__init__()
        self.stride, self.padding = base.stride, base.padding
        self.dilation, self.groups = base.dilation, base.groups
        self.rank, self.alpha = rank, alpha
        self.scaling = alpha / rank

        self.weight0 = nn.Parameter(base.weight.detach().clone(), requires_grad=False)
        self.bias0 = None
        if base.bias is not None:
            self.bias0 = nn.Parameter(base.bias.detach().clone(), requires_grad=False)

        out_ch, in_ch_g, kh, kw = base.weight.shape
        self.d = in_ch_g * kh * kw
        self.A = nn.Parameter(torch.empty(rank, self.d), requires_grad=False)
        self.B = nn.Parameter(torch.zeros(out_ch, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5), generator=generator)

    def delta_weight(self) -> torch.Tensor:
        return (self.scaling * self.B @ self.A).view_as(self.weight0)

    def forward(self, x):
        return F.conv2d(
            x, self.weight0 + self.delta_weight(), self.bias0,
            self.stride, self.padding, self.dilation, self.groups,
        )

def attach_lora(
    model: nn.Module,
    rank: int,
    alpha: float,
    adapt_first_conv: bool = False,
    seed: int = 0,
) -> nn.Module:
    gen = torch.Generator().manual_seed(seed)
    targets = [
        name for name, m in model.named_modules()
        if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3)
    ]
    if not adapt_first_conv:
        targets = targets[1:]

    for name in targets:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, LoRAConv2d(getattr(parent, child), rank, alpha, generator=gen))

    for p in model.parameters():
        p.requires_grad_(False)
    for m in lora_layers(model):
        m.B.requires_grad_(True)
    for p in model.fc.parameters():
        p.requires_grad_(True)
    return model

def lora_layers(model: nn.Module) -> List[LoRAConv2d]:
    return [m for m in model.modules() if isinstance(m, LoRAConv2d)]

def lora_B_params(model: nn.Module) -> List[nn.Parameter]:
    return [m.B for m in lora_layers(model)]

def lora_A_params(model: nn.Module) -> List[nn.Parameter]:
    return [m.A for m in lora_layers(model)]

def lora_W0_params(model: nn.Module) -> List[nn.Parameter]:
    return [m.weight0 for m in lora_layers(model)]

def count_params(params) -> int:
    return sum(p.numel() for p in params)