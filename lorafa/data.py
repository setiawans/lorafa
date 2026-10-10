from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

DATASETS = {
    "cifar10": {"cls": torchvision.datasets.CIFAR10, "num_classes": 10,
                "mean": (0.4914, 0.4822, 0.4465), "std": (0.2470, 0.2435, 0.2616)},
    "cifar100": {"cls": torchvision.datasets.CIFAR100, "num_classes": 100,
                 "mean": (0.5071, 0.4865, 0.4409), "std": (0.2673, 0.2564, 0.2762)},
}

def num_classes(name: str) -> int:
    return DATASETS[name]["num_classes"]

@dataclass(frozen=True)
class Normalizer:
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]

    def _tensors(self, ref: torch.Tensor):
        mean = torch.tensor(self.mean, device=ref.device, dtype=ref.dtype).view(1, 3, 1, 1)
        std = torch.tensor(self.std, device=ref.device, dtype=ref.dtype).view(1, 3, 1, 1)
        return mean, std

    def normalize(self, x_raw: torch.Tensor) -> torch.Tensor:
        mean, std = self._tensors(x_raw)
        return (x_raw - mean) / std

    def denormalize(self, x_norm: torch.Tensor) -> torch.Tensor:
        mean, std = self._tensors(x_norm)
        return x_norm * std + mean

    def bounds(self, ref: torch.Tensor):
        mean, std = self._tensors(ref)
        return (0.0 - mean) / std, (1.0 - mean) / std

    def transform(self) -> T.Normalize:
        return T.Normalize(self.mean, self.std)

    @classmethod
    def for_dataset(cls, name: str) -> "Normalizer":
        return cls(DATASETS[name]["mean"], DATASETS[name]["std"])

    @classmethod
    def from_meta(cls, meta: dict) -> "Normalizer":
        return cls(tuple(meta["norm_mean"]), tuple(meta["norm_std"]))

def get_dataset(name: str, split: str, root: str, norm: Normalizer, augment: bool = False) -> Dataset:
    tf = [T.ToTensor(), norm.transform()]
    if augment:
        tf = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()] + tf
    return DATASETS[name]["cls"](root, train=(split == "train"), download=True, transform=T.Compose(tf))

def get_loader(dataset: Dataset, batch_size: int, shuffle: bool, workers: int, seed: int = 0) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
        pin_memory=torch.cuda.is_available(), drop_last=shuffle, generator=g,
        persistent_workers=workers > 0,
    )

def target_indices(n: int, offset: int = 0) -> List[int]:
    return list(range(offset, offset + n))

def load_targets(dataset: Dataset, indices: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    xs, ys = zip(*(dataset[i] for i in indices))
    return torch.stack(xs), torch.tensor(ys)

@torch.no_grad()
def dataset_mean_image(name: str, root: str) -> torch.Tensor:
    ds = DATASETS[name]["cls"](root, train=True, download=True, transform=T.ToTensor())
    total = torch.zeros(3, 32, 32)
    for x, _ in DataLoader(ds, batch_size=512, num_workers=0):
        total += x.sum(0)
    return (total / len(ds)).unsqueeze(0)