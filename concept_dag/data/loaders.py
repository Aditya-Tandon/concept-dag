"""
Dataset loaders for continual learning experiments.

Supported datasets:
  - SyntheticComposition:  Toy dataset for Phase 1 / Experiment 1a (no download needed).
                           Two parent feature generators (shape + colour), combined tasks.
  - SplitCIFAR10:          CIFAR-10 split into 5 tasks of 2 classes each. Requires download.
  - SplitCIFAR100:         CIFAR-100 split into 20 tasks of 5 classes each. Requires download.

NOTE: Real dataset loaders will raise DatasetNotDownloadedError if the data is not present.
      Call loader.download() with explicit user permission before using them.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from typing import List, Optional, Tuple, Dict
import numpy as np


# ---------------------------------------------------------------------------
# Custom exception for explicit download gating
# ---------------------------------------------------------------------------

class DatasetNotDownloadedError(RuntimeError):
    def __init__(self, dataset_name: str, data_root: str):
        super().__init__(
            f"Dataset '{dataset_name}' not found at '{data_root}'. "
            f"Call the download function with explicit permission first."
        )


# ---------------------------------------------------------------------------
# Synthetic Composition Dataset (no download needed)
# ---------------------------------------------------------------------------

class SyntheticCompositionDataset(Dataset):
    """
    Toy dataset for Phase 1 (Experiment 1a) — no real data needed.

    Generates samples that have two independent feature components:
      - "shape" features: drawn from Gaussian clusters (n_shape_classes)
      - "colour" features: drawn from Gaussian clusters (n_colour_classes)

    A "composition task" requires combining both components to classify.
    This mimics the scenario where a child concept must aggregate two parent concepts.

    Each sample: x ∈ R^(shape_dim + colour_dim), y ∈ {0, ..., n_shape_classes * n_colour_classes - 1}

    Args:
        n_shape_classes:  Number of shape categories (default 4).
        n_colour_classes: Number of colour categories (default 4).
        shape_dim:        Dimensionality of the shape feature space (default 32).
        colour_dim:       Dimensionality of the colour feature space (default 32).
        n_samples:        Total number of samples.
        noise:            Gaussian noise std added to features.
        seed:             Random seed.
        task:             "shape" / "colour" / "composition"
    """

    def __init__(
        self,
        n_shape_classes: int = 4,
        n_colour_classes: int = 4,
        shape_dim: int = 32,
        colour_dim: int = 32,
        n_samples: int = 5000,
        noise: float = 0.3,
        seed: int = 42,
        task: str = "composition",
    ):
        assert task in ("shape", "colour", "composition")
        self.task = task
        self.shape_dim = shape_dim
        self.colour_dim = colour_dim
        self.in_dim = shape_dim + colour_dim

        rng = np.random.default_rng(seed)

        # Create class centres in each feature space
        shape_centres = rng.standard_normal((n_shape_classes, shape_dim))
        colour_centres = rng.standard_normal((n_colour_classes, colour_dim))

        shape_labels = rng.integers(0, n_shape_classes, size=n_samples)
        colour_labels = rng.integers(0, n_colour_classes, size=n_samples)

        shape_feats = shape_centres[shape_labels] + rng.standard_normal((n_samples, shape_dim)) * noise
        colour_feats = colour_centres[colour_labels] + rng.standard_normal((n_samples, colour_dim)) * noise

        self.X = torch.tensor(
            np.concatenate([shape_feats, colour_feats], axis=1), dtype=torch.float32
        )

        if task == "shape":
            self.Y = torch.tensor(shape_labels, dtype=torch.long)
            self.n_classes = n_shape_classes
        elif task == "colour":
            self.Y = torch.tensor(colour_labels, dtype=torch.long)
            self.n_classes = n_colour_classes
        else:  # composition: label = (shape_class, colour_class) pair
            comp_labels = shape_labels * n_colour_classes + colour_labels
            self.Y = torch.tensor(comp_labels, dtype=torch.long)
            self.n_classes = n_shape_classes * n_colour_classes

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]


def make_synthetic_loaders(
    task: str = "composition",
    n_shape_classes: int = 4,
    n_colour_classes: int = 4,
    shape_dim: int = 32,
    colour_dim: int = 32,
    n_samples: int = 6000,
    batch_size: int = 128,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Create train/val/test loaders for the SyntheticCompositionDataset.
    Returns (train_loader, val_loader, test_loader, n_classes).
    """
    ds = SyntheticCompositionDataset(
        n_shape_classes=n_shape_classes,
        n_colour_classes=n_colour_classes,
        shape_dim=shape_dim,
        colour_dim=colour_dim,
        n_samples=n_samples,
        task=task,
        seed=seed,
    )
    n = len(ds)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    n_train = n - n_val - n_test

    train_ds, val_ds, test_ds = random_split(
        ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed),
    )

    def make_loader(d, shuffle):
        return DataLoader(d, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=False)

    return (
        make_loader(train_ds, shuffle=True),
        make_loader(val_ds, shuffle=False),
        make_loader(test_ds, shuffle=False),
        ds.n_classes,
    )


# ---------------------------------------------------------------------------
# Split-CIFAR-10 (requires download)
# ---------------------------------------------------------------------------
# Robust CIFAR-10 downloader (resume-capable, no torchvision dependency)
# ---------------------------------------------------------------------------

def _ensure_cifar10(data_root: str, max_attempts: int = 10):
    """
    Ensure CIFAR-10 is downloaded and extracted at data_root.

    Bypasses torchvision's downloader entirely. Uses HTTP Range requests
    to resume interrupted downloads, so it's safe to kill and restart.
    Idempotent: does nothing if the extracted directory already exists.
    """
    import pathlib, tarfile, urllib.request, time, sys

    root      = pathlib.Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    extracted = root / "cifar-10-batches-py"
    archive   = root / "cifar-10-python.tar.gz"
    url       = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    expected_size = 170_498_071  # bytes (official .tar.gz size)

    # 1. Already fully extracted — nothing to do.
    if (extracted / "batches.meta").exists():
        return

    # 2. Download with resume support.
    for attempt in range(1, max_attempts + 1):
        current = archive.stat().st_size if archive.exists() else 0
        if current >= expected_size:
            break                        # archive is complete

        headers = {"Range": f"bytes={current}-"} if current > 0 else {}
        mode    = "ab" if current > 0 else "wb"
        pct     = f"{100 * current / expected_size:.1f}%"

        if current > 0:
            print(f"  Resuming CIFAR-10 download from {pct} (attempt {attempt}/{max_attempts})...")
        else:
            print(f"  Downloading CIFAR-10 (attempt {attempt}/{max_attempts})...")

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp, \
                 open(archive, mode) as f:
                downloaded = current
                while True:
                    chunk = resp.read(1 << 16)   # 64 KB chunks
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    done = 100 * downloaded / expected_size
                    print(f"\r  {done:5.1f}%  ({downloaded // 1_048_576} MB / "
                          f"{expected_size // 1_048_576} MB)", end="", flush=True)
            print()  # newline after progress
        except Exception as exc:
            print(f"\n  Interrupted: {exc}. Retrying in 3s...")
            time.sleep(3)

    if not archive.exists() or archive.stat().st_size < expected_size * 0.99:
        raise RuntimeError(
            f"CIFAR-10 download failed after {max_attempts} attempts.\n"
            f"Please download manually:\n"
            f"  curl -C - -o {archive} {url}\n"
            f"Then re-run the experiment."
        )

    # 3. Extract.
    print("  Extracting CIFAR-10...")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(root)
    print("  CIFAR-10 ready.")


# ---------------------------------------------------------------------------
# Module-level remap helper — must be at top level so DataLoader workers
# can pickle it (local classes inside functions are not picklable).
# ---------------------------------------------------------------------------

class _RemapLabels(Dataset):
    """Wraps a Subset and remaps integer labels through a dict."""
    def __init__(self, subset, lmap: dict):
        self.subset = subset
        self.lmap = lmap

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, i):
        x, y = self.subset[i]
        return x, self.lmap[int(y)]


# ---------------------------------------------------------------------------

def make_split_cifar10(
    data_root: str = "./data",
    n_tasks: int = 5,
    batch_size: int = 128,
    val_frac: float = 0.1,
    seed: int = 42,
) -> List[Dict]:
    """
    Split CIFAR-10 into n_tasks tasks of (10 // n_tasks) classes each.
    Requires torchvision and data already downloaded.

    Returns a list of task dicts:
        [{"task_id": int, "train": DataLoader, "val": DataLoader, "test": DataLoader,
          "n_classes": int, "class_ids": List[int], "name": str}, ...]
    """
    try:
        import torchvision.transforms as T
        import torchvision.datasets as datasets
    except ImportError:
        raise ImportError("torchvision is required. Install with: pip install torchvision --break-system-packages")

    # Ensure the data is on disk before letting torchvision touch it.
    # Uses a resumable downloader — safe to kill and restart at any point.
    _ensure_cifar10(data_root)

    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    try:
        train_full = datasets.CIFAR10(data_root, train=True,  download=False, transform=transform_train)
        test_full  = datasets.CIFAR10(data_root, train=False, download=False, transform=transform_test)
    except Exception as e:
        raise DatasetNotDownloadedError("CIFAR-10", data_root) from e

    classes_per_task = 10 // n_tasks
    tasks = []

    for t in range(n_tasks):
        task_classes = list(range(t * classes_per_task, (t + 1) * classes_per_task))
        label_map = {c: i for i, c in enumerate(task_classes)}

        # Filter by class
        train_idx = [i for i, (_, y) in enumerate(train_full) if y in task_classes]
        test_idx  = [i for i, (_, y) in enumerate(test_full)  if y in task_classes]

        # Val split from train
        rng = np.random.default_rng(seed + t)
        rng.shuffle(train_idx)
        n_val = int(len(train_idx) * val_frac)
        val_idx = train_idx[:n_val]
        tr_idx  = train_idx[n_val:]

        train_ds = _RemapLabels(Subset(train_full, tr_idx), label_map)
        val_ds   = _RemapLabels(Subset(train_full, val_idx), label_map)
        test_ds  = _RemapLabels(Subset(test_full, test_idx), label_map)

        tasks.append({
            "task_id": t,
            "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                num_workers=2, persistent_workers=True, pin_memory=False),
            "val":   DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                                num_workers=2, persistent_workers=True, pin_memory=False),
            "test":  DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                                num_workers=2, persistent_workers=True, pin_memory=False),
            "n_classes": classes_per_task,
            "class_ids": task_classes,
            "name": f"CIFAR10_task{t}_{task_classes}",
        })

    return tasks


# ---------------------------------------------------------------------------
# Split-CIFAR-100 (requires download)
# ---------------------------------------------------------------------------


def _ensure_cifar100(data_root: str, max_attempts: int = 10):
    """Ensure CIFAR-100 is downloaded and extracted. Resume-capable, idempotent."""
    import pathlib, tarfile, urllib.request, time

    root      = pathlib.Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    extracted = root / "cifar-100-python"
    archive   = root / "cifar-100-python.tar.gz"
    url       = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    expected_size = 169_001_437  # bytes

    if (extracted / "meta").exists():
        return

    for attempt in range(1, max_attempts + 1):
        current = archive.stat().st_size if archive.exists() else 0
        if current >= expected_size:
            break
        headers = {"Range": f"bytes={current}-"} if current > 0 else {}
        mode    = "ab" if current > 0 else "wb"
        pct     = f"{100 * current / expected_size:.1f}%"
        if current > 0:
            print(f"  Resuming CIFAR-100 download from {pct} (attempt {attempt}/{max_attempts})...")
        else:
            print(f"  Downloading CIFAR-100 (attempt {attempt}/{max_attempts})...")
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp, open(archive, mode) as f:
                downloaded = current
                while True:
                    chunk = resp.read(1 << 16)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    done = 100 * downloaded / expected_size
                    print(f"\r  {done:5.1f}%  ({downloaded // 1_048_576} MB / "
                          f"{expected_size // 1_048_576} MB)", end="", flush=True)
            print()
        except Exception as exc:
            print(f"\n  Interrupted: {exc}. Retrying in 3s...")
            time.sleep(3)

    if not archive.exists() or archive.stat().st_size < expected_size * 0.99:
        raise RuntimeError(
            f"CIFAR-100 download failed after {max_attempts} attempts.\n"
            f"Please download manually:\n"
            f"  curl -C - -o {archive} {url}\n"
            f"Then re-run the experiment."
        )
    print("  Extracting CIFAR-100...")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(root)
    print("  CIFAR-100 ready.")


def make_split_cifar100(
    data_root: str = "./data",
    n_tasks: int = 20,
    batch_size: int = 128,
    val_frac: float = 0.1,
    seed: int = 42,
) -> List[Dict]:
    """
    Split CIFAR-100 into n_tasks tasks of (100 // n_tasks) classes each.

    Returns a list of task dicts with the same schema as make_split_cifar10:
        [{"task_id", "train", "val", "test", "n_classes", "class_ids"}, ...]
    """
    try:
        import torchvision.transforms as T
        import torchvision.datasets as datasets
    except ImportError:
        raise ImportError("torchvision is required.")

    _ensure_cifar100(data_root)

    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),
        T.ToTensor(),
        T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    try:
        train_full = datasets.CIFAR100(data_root, train=True,  download=False, transform=transform_train)
        test_full  = datasets.CIFAR100(data_root, train=False, download=False, transform=transform_test)
    except Exception as e:
        raise DatasetNotDownloadedError("CIFAR-100", data_root) from e

    # Use numpy for fast class-based indexing (avoids iterating 60k samples per task)
    train_targets = np.array(train_full.targets)
    test_targets  = np.array(test_full.targets)

    classes_per_task = 100 // n_tasks
    tasks = []

    for t in range(n_tasks):
        task_classes = list(range(t * classes_per_task, (t + 1) * classes_per_task))
        label_map    = {c: i for i, c in enumerate(task_classes)}

        train_idx = np.where(np.isin(train_targets, task_classes))[0].tolist()
        test_idx  = np.where(np.isin(test_targets,  task_classes))[0].tolist()

        rng = np.random.default_rng(seed + t)
        rng.shuffle(train_idx)
        n_val   = int(len(train_idx) * val_frac)
        val_idx = train_idx[:n_val]
        tr_idx  = train_idx[n_val:]

        train_ds = _RemapLabels(Subset(train_full, tr_idx),  label_map)
        val_ds   = _RemapLabels(Subset(train_full, val_idx), label_map)
        test_ds  = _RemapLabels(Subset(test_full,  test_idx), label_map)

        tasks.append({
            "task_id":   t,
            "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                num_workers=2, persistent_workers=True, pin_memory=False),
            "val":   DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                                num_workers=2, persistent_workers=True, pin_memory=False),
            "test":  DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                                num_workers=2, persistent_workers=True, pin_memory=False),
            "n_classes": classes_per_task,
            "class_ids": task_classes,
        })

    return tasks


def download_cifar10(data_root: str = "./data"):
    """Download CIFAR-10. Call this only with explicit user permission."""
    import torchvision.datasets as datasets
    import torchvision.transforms as T
    datasets.CIFAR10(data_root, train=True, download=True, transform=T.ToTensor())
    datasets.CIFAR10(data_root, train=False, download=True, transform=T.ToTensor())
    print(f"CIFAR-10 downloaded to {data_root}")


def download_cifar100(data_root: str = "./data"):
    """Download CIFAR-100. Call this only with explicit user permission."""
    import torchvision.datasets as datasets
    import torchvision.transforms as T
    datasets.CIFAR100(data_root, train=True, download=True, transform=T.ToTensor())
    datasets.CIFAR100(data_root, train=False, download=True, transform=T.ToTensor())
    print(f"CIFAR-100 downloaded to {data_root}")
