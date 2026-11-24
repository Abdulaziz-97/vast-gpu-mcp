"""Example training job: ResNet-18 on CIFAR-10, bf16 mixed precision.
Data comes from the Hugging Face mirror (uoft-cs/cifar10) and is kept on the device,
with augmentation done there too. Uses a CUDA/ROCm GPU, Apple Silicon (MPS) or the CPU.

Run via MCP:
  start_job("pip install -q datasets torchvision")  then
  start_job("python examples/train_cifar.py --epochs 5")
"""
import argparse
import contextlib
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from datasets import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=5)
ap.add_argument("--batch", type=int, default=512)
ap.add_argument("--out", default="cifar_resnet18.pt")
ap.add_argument("--limit", type=int, default=0, help="use only the first N training images (quick test)")
args = ap.parse_args()

if torch.cuda.is_available():
    dev = "cuda"
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    dev = "mps"
else:
    dev = "cpu"
torch.backends.cudnn.benchmark = True
# bf16 mixed precision on CUDA; full precision elsewhere
amp = torch.autocast("cuda", dtype=torch.bfloat16) if dev == "cuda" else contextlib.nullcontext()


def sync():
    if dev == "cuda":
        torch.cuda.synchronize()
    elif dev == "mps":
        torch.mps.synchronize()

t_load = time.time()
ds = load_dataset("uoft-cs/cifar10")


def to_tensors(split):
    imgs = np.stack([np.asarray(im, dtype=np.uint8) for im in ds[split]["img"]])
    x = torch.from_numpy(imgs).permute(0, 3, 1, 2).to(dev).float().div_(255)
    y = torch.tensor(ds[split]["label"], device=dev)
    return x, y


xtr, ytr = to_tensors("train")
if args.limit:
    xtr, ytr = xtr[:args.limit], ytr[:args.limit]
xte, yte = to_tensors("test")
mean = torch.tensor([0.4914, 0.4822, 0.4465], device=dev).view(1, 3, 1, 1)
std = torch.tensor([0.247, 0.243, 0.261], device=dev).view(1, 3, 1, 1)
xtr, xte = (xtr - mean) / std, (xte - mean) / std
print(f"data ready: train {tuple(xtr.shape)} test {tuple(xte.shape)} in {time.time()-t_load:.1f}s",
      flush=True)


def augment(x):
    # random crop with 4px padding + horizontal flip, all on GPU
    n = x.shape[0]
    x = F.pad(x, (4, 4, 4, 4), mode="reflect")
    i = torch.randint(0, 9, (n,), device=dev)
    j = torch.randint(0, 9, (n,), device=dev)
    idx = torch.arange(32, device=dev)
    rows = (i[:, None] + idx[None]).view(n, 1, 32, 1).expand(n, 3, 32, 40)
    x = x.gather(2, rows)                                   # (n, 3, 32, 40)
    cols = (j[:, None] + idx[None]).view(n, 1, 1, 32).expand(n, 3, 32, 32)
    x = x.gather(3, cols)                                   # (n, 3, 32, 32)
    flip = torch.rand(n, device=dev) < 0.5
    x[flip] = x[flip].flip(3)
    return x


model = torchvision.models.resnet18(num_classes=10).to(dev).to(memory_format=torch.channels_last)
steps_per_epoch = max(1, len(xtr) // args.batch)
opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 0.2, total_steps=args.epochs * steps_per_epoch)
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
name = torch.cuda.get_device_name(0) if dev == "cuda" else dev
print(f"device={name} steps/epoch={steps_per_epoch}", flush=True)

for ep in range(args.epochs):
    model.train()
    sync()
    t0 = time.time()
    perm = torch.randperm(len(xtr), device=dev)
    for s in range(steps_per_epoch):
        b = perm[s * args.batch:(s + 1) * args.batch]
        x = augment(xtr[b]).contiguous(memory_format=torch.channels_last)
        with amp:
            loss = loss_fn(model(x), ytr[b])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if s % 25 == 0:
            print(f"epoch {ep+1} step {s}/{steps_per_epoch} loss {loss.item():.3f}", flush=True)
    sync()
    dt = time.time() - t0
    model.eval()
    with torch.no_grad(), amp:
        correct = sum((model(xte[k:k + 2000].contiguous(memory_format=torch.channels_last))
                       .argmax(1) == yte[k:k + 2000]).sum().item() for k in range(0, len(xte), 2000))
    print(f"== epoch {ep+1}: test_acc {correct/len(xte):.3f}  "
          f"{steps_per_epoch*args.batch/dt:.0f} img/s  {dt:.1f}s  "
          + (f"vram_peak {torch.cuda.max_memory_allocated()/1e9:.2f} GB" if dev == "cuda" else ""),
          flush=True)

torch.save(model.state_dict(), args.out)
print(f"saved {args.out}", flush=True)
