"""Example training job: ResNet-18 on CIFAR-10, bf16 mixed precision.
Data comes from the Hugging Face mirror (uoft-cs/cifar10) and is kept on the GPU,
with augmentation done on the GPU too.

Run via MCP:
  start_job("pip install -q datasets && python examples/train_cifar.py --epochs 5")
"""
import argparse
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
ap.add_argument("--out", default="/root/cifar_resnet18.pt")
args = ap.parse_args()

dev = "cuda"
torch.backends.cudnn.benchmark = True
t_load = time.time()
ds = load_dataset("uoft-cs/cifar10")


def to_tensors(split):
    imgs = np.stack([np.asarray(im, dtype=np.uint8) for im in ds[split]["img"]])
    x = torch.from_numpy(imgs).permute(0, 3, 1, 2).to(dev).float().div_(255)
    y = torch.tensor(ds[split]["label"], device=dev)
    return x, y


xtr, ytr = to_tensors("train")
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
steps_per_epoch = len(xtr) // args.batch
opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 0.2, total_steps=args.epochs * steps_per_epoch)
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
print(f"device={torch.cuda.get_device_name(0)} steps/epoch={steps_per_epoch}", flush=True)

for ep in range(args.epochs):
    model.train()
    torch.cuda.synchronize()
    t0 = time.time()
    perm = torch.randperm(len(xtr), device=dev)
    for s in range(steps_per_epoch):
        b = perm[s * args.batch:(s + 1) * args.batch]
        x = augment(xtr[b]).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = loss_fn(model(x), ytr[b])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if s % 25 == 0:
            print(f"epoch {ep+1} step {s}/{steps_per_epoch} loss {loss.item():.3f}", flush=True)
    torch.cuda.synchronize()
    dt = time.time() - t0
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        correct = sum((model(xte[k:k + 2000].contiguous(memory_format=torch.channels_last))
                       .argmax(1) == yte[k:k + 2000]).sum().item() for k in range(0, len(xte), 2000))
    print(f"== epoch {ep+1}: test_acc {correct/len(xte):.3f}  "
          f"{steps_per_epoch*args.batch/dt:.0f} img/s  {dt:.1f}s  "
          f"vram_peak {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)

torch.save(model.state_dict(), args.out)
print(f"saved {args.out}", flush=True)
