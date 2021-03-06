import time
import shutil
from utils import AverageMeter
import torch
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from contextlib import nullcontext


def train(train_loader, model, criterion, optimizer, epoch, args):
    # Train for one epoch
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    scaler = GradScaler()
    train_loader_len = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=train_loader_len // args.grad_accumulate,
        epochs=args.epochs,
    )

    for i, data in enumerate(train_loader):
        input = data[0]
        target = data[1]

        if args.bf16:
            input = input.bfloat16()

        # compute output & gradients
        context = autocast(enabled=True) if args.amp else nullcontext()
        with context:
            output = model(input)
            loss = criterion(output, target)

        if args.amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # optim step
        if i > 0 and i % args.grad_accumulate == 0:
            if args.amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        if i % args.print_freq == 0:
            # Every print_freq iterations, check the loss, accuracy, and speed.
            # For best performance, it doesn't make sense to print these metrics every
            # iteration, since they incur an allreduce and some host<->device syncs.

            # Measure accuracy
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            # incurs a host<->device sync
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            torch.cuda.synchronize()
            batch_time.update((time.time() - end) / args.print_freq)
            end = time.time()

            samples_seen = args.batch_size * args.grad_accumulate
            print(
                f"Epoch: [{epoch}][{i}/{train_loader_len}] | "
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | "
                f"Speed {round(samples_seen / batch_time.val)} ({round(samples_seen / batch_time.avg)}) | "
                f"LR {scheduler.get_last_lr()[0]:.5f} | "
                f"Loss {losses.val:.3f} ({losses.avg:.4f}) | "
                f"Prec@1 {top1.val:.3f} ({top1.avg:.3f}) | "
                f"Prec@5 {top5.val:.3f} ({top5.avg:.3f})"
            )

    return batch_time.avg


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


def adjust_learning_rate(optimizer, epoch, step, len_epoch, args):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    factor = epoch // 30

    if epoch >= 80:
        factor = factor + 1

    lr = args.lr * (0.1**factor)

    """Warmup"""
    warmup = 2
    if epoch < warmup:
        lr = lr * float(1 + step + epoch * len_epoch) / (float(warmup) * len_epoch)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
