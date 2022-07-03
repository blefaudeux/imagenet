import torch

import gc
from torch import nn
from torchmetrics import Accuracy

from xformers.components import MultiHeadDispatch
from xformers.components.attention import ScaledDotProduct
from xformers.components.patch_embedding import PatchEmbeddingConfig  # noqa
from xformers.components.patch_embedding import build_patch_embedding  # noqa
from xformers.factory import xFormer, xFormerConfig
from xformers.helpers.hierarchical_configs import (
    BasicLayerConfig,
    get_hierarchical_configuration,
)

import os
from utils import AverageMeter
from args import parse
from trainer import train, save_checkpoint

# CREDITS: Inspired by the Dali and FFCV imagenet examples
from dataloaders import get_ffcv_imagenet_dataloader


class EfficientFormer(nn.Module):
    def __init__(
        self,
        steps,
        learning_rate=5e-4,
        betas=(0.9, 0.99),
        weight_decay=0.03,
        image_size=32,
        num_classes=10,
        dim=384,
        linear_warmup_ratio=0.1,
    ):

        super().__init__()

        # Generate the skeleton of our hierarchical Transformer
        # This implements a model close to the L1 suggested in "EfficientFormer" (https://arxiv.org/abs/2206.01191)
        base_hierarchical_configs = [
            BasicLayerConfig(
                embedding=48,
                attention_mechanism="pooling",
                patch_size=3,
                stride=2,
                padding=1,
                seq_len=image_size * image_size // 16,
                feedforward="ConvBN",
                normalization="skip",
            ),
            BasicLayerConfig(
                embedding=96,
                attention_mechanism="pooling",
                patch_size=3,
                stride=2,
                padding=1,
                seq_len=image_size * image_size // 64,
                feedforward="ConvBN",
                normalization="skip",
            ),
            BasicLayerConfig(
                embedding=224,
                attention_mechanism="pooling",
                patch_size=3,
                stride=2,
                padding=1,
                seq_len=image_size * image_size // 256,
                feedforward="ConvBN",
                normalization="skip",
            ),
            BasicLayerConfig(
                embedding=448,
                attention_mechanism="pooling",
                patch_size=3,
                stride=2,
                padding=1,
                seq_len=image_size * image_size // 1024,
                feedforward="ConvBN",
                normalization="skip",
            ),
        ]

        # Fill in the gaps in the config
        xformer_config = get_hierarchical_configuration(
            base_hierarchical_configs,
            layernorm_style="post",
            use_rotary_embeddings=True,
            mlp_multiplier=4,
            in_channels=24,
        )

        # Now instantiate the EfficientFormer trunk
        config = xFormerConfig(xformer_config)
        config.weight_init = "moco"

        self.trunk = xFormer.from_config(config)
        print(self.trunk)

        # This model requires a pre-stem (a conv prior to going through all the layers above)
        self.pre_stem = build_patch_embedding(
            PatchEmbeddingConfig(
                in_channels=3, out_channels=24, kernel_size=3, stride=2, padding=1
            )
        )

        # This model requires a final Attention step
        self.attention = MultiHeadDispatch(
            dim_model=448, num_heads=8, attention=ScaledDotProduct()
        )

        # The classifier head
        dim = base_hierarchical_configs[-1].embedding
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self.val_accuracy = Accuracy()

    def forward(self, x):
        # Main processing
        x = x.flatten(-2, -1).transpose(-1, -2)  # BCHW to BSE
        x = self.pre_stem(x)
        x = self.trunk(x)
        x = self.attention(x)
        x = self.ln(x)

        # Classify
        x = x.mean(dim=1)  # mean over sequence len
        x = self.head(x)
        return x


if __name__ == "__main__":
    args = parse()

    GPUS = 1
    IMG_SIZE = 224
    NUM_CLASSES = 1000

    torch.cuda.manual_seed_all(42)
    torch.manual_seed(42)

    dataloader = get_ffcv_imagenet_dataloader(
        args.batch_size, IMG_SIZE, workers=args.workers
    )

    steps = len(dataloader) // args.epochs

    # compute total number of steps
    batch_size = args.batch_size * GPUS
    model = EfficientFormer(
        steps=steps,
        image_size=IMG_SIZE,
        num_classes=NUM_CLASSES,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Optionally resume from a checkpoint
    if args.resume:
        # Use a local scope to avoid dangling references
        def resume():
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(
                    args.resume,
                    map_location=lambda storage, loc: storage.cuda(args.gpu),
                )
                args.start_epoch = checkpoint["epoch"]
                global best_prec1
                best_prec1 = checkpoint["best_prec1"]
                model.load_state_dict(checkpoint["state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                print(
                    "=> loaded checkpoint '{}' (epoch {})".format(
                        args.resume, checkpoint["epoch"]
                    )
                )
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

        resume()

    total_time = AverageMeter()
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        avg_train_time = train(
            train_loader=dataloader,
            model=model.cuda(),
            criterion=nn.CrossEntropyLoss().cuda(),
            optimizer=optimizer,
            epoch=epoch,
            args=args,
        )

        total_time.update(avg_train_time)

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr": args.lr,
                "batch_size": args.batch_size,
            },
            is_best=False,
            filename=f"checkpoint_{epoch}.pth.tar",
        )

        # Failsafe: clear caches
        torch.cuda.empty_cache()
        gc.collect()

        # evaluate on validation set
        # [prec1, prec5] = validate(val_loader, model, criterion)

        # # remember best prec@1 and save checkpoint
        # if args.local_rank == 0:
        #     is_best = prec1 > best_prec1
        #     best_prec1 = max(prec1, best_prec1)
        #     save_checkpoint(
        #         {
        #             "epoch": epoch + 1,
        #             "arch": args.arch,
        #             "state_dict": model.state_dict(),
        #             "best_prec1": best_prec1,
        #             "optimizer": optimizer.state_dict(),
        #         },
        #         is_best,
        #     )
        #     if epoch == args.epochs - 1:
        #         print(
        #             "##Top-1 {0}\n"
        #             "##Top-5 {1}\n"
        #             "##Perf  {2}".format(
        #                 prec1, prec5, args.total_batch_size / total_time.avg
        #             )
        #         )

        # dataloader.reset()
        # val_loader.reset()
