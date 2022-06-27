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
from utils import AverageMeter, parse
from trainer import train, save_checkpoint

# CREDITS: Inspired by the Dali and FFCV imagenet examples
from dataloaders import get_ffcv_imagenet_dataloader


class MetaFormer(nn.Module):
    def __init__(
        self,
        steps,
        learning_rate=5e-3,
        betas=(0.9, 0.99),
        weight_decay=0.03,
        image_size=224,
        num_classes=10,
        dim=512,
        attention="scaled_dot_product",
        layer_norm_style="pre",
        use_rotary_embeddings=True,
        linear_warmup_ratio=0.1,
    ):

        super().__init__()

        # This corresponds to the S12 Poolformer

        # FIXME: pass a "repeat" field instead

        # Generate the skeleton of our hierarchical Transformer
        base_hierarchical_configs = [
            BasicLayerConfig(
                embedding=64,
                attention_mechanism=attention,
                patch_size=7,
                stride=4,
                padding=3,
                seq_len=image_size * image_size // 16,
                feedforward="Conv2DFeedforward",
                repeat_layer=2,
            )
        ]

        base_hierarchical_configs += [
            BasicLayerConfig(
                embedding=128,
                attention_mechanism=attention,
                patch_size=3,
                stride=2,
                padding=1,
                seq_len=image_size * image_size // 64,
                feedforward="Conv2DFeedforward",
                repeat_layer=2,
            )
        ]

        base_hierarchical_configs += [
            BasicLayerConfig(
                embedding=320,
                attention_mechanism=attention,
                patch_size=3,
                stride=2,
                padding=1,
                seq_len=image_size * image_size // 256,
                feedforward="Conv2DFeedforward",
                repeat_layer=6,
            )
        ]

        base_hierarchical_configs += [
            BasicLayerConfig(
                embedding=512,
                attention_mechanism=attention,
                patch_size=3,
                stride=2,
                padding=1,
                seq_len=image_size * image_size // 1024,
                feedforward="Conv2DFeedforward",
                repeat_layer=2,
            )
        ]

        # Fill in the gaps in the config
        xformer_config = get_hierarchical_configuration(
            base_hierarchical_configs,
            layernorm_style=layer_norm_style,
            use_rotary_embeddings=use_rotary_embeddings,
            mlp_multiplier=4,
            dim_head=32,
        )

        # Now instantiate the metaformer trunk
        config = xFormerConfig(xformer_config)
        config.weight_init = "timm"
        print(config)
        self.trunk = xFormer.from_config(config)
        print(self.trunk)

        # The classifier head
        dim = base_hierarchical_configs[-1].embedding
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.val_accuracy = Accuracy()

    def forward(self, x):
        x = self.trunk(x)
        x = self.ln(x)

        x = x.mean(dim=1)  # mean over sequence len
        x = self.head(x)
        return x


if __name__ == "__main__":
    args = parse()

    NUM_WORKERS = 6
    GPUS = 1
    IMG_SIZE = 224
    NUM_CLASSES = 1000

    torch.cuda.manual_seed_all(42)
    torch.manual_seed(42)

    dataloader = get_ffcv_imagenet_dataloader(
        args.batch_size, IMG_SIZE, workers=NUM_WORKERS
    )

    steps = len(dataloader) // args.epochs

    # compute total number of steps
    batch_size = args.batch_size * GPUS
    model = MetaFormer(
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
