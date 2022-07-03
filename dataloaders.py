from ffcv.fields.decoders import IntDecoder, RandomResizedCropRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import (
    RandomHorizontalFlip,
    ToDevice,
    ToTensor,
    ToTorchImage,
    NormalizeImage,
    Squeeze,
)
import numpy as np
from pathlib import Path

import torch


def get_ffcv_imagenet_dataloader(batch_size: int, img_size: int, workers: int):
    IN_MEAN = np.array([0.485 * 255, 0.456 * 255, 0.406 * 255])
    IN_STD = np.array([0.229 * 255, 0.224 * 255, 0.225 * 255])

    # Random resized crop
    pipeline_image = [
        RandomResizedCropRGBImageDecoder((img_size, img_size)),
        RandomHorizontalFlip(),
        ToTensor(),
        ToDevice(0, non_blocking=True),
        ToTorchImage(),
        NormalizeImage(
            mean=IN_MEAN,
            std=IN_STD,
            type=np.float16,
        ),
    ]

    pipeline_label = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(0, non_blocking=True),
    ]

    dataloader = Loader(
        str(Path.home()) + "/Data/ImageNet/ds.beton",
        batch_size=batch_size,
        num_workers=workers,
        order=OrderOption.QUASI_RANDOM,
        pipelines={
            "image": pipeline_image,
            "label": pipeline_label,
        },
        os_cache=False,  # too big to fit in memory anyway
        drop_last=True,
    )

    # TODO: filter based on indices and make a test dataloader on top

    return dataloader
