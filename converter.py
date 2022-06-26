from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField

from pathlib import Path
from torchvision.datasets import ImageNet

root_path = str(Path.home()
        / Path(
            "Data/ImageNet/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC"
        ))

# Your dataset (`torch.utils.data.Dataset`) of (image, label) pairs
my_dataset = ImageNet(root=root_path)
write_path = "/home/lefaudeux/Data/ImageNet/ds.beton"

# Pass a type for each data field
writer = DatasetWriter(
    write_path,
    {
        # Tune options to optimize dataset size, throughput at train-time
        "image": RGBImageField(max_resolution=256, jpeg_quality=90),
        "label": IntField(),
    },
)

# Write dataset
writer.from_indexed_dataset(my_dataset)
