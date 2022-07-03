# [complete wip and toybox] imagenet

Minimal training scripts testing some of the [xformer](https://github.com/facebookresearch/xformers) models on the [ImageNet](https://www.image-net.org/) dataset.

Goal is to focus on small models, and fastest IN training time.

Example run:
`python3 metaformer.py --batch-size 256 --bf16 --grad_accumulate 16 --lr 0.004 --print-freq 32 --wd 0.005 --epochs 5`