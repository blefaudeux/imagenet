class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def describe_model(model):
    num_params = sum(param.numel() for param in model.parameters())
    num_trainable_params = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )
    print(
        f"{round(num_params / 1e6, 2)}M parameters | "
        f"{round(num_trainable_params / 1e6, 2)}M trainable parameters"
    )
