import os
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def maybe_save_checkpoint(model, optimizer, path, epoch_num, save_freq):
    """
    Save a checkpoint specific to Data2Vec
    Args:
        model: a nn.Module instance
        optimizer
        path: path to save checkpoint to
        epoch_num: current epoch number
        save_freq: save frequency based on epoch number

    """
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path, f'{epoch_num}.pt')
    if epoch_num % save_freq == 0:
        checkpoint = {'data2vec': model.state_dict(),
                      'encoder': model.encoder.encoder.state_dict(),
                      'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, path)
        print(f'Saved checkpoint to `{path}`')
