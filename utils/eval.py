from __future__ import print_function, absolute_import
import numpy as np
from sklearn.metrics import accuracy_score
import torch
import matplotlib.pyplot as plt
__all__ = ['accuracy']

def accuracy(output, target, topk=(1,), nb_classes = None):
    """Computes the precision@k for the specified values of k"""
    if nb_classes != 1:
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    else:
        return torch.FloatTensor([accuracy_score(y_pred = (output.total_data.cpu().numpy() > 0.5).astype(np.float32),
                                                 y_true = target.cpu().numpy())]).cuda()



