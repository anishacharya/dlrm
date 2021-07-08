import numpy as np
import torch


def flatten_grads(learner):
    """ Given a model flatten all params and return as np array """
    # flat_grad = np.concatenate([w.grad.data.cpu().numpy().flatten() for w in learner.parameters()])
    flat_grad = []
    for w in learner.parameters():
        grad = w.grad.data
        if grad.is_sparse:
            grad = grad.to_dense()
        flattened = torch.reshape(grad, (-1,))
        # flattened = torch.reshape(w.grad.data.cpu(), (-1,))
    # flat_grad = torch.cat([torch.flatten(w.grad.data.cpu()) for w in learner.parameters()])
    return flat_grad
