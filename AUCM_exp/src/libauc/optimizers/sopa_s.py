import torch
from torch.optim import Adam, SGD

class SOPAs(torch.optim.Optimizer):
    r"""A wrapper class for different optimizing methods.

        Arguments:
            params (iterable): iterable of parameters to optimize or dicts defining
                parameter groups
            lr (float): learning rate
            loss_fn: the instance of loss class
            method (str): optimization method
            weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

            Arguments for SGD optimization method:
                momentum (float, optional): momentum factor (default: 0.9)
                dampening (float, optional): dampening for momentum (default: 0.1)
                nesterov (bool, optional): enables Nesterov momentum (default: False)
            Arguments for ADAM optimization method:
                betas (Tuple[float, float], optional): coefficients used for computing
                    running averages of gradient and its square (default: (0.9, 0.999))
                eps (float, optional): term added to the denominator to improve
                    numerical stability (default: 1e-8)
                amsgrad (boolean, optional): whether to use the AMSGrad variant of this
                    algorithm from the paper `On the Convergence of Adam and Beyond`_
                    (default: False)

        Example:
            >>> loss_fn = pAUC_DRO_Loss(data_len=len(train_dataset), margin=1.0, beta=0.9, Lambda=1.0)
            >>> optimizer = pAUC_DRO_Optimizer(model.parameters(), lr=1e-3, loss_fn=loss_fn,  mode='adam')
            >>> optimizer.zero_grad()
            >>> loss = loss_fn(input, target, index)
            >>> loss.backward()
            >>> optimizer.step() 
    """
    def __init__(self,params, lr, loss_fn, mode = 'adam', weight_decay=0, 
                    momentum=0.0, nesterov=False, # sgd
                    betas=(0.9, 0.999), eps=1e-8,  amsgrad=False  # adam
                ):
        self.loss_fn = loss_fn
        if mode == 'sgd':
           self.optimizer = SGD(params, lr, momentum, momentum, weight_decay, nesterov)
        elif mode == 'adam':
           self.optimizer = Adam(params, lr, betas, eps, weight_decay, amsgrad)
    
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        self.optimizer.step(closure=closure)

    def zero_grad(self, set_to_none: bool = False):
        r"""Sets the gradients of all optimized :class:`torch.Tensor` s to zero.
        Args:
            set_to_none (bool): instead of setting to zero, set the grads to None.
                This will in general have lower memory footprint, and can modestly improve performance.
                However, it changes certain behaviors. For example:
                1. When the user tries to access a gradient and perform manual ops on it,
                a None attribute or a Tensor full of 0s will behave differently.
                2. If the user requests ``zero_grad(set_to_none=True)`` followed by a backward pass, ``.grad``\ s
                are guaranteed to be None for params that did not receive a gradient.
                3. ``torch.optim`` optimizers have a different behavior if the gradient is 0 or None
                (in one case it does the step with a gradient of 0 and in the other it skips
                the step altogether).
        """
        self.optimizer.zero_grad(set_to_none)

    def update_regularizer(self, lr_decay_factor=None, MA_decay_factor=None):
        if lr_decay_factor != None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] =  param_group['lr']/lr_decay_factor
                print ('decaying learning rate to %f  !'%param_group['lr'] )

        if MA_decay_factor != None:
            self.loss_fn.update_MA_coef(MA_decay_factor)
            print ('decaying coefficient for moving average to %f !' %self.loss_fn.get_MA_coef )

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    @property    
    def lr(self):
        return self.optimizer.param_groups[0]['lr']
