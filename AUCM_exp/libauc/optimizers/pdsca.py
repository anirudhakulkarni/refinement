import torch 
import copy

class PDSCA(torch.optim.Optimizer):
    """
    Reference:
    @inproceedings{
                    yuan2022compositional,
                    title={Compositional Training for End-to-End Deep AUC Maximization},
                    author={Zhuoning Yuan and Zhishuai Guo and Nitesh Chawla and Tianbao Yang},
                    booktitle={International Conference on Learning Representations},
                    year={2022},
                    url={https://openreview.net/forum?id=gPvB4pdu_Z}
                    }
    """
    def __init__(self, 
                 model, 
                 a=None, 
                 b=None, 
                 alpha=None, 
                 margin=1.0, 
                 lr=0.1, 
                 lr0=None,
                 gamma=500,
                 beta1=0.99,
                 beta2=0.999,
                 clip_value=1.0, 
                 weight_decay=1e-5, 
                 device = 'cuda',
                 **kwargs):
       
        # TODO: support a,b,alpha is None
        assert a is not None, 'Found no variable a!'
        assert b is not None, 'Found no variable b!'
        assert alpha is not None, 'Found no variable alpha!'
        
        self.margin = margin
        self.model = model
        
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device 
        
        if lr0 is None:
           lr0 = lr
       
        self.lr = lr
        self.lr0 = lr0
        self.gamma = gamma
        self.clip_value = clip_value
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        
        self.a = a 
        self.b = b 
        self.alpha = alpha 
            
        # TODO: 
        self.model_ref = self.init_model_ref()
        self.model_acc = self.init_model_acc()

        self.T = 0
        self.steps = 0
        self.backend='ce' # TODO

        def get_parameters(params):
            for p in params:
                yield p
        if self.a is not None or self.b is not None:
           self.params = get_parameters(list(model.parameters())+[self.a, self.b])
        else:
           self.params = get_parameters(list(model.parameters()))
        self.defaults = dict(lr=self.lr, 
                             lr0=self.lr0,
                             margin=margin, 
                             gamma=gamma, 
                             a=self.a, 
                             b=self.b,
                             alpha=self.alpha,
                             clip_value=self.clip_value,
                             weight_decay=self.weight_decay,
                             beta1=self.beta1,
                             beta2=self.beta2,
                             model_ref=self.model_ref,
                             model_acc=self.model_acc)
        
        super(PDSCA, self).__init__(self.params, self.defaults)

    def __setstate__(self, state):
        super(PDSCA, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def init_model_ref(self):
         self.model_ref = []
         for var in list(self.model.parameters())+[self.a, self.b]: 
            if var is not None:
               self.model_ref.append(torch.empty(var.shape).normal_(mean=0, std=0.01).to(self.device))
         return self.model_ref
     
    def init_model_acc(self):
        self.model_acc = []
        for var in list(self.model.parameters())+[self.a, self.b]: 
            if var is not None:
               self.model_acc.append(torch.zeros(var.shape, dtype=torch.float32,  device=self.device, requires_grad=False).to(self.device)) 
        return self.model_acc
    
    @property    
    def optim_steps(self):
        return self.steps
    
    @property
    def get_params(self):
        return list(self.model.parameters())
    
    def update_lr(self, lr):
        self.param_groups[0]['lr']=lr

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
 
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            clip_value = group['clip_value']
            self.lr =  group['lr']
            self.lr0 = group['lr0']
            gamma = group['gamma']
            m = group['margin']
            beta1 = group['beta1']
            beta2 = group['beta2']
            model_ref = group['model_ref']
            model_acc = group['model_acc']
            a = group['a']
            b = group['b']
            alpha = group['alpha']
            
            for i, p in enumerate(group['params']):
                if p.grad is None: 
                   continue  
                d_p = torch.clamp(p.grad.data , -clip_value, clip_value) + 1/gamma*(p.data - model_ref[i].data) + weight_decay*p.data
                if alpha.grad is None: # sgd + moving p. # TODO: alpha=None mode
                    p.data = p.data - group['lr0']*d_p 
                    if beta1!= 0: 
                        param_state = self.state[p]
                        if 'weight_buffer' not in param_state:
                            buf = param_state['weight_buffer'] = torch.clone(p).detach()
                        else:
                            buf = param_state['weight_buffer']
                            buf.mul_(1-beta1).add_(p, alpha=beta1)
                        p.data =  buf.data # Note: use buf(s) to compute the gradients w.r.t AUC loss can lead to a slight worse performance 
                elif alpha.grad is not None: # auc + moving g. # TODO: alpha=None mode
                   if beta2!= 0: 
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(1-beta2).add_(d_p, alpha=beta2)
                        d_p =  buf
                   p.data = p.data - group['lr']*d_p 
                else:
                    NotImplementedError 
                model_acc[i].data = model_acc[i].data + p.data
                
            if alpha is not None: 
               if alpha.grad is not None: 
                  alpha.data = alpha.data + group['lr']*(2*(m + b.data - a.data)-2*alpha.data)
                  alpha.data  = torch.clamp(alpha.data,  0, 999)
              
        self.T += 1        
        self.steps += 1
        return loss

    def zero_grad(self):
        self.model.zero_grad()
        if self.a is not None and self.b is not None:
           self.a.grad = None
           self.b.grad = None
        if self.alpha is not None:
           self.alpha.grad = None
        
    def update_regularizer(self, decay_factor=None):
        if decay_factor != None:
            self.param_groups[0]['lr'] = self.param_groups[0]['lr']/decay_factor
            self.param_groups[0]['lr0'] = self.param_groups[0]['lr0']/decay_factor
            print ('Reducing learning rate to %.5f (%.5f) @ T=%s!'%(self.param_groups[0]['lr'], self.param_groups[0]['lr0'], self.steps))
            
        print ('Updating regularizer @ T=%s!'%(self.steps))
        for i, param in enumerate(self.model_ref):
            self.model_ref[i].data = self.model_acc[i].data/self.T
        for i, param in enumerate(self.model_acc):
            self.model_acc[i].data = torch.zeros(param.shape, dtype=torch.float32, device=self.device,  requires_grad=False).to(self.device)
        self.T = 0
        
        
