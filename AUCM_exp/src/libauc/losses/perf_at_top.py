import torch 
import torch.nn.functional as F
from .surrogate import squared_loss, squared_hinge_loss, logistic_loss

class TopPush_Loss(torch.nn.Module):
    """Partial AUC Loss: a stochastic one-way partial AUC based on DRO-CVaR (Top Push Loss)
      
      Args:
        pos_length: number of positive examples for the training data
        num_neg: number of negative samples for each mini-batch
        threshold: margin for basic AUC loss
        beta: FPR upper bound for pAUC used for SOTA
        eta: stepsize for CVaR regularization term
        loss type: basic AUC loss to apply.
        
        Return:
            loss value (scalar)
            
        Reference:    
            Zhu, D., Li, G., Wang, B., Wu, X., and Yang, T., 2022. 
            When AUC meets DRO: Optimizing Partial AUC for Deep Learning with Non-Convex Convergence Guarantee. 
            arXiv preprint arXiv:2203.00176.   
            
    """                                     
    def __init__(self, 
                 pos_length, 
                 num_neg, 
                 threshold=1.0, 
                 alpha=0, 
                 beta=0.2, 
                 surrogate_loss='squared_hinge', 
                 top_push=False):
        
        super(pAUC_CVaR_Loss, self).__init__()                                 
        self.beta = 1/num_neg # choose hardest negative samples in mini-batch    
        self.alpha = alpha                                 
        self.eta = 1.0
        self.num_neg = num_neg
        self.pos_length = pos_length
        self.lambda_pos = torch.tensor([0.0]*pos_length).reshape(-1, 1).cuda()             
        self.threshold = threshold                        
        self.surrogate_loss = _get_surrogate_loss(surrogate_loss)
                                       
    def set_eta(self, eta):
        self.eta = eta
                                       
    def update_eta(self, decay_factor):
        self.eta = self.eta/decay_factor
    
    def forward(self, y_pred, y_true, index_p):
        y_pred = _check_tensor_shape(y_pred, (-1, 1))
        y_true = _check_tensor_shape(y_true, (-1, 1))
        index_p = _check_tensor_shape(index_p, (-1,))
        
        f_ps = y_pred[y_true == 1].reshape(-1, 1)
        f_ns = y_pred[y_true == 0].reshape(-1, 1)
        f_ps = f_ps.repeat(1, len(f_ns))
        f_ns = f_ns.repeat(1, len(f_ps))     
        index_p = index_p[index_p>=0]
        
        loss = self.surrogate_loss(self.threshold, f_ps - f_ns.transpose(0,1)) # return element-wsie loss 
        p = loss > self.lambda_pos[index_p]
        self.lambda_pos[index_p] = self.lambda_pos[index_p]-self.eta/self.pos_length*(1 - p.sum(dim=1, keepdim=True)/(self.beta*self.num_neg))
        
        p.detach_()
        loss = torch.mean(p * loss) / self.beta
        return loss
