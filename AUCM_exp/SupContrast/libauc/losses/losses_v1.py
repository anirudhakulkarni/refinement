import torch 
import torch.nn.functional as F


class AUCMLoss_V1(torch.nn.Module):
    """
    AUCM Loss with squared-hinge function: a novel loss function to directly optimize AUROC
    
    inputs:
        margin: margin term for AUCM loss, e.g., m in [0, 1]
        imratio: imbalance ratio, i.e., the ratio of number of postive samples to number of total samples
    outputs:
        loss value 
        
    Reference: 
        Yuan, Z., Yan, Y., Sonka, M. and Yang, T., 
        Large-scale Robust Deep AUC Maximization: A New Surrogate Loss and Empirical Studies on Medical Image Classification. 
        International Conference on Computer Vision (ICCV 2021)
    Link:
        https://arxiv.org/abs/2012.03173
    """
    def __init__(self, margin=1.0, imratio=None, device=None):
        super(AUCMLoss_V1, self).__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device   
        self.margin = margin
        self.p = imratio
        self.a = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device) 
        self.b = torch.zeros(1, dtype=torch.float32, device=self.device,  requires_grad=True).to(self.device) 
        self.alpha = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device) 
        
    def forward(self, y_pred, y_true, auto=True):
        if auto or not self.p:
           self.p = (y_true==1).sum()/y_true.shape[0]   
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(-1, 1)
        if len(y_true.shape) == 1:
            y_true = y_true.reshape(-1, 1) 
        loss = (1-self.p)*torch.mean((y_pred - self.a)**2*(1==y_true).float()) + \
                    self.p*torch.mean((y_pred - self.b)**2*(0==y_true).float())   + \
                    2*self.alpha*(self.p*(1-self.p) + \
                    torch.mean((self.p*y_pred*(0==y_true).float() - (1-self.p)*y_pred*(1==y_true).float())) )- \
                    self.p*(1-self.p)*self.alpha**2
        # norm = torch.norm(loss, p=2, dim=-1, keepdim=True) + 1e-7
        # loss = torch.div(loss,norm) / 0.01
        return loss
class AUCMLoss_MDCA_V1(torch.nn.Module):
    """
    AUCM Loss with squared-hinge function: a novel loss function to directly optimize AUROC
    
    inputs:
        margin: margin term for AUCM loss, e.g., m in [0, 1]
        imratio: imbalance ratio, i.e., the ratio of number of postive samples to number of total samples
    outputs:
        loss value 
        
    Reference: 
        Yuan, Z., Yan, Y., Sonka, M. and Yang, T., 
        Large-scale Robust Deep AUC Maximization: A New Surrogate Loss and Empirical Studies on Medical Image Classification. 
        International Conference on Computer Vision (ICCV 2021)
    Link:
        https://arxiv.org/abs/2012.03173
    """
    def __init__(self, margin=1.0, imratio=None, device=None, beta=1.0):
        print(beta)
        super(AUCMLoss_MDCA_V1, self).__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device   
        self.margin = margin
        self.p = imratio
        self.a = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device) 
        self.b = torch.zeros(1, dtype=torch.float32, device=self.device,  requires_grad=True).to(self.device) 
        self.alpha = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device) 
        self.beta = beta

    def forward(self, y_pred, y_true, auto=True):
        if auto or not self.p:
           self.p = (y_true==1).sum()/y_true.shape[0]   
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(-1, 1)
        if len(y_true.shape) == 1:
            y_true = y_true.reshape(-1, 1) 
        loss = (1-self.p)*torch.mean((y_pred - self.a)**2*(1==y_true).float()) + \
                    self.p*torch.mean((y_pred - self.b)**2*(0==y_true).float())   + \
                    2*self.alpha*(self.p*(1-self.p) + \
                    torch.mean((self.p*y_pred*(0==y_true).float() - (1-self.p)*y_pred*(1==y_true).float())) )- \
                    self.p*(1-self.p)*self.alpha**2

        y_pred = torch.softmax(y_pred, dim=1)
        # [batch, classes]
        loss_mdca = torch.tensor(0.0).cuda()
        batch, classes = y_pred.shape
        for c in range(classes):
            avg_count = (y_true == c).float().mean()
            avg_conf = torch.mean(y_pred[:,c])
            loss_mdca += torch.abs(avg_conf - avg_count)
        denom = classes
        loss_mdca /= denom
        
        return loss+self.beta*loss_mdca      
      
class AUCM_MultiLabel_V1(torch.nn.Module):
    """
    Reference: 
        Yuan, Z., Yan, Y., Sonka, M. and Yang, T., 
        Large-scale Robust Deep AUC Maximization: A New Surrogate Loss and Empirical Studies on Medical Image Classification. 
        International Conference on Computer Vision (ICCV 2021)
    Link:
        https://arxiv.org/abs/2012.03173
    """
    def __init__(self, margin=1.0, imratio=None, num_classes=10, device=None):
        super(AUCM_MultiLabel_V1, self).__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device   
        self.margin = margin
        self.p =  imratio 
        self.num_classes = num_classes
        if self.p:
           assert len(imratio)==num_classes, 'Length of imratio needs to be same as num_classes!'
        else:
            self.p = [0.0]*num_classes
        self.a = torch.zeros(num_classes, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)
        self.b = torch.zeros(num_classes, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)
        self.alpha = torch.zeros(num_classes, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)

    @property
    def get_a(self):
        return self.a.mean()
    @property
    def get_b(self):
        return self.b.mean()
    @property
    def get_alpha(self):
        return self.alpha.mean()
    
    def forward(self, y_pred, y_true, auto=True):
        total_loss = 0
        for idx in range(self.num_classes):
            if len(y_pred[:, idx].shape) == 1:
               y_pred_i = y_pred[:, idx].reshape(-1, 1)
            if len(y_true[:, idx].shape) == 1:
               y_true_i = y_true[:, idx].reshape(-1, 1)
            if auto or not self.p:
               self.p[idx] = (y_true_i==1).sum()/y_true_i.shape[0]   
            loss = (1-self.p[idx])*torch.mean((y_pred_i - self.a[idx])**2*(1==y_true_i).float()) + \
                        self.p[idx]*torch.mean((y_pred_i - self.b[idx])**2*(0==y_true_i).float())   + \
                        2*self.alpha[idx]*(self.p[idx]*(1-self.p[idx]) + \
                        torch.mean((self.p[idx]*y_pred_i*(0==y_true_i).float() - (1-self.p[idx])*y_pred_i*(1==y_true_i).float())) )- \
                        self.p[idx]*(1-self.p[idx])*self.alpha[idx]**2
            total_loss += loss
        return total_loss
           
class CompositionalAUCLoss_V1(torch.nn.Module):
    """  
        Compositional AUC Loss: a novel loss function to directly optimize AUROC
        inputs:
            margin: margin term for AUCM loss, e.g., m in [0, 1]
            imratio: imbalance ratio, i.e., the ratio of number of postive samples to number of total samples
        outputs:
            loss  
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
    def __init__(self, imratio=None,  margin=1, backend='ce', device=None):
        super(CompositionalAUCLoss_V1, self).__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device   
        self.margin = margin
        self.p = imratio
        self.a = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device) 
        self.b = torch.zeros(1, dtype=torch.float32, device=self.device,  requires_grad=True).to(self.device) 
        self.alpha = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device) 
        self.L_AVG = F.binary_cross_entropy_with_logits  # with sigmoid
        self.backend = 'ce'  #TODO: 

    def forward(self, y_pred, y_true, auto=True):
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(-1, 1)
        if len(y_true.shape) == 1:
            y_true = y_true.reshape(-1, 1)
        if self.backend == 'ce':
           self.backend = 'auc'
           return self.L_AVG(y_pred, y_true)
        else:
           self.backend = 'ce'
           if auto or not self.p:
              self.p = (y_true==1).sum()/y_true.shape[0] 
           y_pred = torch.sigmoid(y_pred)
           self.L_AUC = (1-self.p)*torch.mean((y_pred - self.a)**2*(1==y_true).float()) + \
                      self.p*torch.mean((y_pred - self.b)**2*(0==y_true).float())   + \
                      2*self.alpha*(self.p*(1-self.p) + \
                      torch.mean((self.p*y_pred*(0==y_true).float() - (1-self.p)*y_pred*(1==y_true).float())) )- \
                      self.p*(1-self.p)*self.alpha**2
           return self.L_AUC 

class AUCM_MultiLabel_MDCA_V1(torch.nn.Module):
    """
    Reference: 
        Yuan, Z., Yan, Y., Sonka, M. and Yang, T., 
        Large-scale Robust Deep AUC Maximization: A New Surrogate Loss and Empirical Studies on Medical Image Classification. 
        International Conference on Computer Vision (ICCV 2021)
    Link:
        https://arxiv.org/abs/2012.03173
    """
    def __init__(self, margin=1.0, imratio=None, num_classes=10, device=None,beta=1.0):
        super(AUCM_MultiLabel_MDCA_V1, self).__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device   
        self.margin = margin
        self.p =  imratio 
        self.num_classes = num_classes
        if self.p:
           assert len(imratio)==num_classes, 'Length of imratio needs to be same as num_classes!'
        else:
            self.p = [0.0]*num_classes
        self.a = torch.zeros(num_classes, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)
        self.b = torch.zeros(num_classes, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)
        self.alpha = torch.zeros(num_classes, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)
        self.beta = beta
    @property
    def get_a(self):
        return self.a.mean()
    @property
    def get_b(self):
        return self.b.mean()
    @property
    def get_alpha(self):
        return self.alpha.mean()
    
    def forward(self, y_pred, y_true, auto=True):
        total_loss = 0
        for idx in range(self.num_classes):
            if len(y_pred[:, idx].shape) == 1:
               y_pred_i = y_pred[:, idx].reshape(-1, 1)
            if len(y_true[:, idx].shape) == 1:
               y_true_i = y_true[:, idx].reshape(-1, 1)
            if auto or not self.p:
               self.p[idx] = (y_true_i==1).sum()/y_true_i.shape[0]   
            loss = (1-self.p[idx])*torch.mean((y_pred_i - self.a[idx])**2*(1==y_true_i).float()) + \
                        self.p[idx]*torch.mean((y_pred_i - self.b[idx])**2*(0==y_true_i).float())   + \
                        2*self.alpha[idx]*(self.p[idx]*(1-self.p[idx]) + \
                        torch.mean((self.p[idx]*y_pred_i*(0==y_true_i).float() - (1-self.p[idx])*y_pred_i*(1==y_true_i).float())) )- \
                        self.p[idx]*(1-self.p[idx])*self.alpha[idx]**2
            total_loss += loss


        y_pred = torch.softmax(y_pred, dim=1)
        # [batch, classes]
        loss_mdca = torch.tensor(0.0).cuda()
        batch, classes = y_pred.shape
        for c in range(classes):
            avg_count = (y_true == c).float().mean()
            avg_conf = torch.mean(y_pred[:,c])
            loss_mdca += torch.abs(avg_conf - avg_count)
        denom = classes
        loss_mdca /= denom
        
        return total_loss+self.beta*loss_mdca
     

# class AUCM_MultiLabel_MDCA_V1(torch.nn.Module):
#     """
#     Reference: 
#         Yuan, Z., Yan, Y., Sonka, M. and Yang, T., 
#         Large-scale Robust Deep AUC Maximization: A New Surrogate Loss and Empirical Studies on Medical Image Classification. 
#         International Conference on Computer Vision (ICCV 2021)
#     Link:
#         https://arxiv.org/abs/2012.03173
#     """
#     def __init__(self, margin=1.0, imratio=[0.1], num_classes=10, device=None, beta=1.0):
#         super(AUCM_MultiLabel_MDCA_V1, self).__init__()
#         if not device:
#             self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         else:
#             self.device = device   
#         self.margin = margin
#         self.p = torch.FloatTensor(imratio).to(self.device)
#         self.num_classes = num_classes
#         print(imratio)
#         print(num_classes)
#         assert len(imratio)==num_classes, 'Length of imratio needs to be same as num_classes!'
#         self.a = torch.zeros(num_classes, dtype=torch.float32, device="cuda", requires_grad=True).to(self.device)
#         self.b = torch.zeros(num_classes, dtype=torch.float32, device="cuda", requires_grad=True).to(self.device)
#         self.alpha = torch.zeros(num_classes, dtype=torch.float32, device="cuda", requires_grad=True).to(self.device)
#         self.beta = beta
#     @property
#     def get_a(self):
#         return self.a.mean()
#     @property
#     def get_b(self):
#         return self.b.mean()
#     @property
#     def get_alpha(self):
#         return self.alpha.mean()

#     def forward(self, y_pred, y_true):
#         total_loss = 0
#         for idx in range(self.num_classes):
#             y_pred_i = y_pred[:, idx].reshape(-1, 1)
#             y_true_i = y_true[:, idx].reshape(-1, 1)
#             loss = (1-self.p[idx])*torch.mean((y_pred_i - self.a[idx])**2*(1==y_true_i).float()) + \
#                         self.p[idx]*torch.mean((y_pred_i - self.b[idx])**2*(0==y_true_i).float())   + \
#                         2*self.alpha[idx]*(self.p[idx]*(1-self.p[idx]) + \
#                         torch.mean((self.p[idx]*y_pred_i*(0==y_true_i).float() - (1-self.p[idx])*y_pred_i*(1==y_true_i).float())) )- \
#                         self.p[idx]*(1-self.p[idx])*self.alpha[idx]**2
#             total_loss += loss

#         y_pred = torch.softmax(y_pred, dim=1)
#         # [batch, classes]
#         loss_mdca = torch.tensor(0.0).cuda()
#         batch, classes = y_pred.shape
#         for c in range(classes):
#             avg_count = (y_true == c).float().mean()
#             avg_conf = torch.mean(y_pred[:,c])
#             loss_mdca += torch.abs(avg_conf - avg_count)
#         denom = classes
#         loss_mdca /= denom
        
#         return total_loss+self.beta*loss_mdca
 
# alias
AUCMLoss = AUCMLoss_V1
AUCM_MDCALoss = AUCMLoss_MDCA_V1
AUCM_MultiLabel = AUCM_MultiLabel_V1
CompositionalAUCLoss = CompositionalAUCLoss_V1
AUCM_MultiLabel_MDCA=AUCM_MultiLabel_MDCA_V1