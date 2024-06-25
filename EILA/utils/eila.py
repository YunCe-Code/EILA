
from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from random import shuffle
def clip_epsilon(x_adv, x, epsilon, min=0, max=1):
    x_adv = torch.where(x_adv > x + epsilon, x + epsilon, x_adv)
    x_adv = torch.where(x_adv < x - epsilon, x - epsilon, x_adv)
    x_adv = torch.clamp(x_adv, min=min, max=max)
    return x_adv

def hook_ilout(module, input, output):
    module.output=output

class Proj_Loss(torch.nn.Module):
    def __init__(self,ori_h_feats, guide_feats):
        super(Proj_Loss, self).__init__()
        n_imgs = ori_h_feats.size(0)
        self.n_imgs = n_imgs
        self.ori_h_feats = ori_h_feats.contiguous().view(n_imgs, -1)
        guide_feats = guide_feats.contiguous().view(n_imgs, -1)
        # print (guide_feats.norm(p=2, dim=1).size())
        self.guide_feats = guide_feats / guide_feats.norm(p=2, dim=1, keepdim = True)
    def forward(self, att_h_feats):
        att_h_feats = att_h_feats.contiguous().view(self.n_imgs, -1)
        loss = ((att_h_feats - self.ori_h_feats) * self.guide_feats).sum() / self.n_imgs
        return loss

class EILA(object):
    def __init__(self, models, eps=8/255, alpha=1/255, max_value=1., min_value=0., iters1=10, iters2=50,
                device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        
        self.device = device
        self.models = models
        self.num_models = len(self.models)
        self.model_list = ['self.models[0][1].layer3[0]', 'self.models[1][1].layer3[0]', 'self.models[2][1].Mixed_6a', 'self.models[3][1].layers[2].blocks[1]', 
                           'self.models[4][1].blocks[6]', 'self.models[5][1].blocks[6]']
        hook = [eval(self.model_list[idx]).register_forward_hook(hook_ilout) for idx in range(len(self.models))]
        
        for model in models:
            model.eval()

        # attack parameter
        self.eps = eps
        self.max_value = max_value
        self.min_value = min_value
        self.alpha = alpha
        self.iters1=iters1
        self.iters2=iters2
       
        
    
    def get_adv_example(self, ori_data, adv_data, grad, attack_step=None, eps=None):
        if eps is None:
            eps = self.eps 
        if attack_step is None:
            adv_example = adv_data.detach() + grad.sign() * self.alpha
        else:
            adv_example = adv_data.detach() + grad.sign() * attack_step
        delta = torch.clamp(adv_example - ori_data.detach(), -eps, eps)
        return torch.clamp(ori_data.detach() + delta, max=self.max_value, min=self.min_value)
    
    def weighted(self, ori_data, cur_adv, grad, label):
        loss_func = torch.nn.CrossEntropyLoss()
        # generate adversarial example
        adv_exp = [self.get_adv_example(ori_data=ori_data, adv_data=cur_adv, grad=grad[idx])
                   for idx in range(self.num_models)]
        w = torch.zeros(size=(self.num_models,), device=self.device)
        for j in range(self.num_models):
            for i in range(self.num_models):
                if i == j:
                    continue
                w[j] += loss_func(self.models[i](adv_exp[j]), label)
        w = torch.softmax(w, dim=0)
        return w
    
    def _apply_pcgrad(self, grads):
        task_order = list(range(len(grads)))
        # shuffle(task_order) 
        grad_pc = [g.clone() for g in grads]
        for i in task_order:
            other_tasks = [j for j in task_order if j != i]
            for j in other_tasks:
                grad_j = grads[j]
        # Compute inner product and check for conflicting gradients
                inner_prod_sign = torch.sum(grad_pc[i]*grad_j)
                inner_prod = torch.sum(grad_pc[i]*grad_j)
                if inner_prod_sign < 0:
                    grad_pc[i] -= inner_prod / (grad_j ** 2).sum() * grad_j
        return grad_pc
    
    def attack(self, data, label, idx=-1):
        data, label = data.clone().detach().to(self.device), label.clone().detach().to(self.device)
        loss_func = nn.CrossEntropyLoss()

        adv_data = data.clone().detach() + 0.001 * torch.randn(data.shape, device=self.device)
        adv_data = adv_data.detach()

        for i in range(self.iters1):
            adv_data.requires_grad = True     
            outputs = [self.models[idx](adv_data) for idx in range(len(self.models))]
            losses = [loss_func(outputs[idx], label) for idx in range(len(self.models))]
            grads = [torch.autograd.grad(losses[idx], adv_data, retain_graph=True, create_graph=False)[0]
                     for idx in range(len(self.models))]
            with torch.no_grad():
                alpha = self.weighted(ori_data=data, cur_adv=adv_data, grad=grads, label=label)
            # print (self.alpha, alpha)
            output = torch.stack(outputs, dim=0)* alpha.view(self.num_models, 1, 1)
            output = output.sum(dim=0)
            loss = loss_func(output, label)
            grad = torch.autograd.grad(loss.sum(dim=0), adv_data)[0]
            adv_data = self.get_adv_example(ori_data=data, adv_data=adv_data, grad=grad)
            adv_data.detach_()    

        img = data.clone()
        for i in range(len(self.models)):
                self.models[i](data) 
        ori_mid = [eval(self.model_list[idx]).output for idx in range(len(self.models))]
        for idx in range(len(self.models)):
            self.models[idx](adv_data)
        x_adv_mid = [eval(self.model_list[idx]).output.detach() for idx in range(len(self.models))]
        loss_fn = [Proj_Loss(ori_h_feats=ori_mid[idx].data, guide_feats=(x_adv_mid[idx]-ori_mid[idx]).data) for idx in range(len(self.models))] 

        for iters in range(self.iters2):
            img.requires_grad = True 
            for idx in range(len(self.models)):
                self.models[idx].zero_grad()
                self.models[idx](img)
            x_adv2_mid = [eval(self.model_list[idx]).output for idx in range(len(self.models))]
            losses = [loss_fn[idx](x_adv2_mid[idx]) for idx in range(len(self.models))]           
            grads = [torch.autograd.grad(losses[idx], img, retain_graph=False, create_graph=False)[0]
            for idx in range(len(self.models))]
            grad_pc = self._apply_pcgrad(grads=grads)
            grad = torch.zeros_like(grads[0],device=self.device) 
            for idx in range(len(grads)):
                grad += grad_pc[idx]/torch.mean(torch.abs(grad_pc[idx]), dim=(1, 2, 3), keepdim=True) 

            img = img.data + self.alpha*torch.sign(grad)
            img = clip_epsilon(img, data, self.eps)
            img.detach_()
        return img
    
    def __call__(self, data, label, idx=-1):
        return self.attack(data, label, idx)
    

