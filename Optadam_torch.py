from torch.optim import *
import torch
import math
#import numpy as np
 

class Optadam_torch(Optimizer):

    """Implements OptAdam algorithm.
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
        span (int): the number of previous gradient used for the gradient prediction
    """


    def __init__(self, params, lr=1e-1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=True, span=5):
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad, span=span)
        super(Optadam_torch, self).__init__(params, defaults)


    def step(self,  optimizer_aux):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        for (group, group_aux) in zip( self.param_groups, optimizer_aux.param_groups  ):
            for (p,q) in zip( group['params'], group_aux['params'] ):
                #print (p)
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']
                span = group['span']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data).cuda()
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data).cuda()
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data).cuda()
                    # Add More State Varialbes for Opt
                    a_size = p.data.size();
                    t=1;
                    for i in range(len(a_size)):
                        t= t*a_size[i]
                    state['prev_predict_grad'] = torch.zeros_like(p.data).cuda()
                    state['w_ma']  = torch.zeros([span, t], dtype=torch.float64).cuda()
                    state['w_his'] = torch.zeros([span, t], dtype=torch.float64).cuda()
                    state['prev_w']            = p.data
                    state['aux_w']             = p.data
                ##
                aux_w  = state['aux_w']        
                a_size = p.data.size();
                t=1;
                for i in range(len(a_size)):
                    t= t*a_size[i]
                prev_w, w_ma, w_his = state['prev_w'], state['w_ma'], state['w_his']
                prev_predict_grad   = state['prev_predict_grad']

                w_diff = p.data - prev_w 
                w_diff = torch.reshape(w_diff, (1,t) )

                if ( state['step'] >= 1 and state['step'] <= span ) :
                    w_ma[state['step']-1,:] = w_diff
                if ( state['step'] > span ):
                    w_ma[:-1,:] = w_ma[1:,:]
                    w_ma[-1,:]  = w_diff

                if ( state['step'] < span ) :
                    w_his[ state['step'],:] = torch.reshape( p.data , (1,t) )
                else:
                    w_his[:-1,:] = w_his[1:,:]
                    w_his[-1,:]  = torch.reshape( p.data , (1,t) )

                wtmp = torch.zeros_like(p.data)
                if ( state['step'] >= span ):
                    la = torch.mm( w_ma, w_ma.t() )
                    la = torch.add( la , 0.001*torch.eye(span,dtype=torch.float64).cuda() )
                    lb = torch.ones([span,1],dtype=torch.float64).cuda()
                    x, LU = torch.solve(lb, la)
                    x = x / sum(x)
                    wtmp = torch.mm( w_his.t() , x) 
                    wtmp = torch.reshape( wtmp, a_size)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1)
                prev_predict_grad = q.grad.data ###
                tmp_exp = torch.add( exp_avg, (1-beta1), prev_predict_grad )
                exp_avg.add_(1 - beta1, grad)

                tmp = prev_predict_grad - grad  ### 
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, tmp, tmp)
                if amsgrad:
                   # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                aux_w.addcdiv_(-step_size, exp_avg, denom)
                prev_w = p.data ###
                p.data = torch.addcdiv(aux_w,-step_size, tmp_exp.float(), denom) 
                q.data = wtmp.float()  ###
        return loss
    
