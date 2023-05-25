import copy
import time
import torch
import torch.nn.functional as F
import numpy as np

from .ken_base import StochLineSearchBase, get_grad_list, compute_grad_norm, random_seed_torch, try_sgd_update

#gets a nested list of parameters as input
class KenSLS(StochLineSearchBase):
    def __init__(self,
                 params,
                 n_batches_per_epoch=500,
                 init_step_size=0.1,
                 c=0.2,
                 gamma=2.0,
                 beta=0.999,
                 momentum=0.9,
                 gv_option='per_param',
                 base_opt='adam',
                 pp_norm_method='pp_armijo',
                 strategy = "cycle",
                 mom_type='standard',
                 clip_grad=False,
                 beta_b=0.9,
                 beta_s = 0.99,
                 reset_option=1,
                 timescale = 0.05,
                 line_search_fn="armijo",
                 smooth = True,
                 smooth_after = 0,
                 only_decrease = False):
        params = list(params)
        super().__init__(params,
                         n_batches_per_epoch=n_batches_per_epoch,
                         init_step_size=init_step_size,
                         c=c,
                         beta_b=beta_b,
                         gamma=gamma,
                         reset_option=reset_option,
                         line_search_fn=line_search_fn)
        self.mom_type = mom_type
        self.pp_norm_method = pp_norm_method

        self.init_step_size = init_step_size
        self.step_size = init_step_size 

        self.smooth = smooth
        # sls stuff
        self.beta_b = beta_b
        self.beta_s = beta_s
        self.smooth_after = smooth_after
        self.only_decrease = only_decrease
        if not self.smooth_after == 0:
            self.smooth = False

        # others
        self.nextcycle = 0
        self.params = params
        if self.mom_type == 'heavy_ball':
            self.params_prev = copy.deepcopy(params) 


        self.momentum = momentum
        self.beta = beta
        self.first_step = True
        self.timescale = timescale
        # self.state['step_size'] = init_step_size

      #  self.avg_decrease = torch.zeros(len(params))#(0.0 for i in range(len(params))]
        self.avg_step_size = torch.zeros(len(params))#[0.0 for i in range(len(params))]
      #  self.avg_gradient_norm_scaled = torch.zeros(len(params))#[0.0 for i in range(len(params))]

        self.clip_grad = clip_grad
        self.gv_option = gv_option
        self.base_opt = base_opt
        # self.step_size_method = step_size_method

        # gv options
        self.gv_option = gv_option
        if self.gv_option in ['scalar']:
            self.state['gv'] = [[0.]]

        elif self.gv_option == 'per_param':
            self.state['gv'] = [torch.zeros(p.shape).to(p.device) for p in self.params]
            self.state['mv'] = [torch.zeros(p.shape).to(p.device) for p in self.params] 
        self.state["ema_cosine_similarity"] = 0.0
        

    def step(self, closure, closure_with_backward = None):
        # deterministic closure
        seed = time.time()
        start = time.time()
        def closure_deterministic():
            with random_seed_torch(int(seed)):
                return closure()

        if closure_with_backward is not None:
            loss = closure_with_backward()
        else:
            loss = closure_deterministic()
            loss.backward()
        # print("time for backwards:", time.time()-start)
        # start = time.time()
        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.params, 0.25)
        # increment # forward-backward calls
        self.state['n_forwards'] += 1
     #   self.state['n_backwards'] += 1        
        # save the current parameters:
        params_current = copy.deepcopy(self.params)
        grad_current = get_grad_list(self.params) 
        # print("time for copies:", time.time()-start)
        # start = time.time()

       # grad_norm = [compute_grad_norm(grad) for grad in grad_current]
        # print("time for grad norm:", time.time()-start)
        # start = time.time()
        #  Gv options
        # =============
        if self.gv_option == 'per_param':
            # update gv
            for a, g in enumerate(grad_current):
                    if isinstance(g, torch.Tensor) and isinstance(self.state['gv'][a], torch.Tensor):
                        if g.device != self.state['gv'][a].device:
                            self.state['gv'][a] = self.state['gv'][a].to(g.device)
                    if isinstance(g, torch.Tensor) and isinstance(self.state['mv'][a], torch.Tensor):
                        if g.device != self.state['mv'][a].device:
                            self.state['mv'][a] = self.state['mv'][a].to(g.device)
                    if self.base_opt == 'adam':
                        self.state['gv'][a] = (1-self.beta)*(g**2) + (self.beta) * self.state['gv'][a]
                        self.state['mv'][a] = (1-self.momentum)*g + (self.momentum) * self.state['mv'][a]
                    if self.base_opt == 'lion':
                       # self.state['gv'][a][i] = (1-self.beta)*(g**2) + (self.beta) * self.state['gv'][a][i]
                        self.state['mv'][a] = (1-self.beta)*g + (self.beta) * self.state['mv'][a]

                    # else:
                    #     raise ValueError('%s does not exist' % self.base_opt)
            
            momentum_vector = torch.cat([torch.flatten(mv) for mv in self.state['mv']])
            grad_vector = torch.cat([torch.flatten(mv) for mv in grad_current])
            cosine_similarity = F.cosine_similarity(momentum_vector, grad_vector, 0,  1e-8)
          #  print("cosine similarity:", cosine_similarity)
            self.state["cosine_similarity"]  = cosine_similarity
            self.state["ema_cosine_similarity"] = (1-self.beta_s)*cosine_similarity + self.beta_s * self.state["ema_cosine_similarity"]
         #   print("ema cosine similarity:", self.state["ema_cosine_similarity"])
        # print("time for gv and mv calcs:", time.time()-start)
        # start = time.time()
        # if self.base_opt == "scalar":
        #     pp_norm = grad_norm
        # else:
        #     pp_norm =[self.get_pp_norm(g_cur,i) for i,g_cur in enumerate(grad_current)]
        step_size = self.step_size

        # compute step size and execute step
        # =================
      #  print("at step:",self.state['step'])
        

     #   self.state['gradient_norm'] =  [a.item() for a in pp_norm]
     #   self.pp_norm = pp_norm
        #    self.avg_gradient_norm_scaled[i] = self.avg_gradient_norm[i]/(1-self.beta)**(self.state['step']+1)
        if self.smooth == False and not self.smooth_after == 0:
            if self.state['step'] > self.smooth_after:
                self.smooth = True

        if self.first_step:
            # step_size, loss_next = self.line_search(step_sizes[0], params_current, grad_current, loss, closure_deterministic,  precond=True)
            # step_sizes = [step_size for i in range(len(step_sizes))]
            # self.try_sgd_precond_update(self.params, step_size, params_current, grad_current, self.momentum)
            if self.state['step'] > 5:
                self.first_step = False
                # for i in range(len(self.avg_gradient_norm)):
                #     self.avg_gradient_norm[i] = self.avg_gradient_norm[0]
                #     self.avg_decrease[i] = self.avg_decrease[-1]   

        step_size , loss_next = self.line_search(step_size, params_current, grad_current, loss, closure_deterministic, precond=True)
        #print(self.state["ema_cosine_similarity"]/(1-self.beta_s**(self.state['step']+1)))#*self.state["ema_cosine_similarity"]/((1-self.beta_s)**(self.state['step']+1))
        #print(self.state["ema_cosine_similarity"]/(1-self.beta_s**(self.state['step']+1)))
         #step_size*self.state["ema_cosine_similarity"]/(1-self.beta_s**(self.state['step']+1))
        self.try_sgd_precond_update(self.params,step_size, params_current, grad_current, self.momentum)

        # for i in range(len(self.avg_step_size)):
        #     if i == self.nextcycle:
        #         self.avg_step_size[i] = self.avg_step_size[i] * self.beta_s + (step_size[i]) *(1-self.beta_s)
        
       # print(step_sizes)
        self.step_size = step_size
        self.save_state(step_size, loss, loss_next)

        if torch.isnan(self.params[0]).sum() > 0:
            raise ValueError('nans detected')

        return loss

 

    @torch.no_grad()
    def try_sgd_precond_update(self, params, step_size, params_current, grad_current, momentum):
        if self.gv_option in ['scalar']:
            zipped = zip(params, params_current, grad_current)

            for p_next, p_current, g_current in zipped:
                p_next.data[:] = p_current.data
                p_next.data.add_(g_current, alpha=- step_size)
        
        elif self.gv_option == 'per_param':
            if self.base_opt == 'adam':
                zipped = zip(params, params_current, grad_current, self.state['gv'], self.state['mv'])
                for p_next, p_current, g_current, gv_i, mv_i in zipped:
                    gv_i_scaled = scale_vector(gv_i, self.beta, self.state['step']+1)
                    pv_list = 1. / (torch.sqrt(gv_i_scaled) + 1e-8)

                    if momentum == 0. or  self.mom_type == 'heavy_ball':
                        mv_i_scaled = g_current
                    elif self.mom_type == 'standard':
                        mv_i_scaled = scale_vector(mv_i, momentum, self.state['step']+1)
                    
                    p_next.data[:] = p_current.data
                    p_next.data.add_((pv_list *  mv_i_scaled), alpha=- step_size)
            elif self.base_opt == 'lion':
                zipped = zip(params, params_current, grad_current, self.state['mv'])
                for p_next, p_current, g_current, mv_i in zipped:
                    dk = torch.sign(momentum * mv_i + g_current * (1-momentum))
                    p_next.data[:] = p_current.data
                    p_next.data.add_(dk, alpha=- step_size)
            
            

            else:
                raise ValueError('%s does not exist' % self.base_opt)

        else:
            raise ValueError('%s does not exist' % self.gv_option)

def scale_vector(vector, alpha, step, eps=1e-8):
    scale = (1-alpha**(max(1, step)))
    return vector / scale

