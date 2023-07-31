import copy
import time
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import wandb
from .ken_base import StochLineSearchBase, get_grad_list, compute_grad_norm, random_seed_torch, try_sgd_update
from PIL import Image

#gets a nested list of parameters as input
class KenSLS(StochLineSearchBase):
    def __init__(self,
                 params,
                 n_batches_per_epoch=500,
                 init_step_size=0.1,
                 c=0.1,
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
                 only_decrease = False, log = False):
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
        self.log = log
        # self.state['step_size'] = init_step_size

      #  self.avg_decrease = torch.zeros(len(params))#(0.0 for i in range(len(params))]
       # self.avg_step_size = torch.zeros(len(params))#[0.0 for i in range(len(params))]
      #  self.avg_gradient_norm_scaled = torch.zeros(len(params))#[0.0 for i in range(len(params))]

        self.clip_grad = clip_grad
        self.gv_option = gv_option
        self.base_opt = base_opt
        # self.step_size_method = step_size_method
        self.tslls = 100
        self.avg_step_size_slow = 1e-8
        self.avg_step_size_fast = 1e-8
        self.avg_step_size  = 1e-8
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
            
          #  momentum_vector = torch.cat([torch.flatten(mv) for mv in self.state['mv']])
           # grad_vector = torch.cat([torch.flatten(mv) for mv in grad_current])
         #   cosine_similarity = F.cosine_similarity(momentum_vector, grad_vector, 0,  1e-8)
          #  print("cosine similarity:", cosine_similarity)
          #  self.state["cosine_similarity"]  = cosine_similarity
          #  self.state["ema_cosine_similarity"] = (1-self.beta_s)*cosine_similarity + self.beta_s * self.state["ema_cosine_similarity"]
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

      #  if self.first_step:
            # step_size, loss_next = self.line_search(step_sizes[0], params_current, grad_current, loss, closure_deterministic,  precond=True)
            # step_sizes = [step_size for i in range(len(step_sizes))]
            # self.try_sgd_precond_update(self.params, step_size, params_current, grad_current, self.momentum)
           # if self.state['step'] > 1:
              #  self.first_step = False
                # for i in range(len(self.avg_gradient_norm)):
                #     self.avg_gradient_norm[i] = self.avg_gradient_norm[0]
                #     self.avg_decrease[i] = self.avg_decrease[-1]   

      #  self.tslls = 1
        # if self.state['step'] < 10:
        #     step_size , loss_next = self.line_search(step_size, params_current, grad_current, loss, closure_deterministic, precond=True)
        #     self.avg_step_size_slow = self.avg_step_size_slow * 0.99 + (step_size) *(1-0.99)
        #     self.avg_step_size_fast = self.avg_step_size_fast * 0.9 + (step_size) *(1-0.9)
        # else:
      #  print("avg step size slow:", (self.avg_step_size_slow))#/((1-0.99)**(self.state['step']+1))))
     #   print("avg step size fast:", (self.avg_step_size_fast)) # /((1-0.9)**(self.state['step']+1))))

     #efficient training
      #  if not (self.state['step'] % 10 == 0 and self.log):
        # rate_of_change = self.avg_step_size_fast /self.avg_step_size_slow
        # if rate_of_change < 1:
        #     rate_of_change = 1/rate_of_change
        # neededchecks = min(10,1/((rate_of_change-1)+ 1e-8))
    #  print("needed checks:", neededchecks)
        
       # if self.tslls > neededchecks:
        g_norm =self.get_pp_norm(grad_current)
        #  explore_step_size = step_size if self.first_step else self.avg_step_size_fast/(1-0.9**(self.state['step']+1))
        #   print(explore_step_size)
        step_size , loss_next = self.line_search(step_size, params_current, grad_current,g_norm, loss, closure_deterministic, precond=True)
        #   print(step_size)
        # #  print(step_size)
        # self.tslls  = 1
        # # self.next_explore_step_size = self.next_explore_step_size * 0.99 + (step_size) *(1-0.95)
        # #  if step_size > 5e-8:
        # self.avg_step_size_slow = self.avg_step_size_slow * 0.99 + (step_size) *(1-0.99)
        # self.avg_step_size_fast = self.avg_step_size_fast * 0.9 + (step_size) *(1-0.9)
        # else:
        #     loss_next  = torch.Tensor([0])
        #     self.tslls += 1
        # # # slow but informative training
        # else:
#         with torch.no_grad():
#             loss_next  = torch.Tensor([0])
#             losses, step_sizes = self.basic_line_search(5e-3, params_current, grad_current, loss, closure_deterministic, precond=True)
            
#             lossesdec = [(loss - l).cpu().numpy() for l in losses]
#             losses = [ l.cpu().numpy() for l in losses]
#         next_pos = np.argmax(lossesdec)
#         while True:
#             if next_pos == len(lossesdec)-1:
#                 break
#             next_pos = next_pos + 1
#             if lossesdec[next_pos] < 0.9 * lossesdec[np.argmax(lossesdec)]:
#                 next_pos = next_pos - 1
#                 break
#         step_size = step_sizes[next_pos]
#         losses = np.asarray(losses) - losses[np.argmin(losses)]
#         lossesdec_scaled = np.asarray(lossesdec)/np.sqrt(np.asarray(step_sizes))
#         step_size2 = step_sizes[np.argmax(lossesdec_scaled)]

#         lossesdec_log_scaled = np.asarray(lossesdec)/np.power(np.asarray(step_sizes), (1.0/3.0))
#         step_size3 = step_sizes[np.argmax(lossesdec_log_scaled)]

#         #calculate gradient_norm
#       #  print(step_sizes)
#         c = 0.1
#         g_norm =self.get_pp_norm(grad_current).item()
#         index4 = [g-c*g_norm * s for g,s in zip(lossesdec, step_sizes)]
#         found = False
#         for i,a in enumerate(index4):
#             if a > 0:
#                 index4 = i
#                 found = True
#                 break
#         if not found:
#             index4 = len(index4)-1
#         step_size4 = step_sizes[index4]
#        # print(step_size4)
#        # print(step_sizes)
#         step_size4 = step_sizes[np.argmax(lossesdec[index4:])+index4]
#         loss4 = np.max(lossesdec[index4:])
#       #  print(step_size4)
#       #  print(g_norm)

#         #calculate momentum step size
#         self.avg_step_size = step_size4 * (1-0.9) + self.avg_step_size * 0.9
#         step_size5 = self.avg_step_size if self.avg_step_size < step_size4 else step_size4
#         losss5 = lossesdec[np.argmin([abs(s-step_size5) for s in step_sizes])]

#         plt.plot(step_sizes, lossesdec)#,'-gD', markevery = [next_pos])
#         plt.plot(step_sizes, [c*g_norm * s for s in step_sizes], c = "k")

#         plt.scatter(step_size, lossesdec[next_pos], c = 'r')
#         plt.scatter(step_size2, lossesdec[np.argmax(lossesdec_scaled)], c = 'g')
#         plt.scatter(step_size3, lossesdec[np.argmax(lossesdec_log_scaled)], c = 'y')
#         plt.scatter(step_size4, loss4, c = 'b')
#         plt.scatter(step_size5, losss5, c = 'm')
#        # print(step_size3,step_size4)
#       #  print(lossesdec[np.argmax(lossesdec_log_scaled)],0.3*g_norm*step_size4)
#         plt.xscale('log')
#         plt.ylim((-lossesdec[np.argmax(lossesdec)]*1.1,lossesdec[np.argmax(lossesdec)]*1.1))
#    #     plt.show()
#         plt.savefig("plots/losslandscape"+ str(self.state['step']) +".png")
#         img = Image.open("plots/losslandscape"+ str(self.state['step']) +".png")
#         img = wandb.Image(img, caption="loss landscape" + str(self.state['step']))
#         wandb.log({"loss landscape": img})
#         plt.clf()

        self.step_size = step_size
        self.try_sgd_precond_update(self.params,self.step_size, params_current, grad_current, self.momentum)

        
        self.save_state(self.step_size, loss, loss_next)

        if torch.isnan(self.params[0]).sum() > 0:
            raise ValueError('nans detected')
        
        if self.first_step:
            self.first_step = False

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

    def get_pp_norm(self, grad_current):
        if self.pp_norm_method in ['pp_armijo', "just_pp"]:
            pp_norm = 0
            for g_i, gv_i in zip(grad_current, self.state['gv']):
                if self.base_opt == 'adam':
                    gv_i_scaled = scale_vector(gv_i, self.beta, self.state['step']+1)
                    pv_i = 1. / (torch.sqrt(gv_i_scaled) + 1e-8)
                #    print(g_i.shape)
                    if self.pp_norm_method == 'pp_armijo':
                        layer_norm = ((g_i**2) * pv_i).sum()
                    elif self.pp_norm_method == "just_pp":
                        layer_norm = pv_i.sum()
                    else:
                        raise ValueError('%s not found' % self.base_opt)
                pp_norm += layer_norm
             #   pp_norm = torch.sqrt(pp_norm)
              #  pp_norms.append(layer_norm.item())

        else:
            raise ValueError('%s does not exist' % self.pp_norm_method)

        return pp_norm

def scale_vector(vector, alpha, step, eps=1e-8):
    scale = (1-alpha**(max(1, step)))
    return vector / scale

