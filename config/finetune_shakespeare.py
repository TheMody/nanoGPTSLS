import time

out_dir = 'out-shakespeare'
eval_interval = 20
eval_iters = 10
wandb_log = True # feel free to turn on
wandb_project = 'shakespeare'


dataset = 'shakespeare'
init_from = 'gpt2' # this is the largest GPT-2 model

wandb_run_name = 'ft-sgdsSls'+ init_from + str(time.time())

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
batch_size = 4
gradient_accumulation_steps = 8
max_iters = 200
warmup_iters = 0

# finetune at constant LR
learning_rate = 3e-3
decay_lr = False
