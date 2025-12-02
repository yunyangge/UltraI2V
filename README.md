# A script for video training with native torch implementation
## 😊Code Environment
1. Conda env creation
```
conda create -n torchvideo python=3.10
```
2. Install repos
```
pip install -r requirements.txt
```
3. Pip editable mode
```
pip install -e .
```
## 🧑‍🏭Training
1. Modify the config file
In `config/`, find the config you need and modify it.
For example, training text-to-video on WanT2V 14B, you should modify configs/train/xpu/t2v_14b.yaml
```
# test yaml
model_name: "wan_t2v" # model_name, default value is Wan T2V
seed: 1024

output_dir: "output/t2v" # dir for saving checkpoints

training_iteration: 1000000 # max training steps
# ddp_size: 1 
fsdp_size: 8 # fsdp size. When fsdp_size == world_size, we use full sharding fsdp; when fsdp_size < world_size and world_size % fsdp_size == 0, we use HSDP (aka, using ddp and fsdp simultaneously)
cp_size: 2 # DeepSpeed-Ulysses context parallel size
use_context_parallel: True # only use_context_parallel == True and cp_size > 1, we use context parallel
reshard_after_forward: null # whether gather params after fsdp forward.
gradient_checkpointing: True 
gradient_accumulation_steps: 1
init_max_grad_norm: 1.0 # initial grad norm threshold value. Because we use adaptive grad clipping, the grad norm threshold will change with training steps.
log_interval: 1
save_interval: 1000
weight_dtype: "bf16"
ema_decay: 0.9999
ema_update_interval: 1
save_with_dcp_api: True

wandb_config:
  project_name: "t2v"
  exp_name: "t2v"

model_config:
  pretrained_model_dir_or_checkpoint: "/mnt/data2/Wan2.1-T2V-14B/"

scheduler_config:
  scheduler_name: "flow_matching"
  use_dynamic_shifting: True
  use_logitnorm_time_sampling: True

vae_config:
  vae_path: "/mnt/data2/Wan2.1-T2V-14B/Wan2.1_VAE.pth"
  dtype: "fp32"

text_encoder_config:
  text_len: 512
  checkpoint_path: "/mnt/data2/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth"
  use_fsdp: True

data_config:
  batch_size: 1
  num_workers: 16
  pin_memory: False
  drop_last: True
  shuffle: True
  dataset_name: "t2v_random"
  dataset_config:
    text_tokenizer_path: "/mnt/data2/Wan2.1-T2V-14B/google/umt5-xxl"
    sample_height: 480
    sample_width: 832
    sample_num_frames: 49
    tokenizer_max_length: 512
    return_prompt_mask: True
  sampler_name: "stateful_distributed"
  collator_name: "wan_t2v"
  
optimizer_config:
  lr: 0.00002
  weight_decay: 0.01
```
2. Run the script
```
bash scripts/train/xpu/train_t2v_14b.sh
```
## 💫Inference
1. Modify the config file
In `config/`, find the config you need and modify it.
For example, training text-to-video on WanT2V 14B, you should modify configs/infer/xpu/t2v_14b.yaml
```
# test yaml

model_name: "wan_t2v"
pipeline_name: "t2v"
seed: 1024

prompt_txt: "assets/t2v/simple_prompts.txt" # prompt
output_dir: "samples/wan_t2v" # save dir

num_frames: 49
height: 480
width: 832
save_fps: 16
batch_size: 1

fsdp_size: 8
cp_size: 8
use_context_parallel: True
reshard_after_forward: False
weight_dtype: "bf16"
save_with_dcp_api: False

model_config:
  dim: 5120
  ffn_dim: 13824
  freq_dim: 256
  in_dim: 16
  num_heads: 40
  num_layers: 40
  out_dim: 16
  text_len": 512
  pretrained_model_dir_or_checkpoint: "/mnt/data2/Wan2.1-T2V-14B/"

scheduler_config:
  scheduler_name: "flow_matching"
  num_inference_steps: 50
  shift: 7.0
  guidance_scale: 5.0

vae_config:
  vae_path: "/mnt/data2/Wan2.1-T2V-14B/Wan2.1_VAE.pth"
  dtype: "fp32"

text_encoder_config:
  text_len: 512
  checkpoint_path: "/mnt/data2/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth"
  text_tokenizer_path: "/mnt/data2/Wan2.1-T2V-14B/google/umt5-xxl"
  use_fsdp: True
```
2. Run the script
```
bash scripts/infer/xpu/train_t2v_14b.sh
```
