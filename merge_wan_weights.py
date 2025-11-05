import torch
from ultrai2v.modules.want2v import WanModel

orig_path = "/mnt/data2/Wan2.1-T2V-14B"
save_path = "./want2v_14b.pt"
model = WanModel.from_pretrained(orig_path)
state_dict = model.state_dict()
torch.save(state_dict, save_path)