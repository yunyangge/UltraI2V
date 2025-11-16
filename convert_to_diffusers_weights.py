import torch
from ultrai2v.modules.flashi2v import FlashI2VModel

orig_weights_path = 'want2v_14b.pt'
save_path = 'diffusers_weights'
state_dict = torch.load(orig_weights_path, map_location='cpu')
config = {
  'dim': 5120,
  'ffn_dim': 13824,
  'freq_dim': 256,
  'in_dim': 16,
  'num_heads': 40,
  'num_layers': 40,
  'out_dim': 16,
  'text_len': 512,
  'low_freq_energy_ratio': 0.1,
  'fft_return_abs': True,
  'conv3x3x3_proj': False,
}
model = FlashI2VModel(**config)
missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
model.save_pretrained(save_path)