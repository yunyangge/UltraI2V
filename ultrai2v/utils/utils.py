import torch

def str_to_precision(s):
    if s == "bfloat16" or s == "bf16":
        return torch.bfloat16
    elif s == "float16" or s == "fp16":
        return torch.float16
    elif s == "float32" or s == "float" or s == "fp32":
        return torch.float32
    else:
        raise ValueError(f"Unsupported precision string: {s}")