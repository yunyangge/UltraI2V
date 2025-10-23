import torch
import torch.distributed as dist

class EncoderCacheManager:
    def __init__(self, tp_cp_group: dist.ProcessGroup = None):
        self.tp_cp_group = tp_cp_group
        self.tp_cp_size = dist.get_world_size(group=tp_cp_group) if tp_cp_group is not None else 1

        self.vae_cache = None
        self.text_cache = None

    def save_cache(self, vae_latents_list, text_embeds_list):
        self.vae_cache = vae_latents_list
        self.text_cache = text_embeds_list

    def get_cache_from_rank(self, src_rank=0):
        if self.vae_cache is None or self.text_cache is None:
            return ValueError("Cache is empty!")
        rank = dist.get_rank(group=self.tp_cp_group) if self.tp_cp_group is not None else 0
        if rank == src_rank:
            returned = [self.vae_cache, self.text_cache]
        else:
            returned = [None, None]
        dist.broadcast_object_list(returned, src=src_rank, group=self.tp_cp_group)
        return returned[0], returned[1]

    def __call__(self, vae_latents_list, text_embeds_list, step):
        if self.tp_cp_size <= 1:
            return vae_latents_list, text_embeds_list
        if step % self.tp_cp_size == 0:
            self.save_cache(vae_latents_list, text_embeds_list)
        return self.get_cache_from_rank(src_rank=step % self.tp_cp_size)


if __name__ == "__main__":
    from ultrai2v.distributed.utils import setup_distributed_env, cleanup_distributed_env
    setup_distributed_env()
    manager = EncoderCacheManager(tp_cp_group=dist.group.WORLD)
    rank = dist.get_rank()
    vae_data, text_data = [torch.tensor([range(rank, rank + 5)])], [torch.tensor([range(rank + 5, rank + 10)])]
    print(f"Rank {rank} original data: vae {vae_data}, text {text_data}")
    for step in range(4):
        vae_data, text_data = manager(vae_data, text_data, step)
        print(f"Rank {rank} step {step} received data: vae {vae_data}, text {text_data}")
    cleanup_distributed_env()