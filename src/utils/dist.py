import torch.distributed as dist

def print_all(msg):
    if not dist.is_available():
        print(msg)
    elif dist.get_rank() % 8 == 0:
        print(f'{dist.get_rank() // 8}: {msg}')