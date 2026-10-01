"""Byte-weighted evaluation with finite-input checks and explicit denominators."""
import math


def wilson_interval(successes, total, z=1.959963984540054):
    if type(successes) is not int or type(total) is not int or not 0 <= successes <= total or total < 1:
        raise ValueError('require integer 0 <= successes <= total and total > 0')
    if not math.isfinite(z) or z <= 0: raise ValueError('z must be finite and positive')
    p=successes/total;denom=1+z*z/total
    center=(p+z*z/(2*total))/denom
    width=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/denom
    return max(0.,center-width),min(1.,center+width)


def evaluate_bpb(model, batches, steps, token_bytes):
    import torch
    import torch.distributed as dist
    if type(steps) is not int or steps < 1: raise ValueError('BPB requires a positive number of batches')
    device=model.get_device()
    dtype=torch.float32 if device.type=='mps' else torch.float64
    if token_bytes.ndim != 1 or token_bytes.device != device or bool((token_bytes < 0).any()):
        raise ValueError('token byte lengths must be a nonnegative vector on the model device')
    nats=torch.zeros((),dtype=dtype,device=device)
    byte_count=torch.zeros((),dtype=torch.int64,device=device)
    world=dist.get_world_size() if dist.is_initialized() else 1
    iterator=iter(batches);training=model.training
    model.eval()
    try:
        with torch.no_grad():
            for _ in range(steps):
                failure=None
                try:
                    x,y=next(iterator)
                    loss=model(x,y,loss_reduction='none').reshape(-1).to(dtype)
                    targets=y.reshape(-1)
                    valid=targets.int()>=0
                    safe=torch.where(valid,targets,torch.zeros_like(targets))
                    lengths=torch.where(valid,token_bytes[safe],torch.zeros_like(safe))
                    selected=torch.where(lengths>0,loss,torch.zeros_like(loss))
                    if not torch.isfinite(selected).all().item(): raise ValueError('non-finite evaluation loss')
                    nats += selected.sum();byte_count += lengths.sum()
                except Exception as exc:
                    failure=f'{type(exc).__name__}: {exc}'
                failed=torch.tensor(int(failure is not None),device=device,dtype=torch.int64)
                if world>1:dist.all_reduce(failed,op=dist.ReduceOp.MAX)
                if failed.item():raise ValueError('evaluation batch failed on at least one rank: '+str(failure))
            if world>1:
                dist.all_reduce(nats,op=dist.ReduceOp.SUM)
                dist.all_reduce(byte_count,op=dist.ReduceOp.SUM)
        if byte_count.item()<=0:raise ValueError('evaluation contains no unmasked text bytes')
        return nats.item()/(math.log(2)*byte_count.item())
    finally:
        model.train(training)
