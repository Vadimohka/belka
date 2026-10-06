"""Reference-correct SDPA fallback with bottom-right causal alignment."""
import torch
import torch.nn.functional as F


def sdpa_attention(q,k,v,window_size=(-1,-1),enable_gqa=False,causal=True):
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or k.shape != v.shape:
        raise ValueError('expected compatible (B,H,T,D) q/k/v tensors')
    if q.device != k.device or q.device != v.device:
        raise ValueError('q/k/v devices differ')
    k,v = k.to(q.dtype),v.to(q.dtype)
    Tq,Tk = q.size(2),k.size(2)
    if Tq<1 or Tk<Tq: raise ValueError('query/key lengths cannot be empty or negatively aligned')
    left,right=window_size
    if left < -1 or right < -1: raise ValueError('invalid attention window')
    if (left<0 or left>=Tk-1) and (causal or right<0 or right>=Tk-1) and Tq==Tk:
        return F.scaled_dot_product_attention(q,k,v,is_causal=causal,enable_gqa=enable_gqa)
    row=(Tk-Tq)+torch.arange(Tq,device=q.device).unsqueeze(1)
    col=torch.arange(Tk,device=q.device).unsqueeze(0)
    mask=torch.ones((Tq,Tk),dtype=torch.bool,device=q.device)
    if causal: mask &= col<=row
    if left>=0: mask &= row-col<=left
    if right>=0: mask &= col-row<=right
    return F.scaled_dot_product_attention(q,k,v,attn_mask=mask,enable_gqa=enable_gqa)


def fallback_with_kvcache(q,k_cache,v_cache,k=None,v=None,cache_seqlens=None,causal=False,window_size=(-1,-1)):
    if q.ndim != 4 or k_cache.ndim != 4 or v_cache.ndim != 4:
        raise ValueError('q and cache must have four dimensions')
    B,T,H,D=q.shape
    if (k_cache.shape != v_cache.shape or k_cache.size(0)!=B or k_cache.size(3)!=D
            or B < 1 or T < 1 or k_cache.size(2)<1 or H % k_cache.size(2)
            or q.device!=k_cache.device or q.device!=v_cache.device):
        raise ValueError('invalid KV cache shape')
    if (k is None)!=(v is None): raise ValueError('k and v must be supplied together')
    if cache_seqlens is None or cache_seqlens.numel()!=B:
        raise ValueError('cache_seqlens must contain one position per row')
    positions=cache_seqlens.detach().cpu().tolist()
    if any(type(p) is not int or p<0 or p+(T if k is not None else 0)>k_cache.size(1) for p in positions):
        raise ValueError('KV write exceeds cache bounds')
    if k is not None and (k.shape!=v.shape or k.shape[:2]!=(B,T) or k.shape[2:]!=k_cache.shape[2:]):
        raise ValueError('new KV shape differs from cache')
    if any(p + (T if k is not None else 0) < T for p in positions):
        raise ValueError('not enough keys for query positions')
    if any(type(w) is not int or w < -1 for w in window_size) or len(window_size) != 2:
        raise ValueError('invalid attention window')
    if k is not None and (k.device != q.device or v.device != q.device):
        raise ValueError('new KV devices differ')
    outputs=[]
    for i,pos in enumerate(positions):
        end=pos+(T if k is not None else 0)
        if end<T: raise ValueError('not enough keys for query positions')
        if k is not None:
            k_cache[i,pos:end]=k[i];v_cache[i,pos:end]=v[i]
        qi=q[i:i+1].transpose(1,2)
        ki=k_cache[i:i+1,:end].transpose(1,2)
        vi=v_cache[i:i+1,:end].transpose(1,2)
        outputs.append(sdpa_attention(qi,ki,vi,window_size,H!=ki.size(1),causal).transpose(1,2))
    return torch.cat(outputs,dim=0)
