"""Deterministic group sampling with measured character budgets and no repeats."""
import hashlib,math
from pathlib import Path
from data_pipeline.contracts import sha256_file
from data_pipeline.source_policy import base_rejection
ALIASES={'bewikisource_full':'bewikisource','bewikibooks_full':'bewikibooks','belarusian_seed':'bootstrap'}

def load_policy(path):
    import yaml
    path=Path(path);policy=yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(policy,dict) or not isinstance(policy.get('sources'),dict):raise ValueError('invalid mixture policy')
    for name,rule in policy['sources'].items():
        weight=rule.get('source_weight')
        if isinstance(weight,bool) or not isinstance(weight,(int,float)) or not math.isfinite(weight) or not 0<=weight<=1:
            raise ValueError(f'{name}: source_weight must be finite in [0,1] (sampling without replacement)')
    for name in ('synthetic_cap','tarask_ratio_max'):
        if not 0<policy.get(name,0)<1:raise ValueError(f'{name} must be in (0,1)')
    policy['sha256']=sha256_file(path)
    return policy

def source_rule(policy,source):
    rule=policy['sources'].get(ALIASES.get(source,source))
    if rule is None:raise ValueError('source missing from mixture policy: '+source)
    return rule

def priority(group):return hashlib.sha256(('belka-mixture-v2\0'+group).encode()).hexdigest()

def select_groups(groups,policy):
    """groups: id -> source/chars/orthography. Fractions are CHARS, not tokens.

    source_weight is deterministic retention probability, not a promised final
    distribution. Hard synthetic/tarask maxima are then enforced on actual chars.
    """
    selected={};excluded={}
    for key,g in groups.items():
        rule=source_rule(policy,g['source']);weight=rule['source_weight']
        if base_rejection(g['source']) or int(priority(key)[:16],16)/(1<<64)>=weight:
            excluded[key]='source_weight';continue
        selected[key]=g
    for label,cap,predicate in (
        ('synthetic_cap',policy['synthetic_cap'],lambda g:source_rule(policy,g['source']).get('synthetic',False)),
        ('tarask_cap',policy['tarask_ratio_max'],lambda g:g.get('orthography')=='tarask')):
        targets=[k for k,g in selected.items() if predicate(g)]
        other=sum(g['chars'] for g in selected.values() if not predicate(g));budget=other*cap/(1-cap);used=0
        for key in sorted(targets,key=priority):
            if used+selected[key]['chars']<=budget:used+=selected[key]['chars']
            else:excluded[key]=label;del selected[key]
    return set(selected),excluded
