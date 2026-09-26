"""Add only io_uring syscalls to an explicitly supplied default-deny profile."""
import argparse,json
from pathlib import Path

def extend(profile):
    if profile.get('defaultAction')!='SCMP_ACT_ERRNO' or not isinstance(profile.get('syscalls'),list):
        raise ValueError('Expected a default-deny Docker seccomp profile')
    calls={'io_uring_setup','io_uring_enter','io_uring_register'}
    result=dict(profile);rules=[]
    for rule in profile['syscalls']:
        copy=dict(rule);copy['names']=[name for name in rule['names'] if name not in calls]
        if copy['names']:rules.append(copy)
    rules.append(dict(names=sorted(calls),action='SCMP_ACT_ALLOW'))
    result['syscalls']=rules
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source');p.add_argument('output');a=p.parse_args()
    profile=extend(json.loads(Path(a.source).read_text(encoding='utf-8')))
    with Path(a.output).open('x',encoding='utf-8') as f:json.dump(profile,f,indent=2);f.write('\n')
