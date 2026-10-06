#!/usr/bin/env python3
"""Make missing CPU dependencies and an unexecuted HF roundtrip fail CI."""
import argparse
import xml.etree.ElementTree as ET

HARDWARE_SKIP_REASONS={'requires CUDA hardware','requires H200 hardware'}
REQUIRED={'test_native_export_roundtrip'}

def validate(path):
    cases=list(ET.parse(path).getroot().iter('testcase'))
    failures=[]
    executed=set()
    if not cases: failures.append('no test cases recorded')
    for case in cases:
        name=case.attrib.get('name','')
        skip=case.find('skipped')
        if skip is not None:
            reason=skip.attrib.get('message','')
            if reason not in HARDWARE_SKIP_REASONS:
                failures.append(f'unexpected skip: {name}: {reason}')
        elif case.find('failure') is None and case.find('error') is None:
            executed.add(name)
        else:
            failures.append(f'failed test: {name}')
    failures.extend(f'required CPU integration did not pass: {name}' for name in sorted(REQUIRED-executed))
    return failures

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('report')
    args=ap.parse_args(argv)
    failures=validate(args.report)
    if failures: ap.exit(1,'\n'.join(failures)+'\n')
    print('Required CPU integration passed; no unintended skips.')
    return 0

if __name__=='__main__': raise SystemExit(main())
