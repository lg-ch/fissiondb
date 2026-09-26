"""Product maintenance commands; each operation reports failures explicitly."""
import argparse
import json
import time


def main():
    parser=argparse.ArgumentParser(prog='fissiondb-engine-python')
    sub=parser.add_subparsers(dest='command',required=True)
    convert=sub.add_parser('convert',help='Resumable residual conversion for dimensions 1..1024')
    convert.add_argument('--index',required=True)
    convert.add_argument('--base',required=True)
    convert.add_argument('--output',required=True)
    args=parser.parse_args()
    from .anchors import AnchorIndex
    started=time.perf_counter()
    with AnchorIndex(args.index,args.base) as index:
        output=index.build_residual(args.output)
    print(json.dumps({'output':str(output),'seconds':time.perf_counter()-started,
                      'bytes':(output/'res512.bin').stat().st_size}))


if __name__=='__main__':
    main()
