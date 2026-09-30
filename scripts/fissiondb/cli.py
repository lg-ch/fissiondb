"""Product maintenance commands; each operation reports failures explicitly."""
import argparse
import json
import time


def main():
    parser=argparse.ArgumentParser(prog='fissiondb-build')
    sub=parser.add_subparsers(dest='command',required=True)
    create=sub.add_parser('create',help='Create an empty collection with automatic cell fission')
    create.add_argument('--index',required=True)
    create.add_argument('--dim',required=True,type=int)
    create.add_argument('--cell-capacity',type=int,default=None,
                        help='Split threshold: automatic max(64, 2*input dimension) by default or with 0; override with 64..65536')
    create.add_argument('--max-cells',type=int,default=300_000)
    convert=sub.add_parser('convert',help='Resumable residual conversion for dimensions 1..1024')
    convert.add_argument('--index',required=True)
    convert.add_argument('--base',required=True)
    convert.add_argument('--output',required=True)
    args=parser.parse_args()
    from .anchors import AnchorIndex
    started=time.perf_counter()
    if args.command=='create':
        with AnchorIndex.create(args.index,args.dim,cell_capacity=args.cell_capacity,max_cells=args.max_cells) as index:
            print(json.dumps({'directory':str(index.directory),'dim':index.dim,'count':index.count,
                              'cell_capacity':index.fission_capacity,'fission':index.fission_stats}))
        return
    with AnchorIndex(args.index,args.base) as index:
        output=index.build_residual(args.output)
    print(json.dumps({'output':str(output),'seconds':time.perf_counter()-started,
                      'bytes':(output/'res512.bin').stat().st_size}))


if __name__=='__main__':
    main()
