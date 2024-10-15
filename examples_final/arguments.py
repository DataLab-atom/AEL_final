import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='GcnInformax Arguments.')
    parser.add_argument('--DS', dest='DS', help='Dataset',default="COLLAB")
    parser.add_argument('--local', dest='local', action='store_const', 
            const=True, default=False)
    parser.add_argument('--glob', dest='glob', action='store_const', 
            const=True, default=False)
    parser.add_argument('--prior', dest='prior', action='store_const', 
            const=True, default=False)

    parser.add_argument('--lr', dest='lr', type=float,
            help='Learning rate.', default=0.01)
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int, default=3,
            help='Number of graph convolution layers before each pooling')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=32,
            help='')
    parser.add_argument('--aug-point-path', dest='aug_point_path', type=str, default='/home/xtanghao/THPycharm/AEL_main/examples/unsupervised_TU_PROTEINS/ael_results/pops_best/9_population_generation_1.json',
            help='')

    parser.add_argument('--aug', type=str, default='minmax')
    parser.add_argument('--gamma', type=str, default=0.1)
    parser.add_argument('--mode', type=str, default='fast')
    parser.add_argument('--seed', type=int, default=0)

    return parser.parse_args()

