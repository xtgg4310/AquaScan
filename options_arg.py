"""
This file is not used now. If you prefer to use argparse, you should modify this file for some parameters.
This file controls all parameters, paths and models in project.
parse_common_args is a global config and used by both train and test.
parse_train_args and parse_test_args control train and test separately.
"""
import argparse
import os, math


def parse_common_args(parser):
    parser.add_argument('--model_type', type=str, default='classify', help='change map in model._init__.py')
    parser.add_argument('--data_type', type=str, default='classify', help='change map in data.__init__.py')
    parser.add_argument('--save_prefix', type=str, default='pref', help='can be the name of this experiment')
    parser.add_argument('--load_model_path', type=str, default='checkpoints/base_model_pref/0.pth',
                        help='model path for pretrain or test')
    parser.add_argument('--load_not_strict', action='store_true', help='NOT USED: allow to load only common state dicts')
    parser.add_argument('--val_list', type=str, default='/data/dataset1/list/classify/val.txt',
                        help='it is val list in train, and test list path in test')
    parser.add_argument('--gpus', nargs='+', type=int)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--channel_num', type=int, default=3)
    parser.add_argument('--wandb', action='store_true', help='use wandb or not')
    parser.add_argument('--wandb_sweep_path', type=str, default='file path to sweep yaml file')
    return parser


def parse_train_args(parser):
    parser = parse_common_args(parser)

    # optimization
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='200,300,400',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--beta', type=float, default=0.999, help='beta for adam')
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    parser.add_argument('--loss', default='ce')
    parser.add_argument('--temperature', type=float, default=0.07, help='temperature for contrastive loss')
    parser.add_argument('--model_dir', type=str, default='', help='leave blank, auto generated')
    parser.add_argument('--train_list', type=str, default='/data/dataset1/list/classify/train.txt')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--augment_type', type=str, default='none')
    parser.add_argument('--epochs', type=int, default=500)
    return parser


def parse_test_args(parser):
    parser = parse_common_args(parser)
    parser.add_argument('--save_viz', action='store_true', help='save visualize result in eval or not')
    parser.add_argument('--result_dir', type=str, default='', help='leave blank, auto generated')
    return parser


def get_train_args():
    parser = argparse.ArgumentParser()
    parser = parse_train_args(parser)
    args = parser.parse_args()
    return args


def get_test_args():
    parser = argparse.ArgumentParser()
    parser = parse_test_args(parser)
    args = parser.parse_args()
    return args


def get_train_model_dir(args):
    model_dir = os.path.join('checkpoints', args.model_type + '_' + args.save_prefix)
    if not os.path.exists(model_dir):
        os.system('mkdir -p ' + model_dir)
    args.model_dir = model_dir


def get_test_result_dir(args):
    ext = os.path.basename(args.load_model_path).split('.')[-1]
    model_dir = args.load_model_path.replace(ext, '')
    val_info = os.path.basename(os.path.dirname(args.val_list)) + '_' + os.path.basename(args.val_list.replace('.txt', ''))
    result_dir = os.path.join(model_dir, val_info + '_' + args.save_prefix)
    if not os.path.exists(result_dir):
        os.system('mkdir -p ' + result_dir)
    args.result_dir = result_dir


def save_args(args, save_dir):
    args_path = os.path.join(save_dir, 'args.txt')
    with open(args_path, 'w') as fd:
        fd.write(str(args).replace(', ', ',\n'))


def prepare_train_args():
    args = get_train_args()
    get_train_model_dir(args)
    save_args(args, args.model_dir)
    # set learning rate decay
    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))

    # warmup for large batch size
    if args.batch_size > 256:
        args.warm = True
    if args.warm:
        args.warmup_from = 0.01
        args.warm_epochs = 10
    if args.cosine:
        eta_min = args.lr * (args.lr_decay_rate ** 3)
        args.warmup_to = eta_min + (args.lr - eta_min) * (1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2
    else:
        args.warmup_to = args.lr
    return args


def prepare_test_args():
    args = get_test_args()
    get_test_result_dir(args)
    save_args(args, args.result_dir)
    return args


if __name__ == '__main__':
    train_args = get_train_args()
    test_args = get_test_args()
