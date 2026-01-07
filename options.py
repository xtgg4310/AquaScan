import json
import os, math
import argparse
from dataclasses import dataclass, field
import yaml
from typing import List, Dict

@dataclass
class Option:
    # General
    model_type: str = 'res18_linear' #'res18_linear'
    data_type: str = 'classify'
    save_prefix: str = 'pref'
    note: str = ''
    load_model_path: str = ''
    load_not_strict: bool = False
    val_list: str = ''
    gpus: List[int] = field(default_factory=list)
    seed: int = 0
    num_classes: int = 5
    channel_num: int = 3
    hidden_size: int=1024
    features_size: int=32
    num_layers: int=7
    enable_wandb: bool = True
    wandb_sweep_path: str = ''
    train: bool = True
    shuffle: bool = True
    infe: bool=False
    label2id: Dict[str, int] = None

    # For train
    lr: float = 1e-3
    lr_decay_epochs: List[int] = field(default_factory=list)
    lr_decay_rate: float = 0.1
    weight_decay: float = 1e-4
    momentum: float = 0.9
    beta: float = 0.999
    cosine: bool = False
    warm: bool = False
    warm_epochs: int = 10
    loss: str = 'ce'
    temperature: float = 0.07
    model_dir: str = ''
    train_list: str = ''
    batch_size: int = 128
    epochs: int = 500
    augment_type: str = 'none'
    print_freq: int = 100
    freeze_layers: List[str] = field(default_factory=list)

    # For test
    save_vis: bool = False
    vis_freq: int = 100
    result_dir: str = ''

    lstm_input_size: int = 256
    lstm_hidden_size: int = 128
    lstm_num_layer: int = 2


def get_train_model_dir(args):
    model_dir = os.path.join('checkpoints', args.model_type + '_' + args.save_prefix)
    if args.model_dir == '':
        args.model_dir = model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    else:
        model_dir = os.path.join(args.model_dir)
    with open(os.path.join(model_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    with open(os.path.join(model_dir, 'args.yaml'), 'w') as f:
        yaml.dump(args.__dict__, f, indent=2)

def get_train_result_dir(args):
    # get model_dir
    ext = os.path.basename(args.load_model_path).split('.')[-1]
    model_dir = args.model_dir.replace('.' + ext, '')
    
    val_info = os.path.basename(os.path.dirname(os.path.abspath(args.val_list))) + '_' + os.path.basename(args.val_list.replace('.txt', ''))
    result_dir = os.path.join(model_dir, val_info + '_' + args.save_prefix)
    if args.result_dir == '':
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        args.result_dir = result_dir
    else:
        result_dir = args.result_dir
    with open(os.path.join(result_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    with open(os.path.join(result_dir, 'args.yaml'), 'w') as f:
        yaml.dump(args.__dict__, f, indent=2)

def get_test_result_dir(args):
    # get model_dir
    ext = os.path.basename(args.load_model_path).split('.')[-1]
    model_dir = args.load_model_path.replace('.' + ext, '')
    
    val_info = os.path.basename(os.path.dirname(os.path.abspath(args.val_list))) + '_' + os.path.basename(args.val_list.replace('.txt', ''))
    print(model_dir,val_info,args.save_prefix)
    result_dir = os.path.join(model_dir, val_info + '_' + args.save_prefix)
    #print(result_dir)
    if args.result_dir == '':
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        args.result_dir = result_dir
    else:
        result_dir = args.result_dir
    with open(os.path.join(result_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    with open(os.path.join(result_dir, 'args.yaml'), 'w') as f:
        yaml.dump(args.__dict__, f, indent=2)


def prepare_train_args(args):
    args.lr = float(args.lr)
    args.weight_decay = float(args.weight_decay)
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


def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_type', type=str, default='yaml', choices=['yaml', 'json'])
    parser.add_argument('--option_path', type=str, required=True, 
                        help='path of the option json file')
    # parser.add_argument('--model',type=str,required=True)
    fargs = parser.parse_args()

    file_path = fargs.option_path
    # Read the json file and store it in the config dictionary
    if fargs.file_type == 'yaml':
        with open(file_path, 'r') as f:
            args = yaml.load(f, Loader=yaml.FullLoader)
        args = Option(**args)
        # args.model_type=fargs.model
    else:
        with open(file_path, 'w') as f:
            args = json.load(f)
        args = Option(**args)
        # args.model_type=fargs.model
    if args.label2id is None:
        args.label2id = {
            "stand": 0,
            "struggle": 1,
            "float": 2,
            "sinking": 3,
            "sink": 3,
            "swim": -1,
            "others": 4,
            "other": 4,
            "noise": 4,
            "abnormal": -1,
        }
    if args.train:
        assert args.train_list != '', 'train_list should be provided'
        args.train_list = os.path.abspath(args.train_list)
        prepare_train_args(args)
        get_train_model_dir(args)
        get_train_result_dir(args)
    else:
        assert args.load_model_path != '', 'load_model_path should be provided'
        assert args.val_list != '', 'val_list should be provided'
        args.load_model_path = os.path.abspath(args.load_model_path)
        args.val_list = os.path.abspath(args.val_list)
        get_test_result_dir(args)
    return args
