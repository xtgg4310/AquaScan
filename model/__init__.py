from model.classify.resnet_uni import resnet_linear
import torch.nn as nn


def select_model(args):
    if args.model_type == 'res18_linear':
        model = resnet_linear(in_channel=args.channel_num, num_classes=args.num_classes, name='resnet18')
    elif args.model_type == 'res34_linear':
        model = resnet_linear(in_channel=args.channel_num, num_classes=args.num_classes, name='resnet34')
    else:
        raise NotImplementedError
    return model

def equip_multi_gpu(model, args):
    model = nn.DataParallel(model, device_ids=args.gpus)
    return model
