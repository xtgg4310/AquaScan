import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import sys
import tracemalloc
import sklearn.metrics as metric
#import wandb

from data import select_train_loader, select_eval_loader
from model import select_model
from options import get_options
from utils.logger import Logger, AverageMeter
import utils.torch_utils as torch_utils

import os
import psutil
import warnings
warnings.filterwarnings("ignore")

def log_device_usage(count, use_cuda):
    mem_Mb = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    cuda_mem_Mb = torch.cuda.memory_allocated(0) / 1024 ** 2 if use_cuda else 0
    print(f"iter {count}, mem: {int(mem_Mb)}Mb, gpu mem:{int(cuda_mem_Mb)}Mb")
    
    
class Trainer:
    def __init__(self):
        args = get_options()
        self.args = args
        torch.manual_seed(args.seed)
        self.logger = Logger(self.args)

        self.train_loader = select_train_loader(self.args)
        self.val_loader = select_eval_loader(self.args)

        self.criterion = self.get_criterion()
        #self.max_trqain
        self.enable_cuda = False
        if len(self.args.gpus) >= 1:
            self.enable_cuda = True
            torch.cuda.set_device('cuda:{}'.format(self.args.gpus[0]))
            self.criterion = self.criterion.cuda()

        self.model = select_model(args)
        if self.args.load_model_path != '':
            print("=> using pre-trained weights")
            if args.load_not_strict:
                torch_utils.load_match_dict(self.model, self.args.load_model_path)
            else:
                self.model.load_state_dict(torch.load(args.load_model_path))
        print(args.model_type)
        self.model = torch_utils.allocate_devices(self.model, self.args.gpus)
        #self.model=self.model.cuda()
        if len(self.args.freeze_layers) > 0:
            self.model = torch_utils.freeze_layers(self.model, self.args.freeze_layers)
        #if self.args.wandb_sweep_path != '':
        #    self.args.lr = wandb.config.lr
        #    self.args.momentum = wandb.config.momentum
        #    self.args.beta = wandb.config.beta
        #    self.args.weight_decay = wandb.config.weight_decay
        #    self.args.batch_size = wandb.config.batch_size
        #    self.args.epochs = wandb.config.epochs

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.args.lr,
                                          betas=(self.args.momentum, self.args.beta),
                                          weight_decay=self.args.weight_decay)
        self.file_path=os.path.join(self.args.model_dir,"train.txt")
        self.file_path_test=os.path.join(self.args.model_dir,"test.txt")
        self.file_name=None
        self.file_name_test=None
        self.max_acc=0.0
        self.max_pre=0.0
        self.max_re=0.0
        self.temp_acc=0.0
        self.temp_pre=0.0
        self.temp_re=0.0
        
        
    def train(self):
        self.file_name=open(self.file_path,'w')
        self.file_name_test=open(self.file_path_test,'w')
        for epoch in range(self.args.epochs):
            torch_utils.adjust_learning_rate(self.args, self.optimizer, epoch)
            # train for one epoch
            self.train_per_epoch(epoch)
            self.val_per_epoch(epoch)
            save_flag=0
            if self.max_acc<self.temp_acc:
                save_flag=1
            elif (self.max_pre<self.temp_pre and self.max_acc==self.temp_acc) or (self.max_re<self.temp_re and self.max_acc==self.temp_acc):
                save_flag=1
            else:
                save_flag=0
            state_dict=self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
            if save_flag==1:
                model_name='optimal.pth'
                model_path=os.path.join(self.args.model_dir,model_name)
                #state_dict = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
                torch.save(state_dict,model_path)
                self.max_acc=self.temp_acc
                self.max_re=self.temp_re
                self.max_pre=self.temp_pre
            model_name='latest.pth'
            model_path=os.path.join(self.args.model_dir,model_name)
            torch.save(state_dict,model_path)
            
            
            #self.logger.save_check_point(self.model, epoch)
        self.file_name.close()
        self.file_name_test.close()
        
    def train_per_epoch(self, epoch):# modify train and val epoch
        self.model.train()
        losses = AverageMeter()
        total_correct = 0
        total_samples = 0
        TP = 0
        FP = 0
        FN = 0
        for i, (x, label, filename, info, human) in enumerate(self.train_loader):
    
            if self.enable_cuda:
                x = x.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)
    
            self.optimizer.zero_grad()
    
            pred = self.model(x)
            loss = self.criterion(pred, label)
    
            loss.backward()
            self.optimizer.step()
    
            bsz = label.size(0)
            losses.update(loss.item(), bsz)
    
            pred_label = pred.argmax(dim=1)

            total_correct += (pred_label == label).sum().item()
            total_samples += bsz

            TP += ((pred_label == 1) & (label == 1)).sum().item()
            FP += ((pred_label == 1) & (label == 0)).sum().item()
            FN += ((pred_label == 0) & (label == 1)).sum().item()
    
            if i % self.args.print_freq == 0:
                print(f"Train: [{epoch}][{i+1}/{len(self.train_loader)}]\t"
                      f"loss {losses.val:.3f} ({losses.avg:.3f})")

        acc = total_correct / total_samples
        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
    
        record = f"{epoch} acc:{acc} recall:{recall} precision:{precision}\n"
        self.file_name.writelines(record)
            
        
    def val_per_epoch(self, epoch):
        self.model.eval()
        losses = AverageMeter()
        total_correct = 0
        total_samples = 0
        TP = 0
        FP = 0
        FN = 0
        with torch.no_grad():
            for i, (x, label, filename, info, human) in enumerate(self.val_loader):
    
                if self.enable_cuda:
                    x = x.cuda(non_blocking=True)
                    label = label.cuda(non_blocking=True)
    
                pred = self.model(x)
                loss = self.criterion(pred, label)
    
                bsz = label.size(0)
                losses.update(loss.item(), bsz)
    
                pred_label = pred.argmax(dim=1)
    
                total_correct += (pred_label == label).sum().item()
                total_samples += bsz
    
                TP += ((pred_label == 1) & (label == 1)).sum().item()
                FP += ((pred_label == 1) & (label == 0)).sum().item()
                FN += ((pred_label == 0) & (label == 1)).sum().item()
        acc = total_correct / total_samples
        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
    
        # 更新最优判断用变量
        self.temp_acc = acc
        self.temp_pre = precision
        self.temp_re = recall
    
        record = f"{epoch} acc:{acc} recall:{recall} precision:{precision}\n"
        self.file_name_test.writelines(record)
        
    def compute_metrics(self, pred, gt):
        # you can call functions in metrics.py
        # l1 = (pred - gt).abs().mean()
        pred = pred.argmax(dim=1)
        acc = metric.accuracy_score(gt.cpu().numpy(), pred.cpu().numpy())
        recall = metric.recall_score(gt.cpu().numpy(), pred.cpu().numpy(), average='macro')
        precision = metric.precision_score(gt.cpu().numpy(), pred.cpu().numpy(), average='macro')
        metrics = {
            'acc': acc,
            'recall': recall,
            'precision': precision
        }
        return metrics
    
    def get_criterion(self):
        if self.args.loss == 'l1':
            criterion = torch.nn.L1Loss()
        elif self.args.loss == 'ce':
            criterion = torch.nn.CrossEntropyLoss()
        else:
            criterion = torch.nn.MSELoss()
        return criterion


def main():
    trainer = Trainer()
    #if trainer.args.wandb_sweep_path != '':
    #    wandb.agent(trainer.logger.sweep_id, function=trainer.train, count=10)
    #else:
    trainer.train()


if __name__ == '__main__':
    main()
