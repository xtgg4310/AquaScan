import numpy as np
import torch
from data_process import select_eval_loader
from model import select_model
from options import get_options
from utils.logger import Logger, AverageMeter
import os
import time
import sklearn.metrics as metric
import utils.torch_utils as torch_utils

import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

class Evaluator:
    def __init__(self):
        args = get_options()
        self.args = args
        self.model = select_model(args)
        #self.model=self.model.double()
        if self.args.load_not_strict:
            torch_utils.load_match_dict(self.model, self.args.load_model_path)
        else:
            print("load model from {}".format(args.load_model_path))
            self.model.load_state_dict(torch.load(args.load_model_path))
        
        self.enable_cuda = False
        if len(self.args.gpus) >= 1:
            self.enable_cuda = True
            torch.cuda.set_device('cuda:{}'.format(self.args.gpus[0]))
            self.model = self.model.cuda()

        self.model.eval()
        self.val_loader = select_eval_loader(self.args)
        self.logger = Logger(self.args)

    def eval(self):
        acc = AverageMeter()
        recall = AverageMeter()
        precision = AverageMeter()
        scenario_acc_dict = {}
        scenario_frame={}
        scenario_re={}
        CM = None
        time_count=0.0
        gt_list=None
        pred_list=None
        self.model.eval()
        with torch.no_grad():
            for idx, (x, label, filename, scenario,file,human,sonar) in enumerate(self.val_loader):
                #torch.no_grad()
                #print(x.shape)
                if self.enable_cuda:
                    x = x.cuda(non_blocking=True)
                    label = label.cuda(non_blocking=True)
                time_temp=time.time()
                bsz = label.size(0)
                pred = self.model(x)
                bsz = label.size(0)
                #print(bsz)
                if pred_list==None:
                    pred_list=pred
                else:
                    pred_list=torch.concat([pred_list,pred])
                if gt_list==None:
                    gt_list=label
                else:
                    gt_list=torch.concat([gt_list,label])
                time_count+=time.time()-time_temp
                metrics = self.compute_metrics(pred, label)
                acc.update(metrics['acc'], bsz)
                recall.update(metrics['recall'], bsz)
                precision.update(metrics['precision'], bsz)
                for s, p,p_arg, l,f,h,s_n in zip(scenario, pred, pred.argmax(dim=1),label,file,human,sonar):
                    temp_p=p.cpu()
                    if s not in scenario_re:
                        scenario_re[s]=[[temp_p.detach().numpy(),p_arg.cpu().detach().numpy(),l.cpu().detach().numpy(),f,h,s_n]]
                    else:
                        scenario_re[s].append([temp_p.detach().numpy(),p_arg.cpu().detach().numpy(),l.cpu().detach().numpy(),f,h,s_n])
                    if s not in scenario_acc_dict:
                        scenario_acc_dict[s] = AverageMeter()
                        scenario_frame[s]=1
                    if p_arg == l:
                        scenario_acc_dict[s].update(1, 1)
                        scenario_frame[s]+=1
                    else:
                        scenario_acc_dict[s].update(0, 1)
                        scenario_frame[s]+=1

                if CM is None:
                    CM = self.compute_confusion_matrix(pred, label)
                else:
                    CM += self.compute_confusion_matrix(pred, label)

                #wandb.log({'val_acc':acc.avg, 'val_recall':recall.avg, 'val_precision':precision.avg})
                # if i % self.args.vis_freq == 0:
                #     self.logger.save_imgs(x, pred.argmax(dim=1), label, filename, False)
            metrics_all = self.compute_metrics(pred_list, gt_list)
            #print(metrics['acc'],metrics['recall'],metrics['precision'],bsz)
            acc_all=AverageMeter()
            recall_all=AverageMeter()
            precision_all=AverageMeter()
            bsz_all=gt_list.size(0)
            acc_all.update(metrics_all['acc'], bsz_all)
            recall_all.update(metrics_all['recall'], bsz_all)
            precision_all.update(metrics_all['precision'], bsz_all)
            print(time_count)
            result_txt_path = os.path.join(self.args.result_dir, 'result.txt')
            
            # write metrics to result dir,
            # you can also use pandas or other methods for better stats
            with open(result_txt_path, 'w') as fd:
                fd.write("=== metrics ===\n")
                fd.write(str(metrics_all))
                #fd.write("acc:"+str(acc_all.avg)+" recall:"+str(recall_all.avg)+" precision:"+str(precision_all.avg)+"\n")
                fd.write('\n')
                fd.write("=== scenario acc ===\n")
                for s in scenario_acc_dict:
                    fd.write("{}: {}, {} \n".format(s, scenario_acc_dict[s].avg,scenario_frame[s]))
                    fd.write('\n')
                fd.write("=== confusion matrix ===\n")
                fd.write(str(CM))
                fd.write('\n')

            # draw confusion matrix
            CM = CM / CM.sum(axis=1)[:, None]
            ticklabels = ['swimming','standing','patting', 'struggling', 'drowning']
            #ticklabels = ['still', 'struggle']
            sns.heatmap(CM, annot=True, fmt='.2f', cmap='Blues', xticklabels=ticklabels, 
                                                            yticklabels=ticklabels)
            plt.xlabel('Predicted label')
            plt.ylabel('True label')
            plt.title(self.args.save_prefix)
            plt.savefig(os.path.join(self.args.result_dir, 'confusion_matrix.png'))
            plt.close()
            #print("save")
            #for s in scenario_acc_dict:
            #    print("{}: {}, {}".format(s, scenario_acc_dict[s].avg, scenario_frame[s]))
            
            new_dir=self.args.result_dir+"/record"
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
            for key in scenario_re:
                s_name=str(key)#.split('/')[-3]
                #print(s_name)
                s_path=os.path.join(new_dir, s_name+'_result.txt')
                with open(s_path,'w') as fd:
                    content=scenario_re[key]
                    for i in range(len(content)):
                        fd.writelines(str(content[i][0][0])+","+str(content[i][0][1])+","+str(content[i][1])+","+str(content[i][2])+","+str(content[i][3])+","+str(content[i][4])+","+str(content[i][5])+"\n")
                #        

    def compute_metrics(self, pred, gt):
        # you can call functions in metrics.py
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

    def compute_confusion_matrix(self, pred, gt):
        pred = pred.argmax(dim=1)
        cm = metric.confusion_matrix(gt.cpu().numpy(), pred.cpu().numpy(), labels=range(0, self.args.num_classes, 1))
        return cm
    
def eval_main():
    torch.cuda.empty_cache()
    print("relasing")
    evaluator = Evaluator()
    evaluator.eval()


if __name__ == '__main__':
    eval_main()
