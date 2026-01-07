from torch.utils.data import Dataset
import numpy as np
import data_process.augment as augment
import cv2
import os

class ClassifyDataset(Dataset):
    def __init__(self, args, is_train,infe=True):
        super(ClassifyDataset, self).__init__()
        if is_train:
            data_list = args.train_list
        else:
            data_list = args.val_list
        #print(args.val_list)
        infos = [line.split() for line in open(data_list).readlines()]
        #print(infos)
        data_paths = [info[0] for info in infos]
        label_paths = [info[1] for info in infos]
        self.augment_type = args.augment_type
        self.data = []
        self.labels = []
        self.humans_id=[]
        self.file_name = []
        self.time_collect=[]
        self.scenarios = []
        self.is_train = is_train
        label2id = args.label2id
        self.sonar=[]
        if infe:
            label_paths.sort(key=lambda x:(x.split('/')[-1]).split('_')[1])
            data_paths.sort(key=lambda x:(x.split('/')[-1]).split('_')[1])
            pass
        else:
            label_paths.sort(key=lambda x:x.split('.')[0])
            data_paths.sort(key=lambda x:x.split('.')[0])
        for data_path, label_path in zip(data_paths, label_paths):
            label_data_single=open(label_path).readlines()
            _label = str(label_data_single[1].strip())
            human_id= str(label_data_single[0].strip())
            #label=_label
            label = label2id.get(_label, -1)

            if args.channel_num == 1:
                sonar_data = cv2.imread(data_path, cv2.IMREAD_GRAYSCALE)
            elif args.channel_num == 3:
                # cv2 stores image in BGR format
                sonar_data = cv2.imread(data_path, cv2.IMREAD_COLOR)
            else:
                sonar_data=np.load(data_path)
                sonar_data=sonar_data.astype(np.float32)
            if label == -1:
                continue
            self.data.append(sonar_data)
            self.labels.append(label)
            self.file_name.append(os.path.basename(data_path))
            self.humans_id.append(human_id)
            #self.sonar=[]

            scenario = str(open(label_path).readlines()[4].strip())#str(open(label_path).readlines()[4].strip().split(',')[-1].split('/')[-3])
            self.time_collect.append(str(open(label_path).readlines()[5])) #str(open(label_path).readlines()[5])
            self.sonar.append(str(open(label_path).readlines()[3][:-1]))#(str(open(label_path).readlines()[4].strip()).split(' ')[2]).split('/')[-2])
            self.scenarios.append(scenario)
        self.infe=infe
    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):
        if self.is_train:
            sonar_data = augment.get_transforms(self.augment_type)(self.data[idx])
        else:
            sonar_data = augment.get_transforms('none')(self.data[idx])
        label = self.labels[idx]
        file_name = self.file_name[idx]
        scenario = self.scenarios[idx]
        human=self.humans_id[idx]
        if self.infe:
            return sonar_data, label,file_name,scenario,self.time_collect[idx],human,self.sonar[idx]
        else:
            return sonar_data, label, file_name, scenario,human #,"sonarx"
