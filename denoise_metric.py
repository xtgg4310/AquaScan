import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
    
def dir_create(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def read_rescale_results(file_path):
    human_ids = []
    states = []
    objs = []
    file_split=file_path.split("/")
    s=file_split[2]
    sonar=file_split[3]
    with open(file_path, 'r') as f:
        lines = f.readlines()
        if '\n' in lines:
            lines.remove('\n')
        for line in lines:
            arr = line.strip().split()
            xmin = int(float(arr[4])) #4
            ymin = int(float(arr[2])) #2
            xmax = int(float(arr[5])) #5
            ymax = int(float(arr[3])) #3
            
            obj = [ymin, ymax, xmin, xmax] #label stored before is ymin, ymax, xmin, xmax
            objs.append(obj)
    return human_ids,states,objs
    
def read_result(file_path):
    objs=[]
    with open(file_path,'r') as f:
        lines=f.readlines()
        lines=list(set(lines))
        for line in lines:
            data=line.split(',')
            x_min=min(np.float32(data[0]),np.float32(data[2]))
            x_max=max(np.float32(data[0]),np.float32(data[2]))
            y_min=min(np.float32(data[1]),np.float32(data[3]))
            y_max=max(np.float32(data[1]),np.float32(data[3]))
            obj=[y_min,y_max,x_min,x_max] #obj stored before is xmin, ymin, xmax, ymax
            objs.append(obj)
    #objs=list(set(objs))
    return objs

def cal_IoU(obj1, obj2, only_label=False):
    cross_area=[max(obj1[0],obj2[0]),min(obj1[1],obj2[1]),max(obj1[2],obj2[2]),min(obj1[3],obj2[3])]
    cross=0
    if cross_area[0]>=cross_area[1] or cross_area[2]>=cross_area[3]:
        cross=0
    else:
        cross=(cross_area[1]-cross_area[0])*(cross_area[3]-cross_area[2])
    obj1_area=(obj1[1]-obj1[0])*(obj1[3]-obj1[2])
    obj2_area=(obj2[1]-obj2[0])*(obj2[3]-obj2[2])
    sum_area=obj1_area+obj2_area
    IoU=(cross)/(sum_area-cross) #sum areas - cross = Union， cross=intersection
    return IoU
    
def read_default_label(file_path):
    '''
    Args:
        file_path: path of label file
    Returns:
        human_ids: list of human id
        states: list of state
        objs: list of object, each object is a list of [ymin, ymax, xmin, xmax]
    '''
    human_ids = []
    states = []
    objs = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            arr = line.strip().split()
            # human object
            if len(arr) == 6:
                human_id = arr[0]
                state = arr[1]
                xmin = int(float(arr[2]))
                ymin = int(float(arr[3]))
                xmax = int(float(arr[4]))
                ymax = int(float(arr[5]))
            
            # noise object
            elif len(arr) == 5:
                human_id = -2
                state = "noise"
                xmin = int(float(arr[1]))
                ymin = int(float(arr[2]))
                xmax = int(float(arr[3]))
                ymax = int(float(arr[4]))

            else:
                raise ValueError('label file format error: {}'.format(file_path))

            obj = [ymin, ymax, xmin, xmax]
            human_ids.append(human_id)
            states.append(state)
            objs.append(obj)
    return human_ids, states, objs

def read_default_label_raw(file_path):
    human_ids = []
    states = []
    objs = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        if '\n' in lines:
            lines.remove('\n')
        for line in lines:
            arr = line.strip().split()
            xmin = int(float(arr[2])) #4
            ymin = int(float(arr[0])) #2
            xmax = int(float(arr[3])) #5
            ymax = int(float(arr[1])) #3
            
            obj = [ymin, ymax, xmin, xmax]
            objs.append(obj)
    return objs

def read_yolo_label(file_path, W=500, H=400):
    humans=[]
    obj_list = []
    states = []
    file_split=file_path.split("/")
    s=file_split[2]
    sonar=file_split[3]
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            arr = line.strip().split()
            human_id = arr[0]
            state = arr[1]
            x_center = float(arr[2])
            y_center = float(arr[3])
            width = float(arr[4])
            height = float(arr[5])

            xmin = int((x_center - width / 2) * W)
            xmax = int((x_center + width / 2) * W)
            ymin = int((y_center - height / 2) * H)
            ymax = int((y_center + height / 2) * H)

            obj = [ymin, ymax, xmin, xmax]
            obj_list.append(obj)
            states.append(state)
            humans.append(human_id)
    return humans, states, obj_list

def generate_miss_iou_map(detected_obj,label_obj):
    miss_re=np.ones(len(label_obj))
    correct_re=np.zeros(len(label_obj))
    record_id=[]
    record_label=[]
    correct_re_label=np.zeros(len(label_obj))
    miss_iou_map=np.zeros((len(label_obj),len(detected_obj)))
    for i in range(len(label_obj)):
        for j in range(len(detected_obj)):
            miss_iou_map[i][j]=cal_IoU(detected_obj[j],label_obj[i],False)

    while (miss_iou_map.any()):
        max_index=np.unravel_index(np.argmax(miss_iou_map, axis=None), miss_iou_map.shape)
        max_iou=miss_iou_map[max_index[0]][max_index[1]]
        print(max_index,max_iou)
        print(miss_iou_map)
        if max_iou!=0:
            miss_re[max_index[0]]=0
            correct_re[max_index[0]]=max_iou
            correct_re_label[max_index[0]]=cal_IoU(detected_obj[max_index[1]],label_obj[max_index[0]],True)
            record_label.append(max_index[0])
            record_id.append(max_index[1])
            miss_iou_map[max_index[0],:]=0
            miss_iou_map[:,max_index[1]]=0
        else:
            pass
    return miss_re,correct_re,correct_re_label

def calculate_statistic(detected_obj, label_obj,detect_path):
    #print(detected_obj)
    '''
    miss_re=np.zeros(len(label_obj))
    correct_re=np.zeros(len(label_obj))
    wrong_re=len(detected_obj)-len(label_obj)
    if wrong_re<0:
        wrong_re=0
    max_id=-1
    max_iou=0
    max_iou_label=0
    record_id=[]
    record_label=[]
    correct_re_label=np.zeros(len(label_obj))
    
    for i in range(len(label_obj)):
        max_id=-1
        max_iou=0.0
        for j in range(len(detected_obj)):
            iou_temp=cal_IoU(detected_obj[j],label_obj[i],False)
            iou_label=cal_IoU(detected_obj[j],label_obj[i],True)
            print(label_obj[i],iou_temp)
            if iou_temp>=max_iou and j not in record_id: #and i not in record_label:
                max_id=j
                max_iou=iou_temp
                max_iou_label=iou_label
        if max_iou==0.0:
            
            miss_re[i]=1
        else:
            correct_re[i]=max_iou
            correct_re_label[i]=max_iou_label
            record_label.append(i)
            record_id.append(max_id)
            #print(max_iou)
    '''
    miss_re,correct_re,correct_re_label=generate_miss_iou_map(detected_obj,label_obj) 
    #'''
    detect_num=0
    for k in range(len(miss_re)):
        if miss_re[k]==0:
            detect_num+=1
    wrong_re=len(detected_obj)-detect_num
    print(miss_re,correct_re,wrong_re,len(detected_obj),len(label_obj))
    return miss_re,correct_re,wrong_re,correct_re_label

def inter_section(sec1,sec2):
    if sec1[0]>sec2[1] or sec1[1]<sec2[0]:
        return False,0.0
    else:
        section=(min(sec1[1],sec2[1])-max(sec1[0],sec2[0]))/(max(sec1[1],sec2[1])-min(sec1[0],sec2[0]))
        return True, section
                 
def compare_results(detect_path,label_path,yolo=False,eval_format=False):
    obj_num=0
    miss_num=0
    IoU=[]
    IoU_label=[]
    wrong_num=0
    frame_num=0
    for i in range(len(detect_path)):
        if not eval_format:
            detect_obj=read_result(detect_path[i])
        else:
            detect_obj=read_default_label_raw(detect_path[i])
        if yolo:
            _,_,label_obj=read_yolo_label(label_path[i])
        else:
            _,_,label_obj=read_rescale_results(label_path[i])
        miss_re,correct_re,wrong_re,corr_label=calculate_statistic(detect_obj,label_obj,detect_path)
        wrong_num+=wrong_re
        frame_num+=1
        obj_num+=len(label_obj)
        for j in range(len(miss_re)): # here we recognize all the subject has only one box. But in state recognition, the ones split from the same subject will share the same GT label.
            if miss_re[j]==1:
                miss_num+=1
            else:
                IoU.append(correct_re[j])
                IoU_label.append(corr_label[j])
    if obj_num!=0:
        miss_rate=miss_num/obj_num*1.0
        wrong_aver=wrong_num/frame_num*1.0
    else:
        miss_rate=0
        wrong_aver=wrong_num/frame_num*1.0
    detected_obj=obj_num-miss_num
    print("detect")
    print(detected_obj)
    print("wrong")
    print(wrong_num)
    print("miss_num")
    print(miss_num)
    precision=detected_obj/(detected_obj+wrong_num)
    recall=detected_obj/(detected_obj+miss_num)
    F_score=2*precision*recall/(precision+recall)
    IoU_aver=np.mean(np.array(IoU))
    IoU_var=np.var(np.array(IoU))
    print("number total:")
    print(obj_num,frame_num)
    print(" ")
    return wrong_aver,miss_rate,IoU_aver,IoU_var,IoU,IoU_label,F_score,wrong_num,miss_num,obj_num,frame_num
            
def data_label_tune(obj_dir,label_dir,scenario_list=[]):
    detect_data_our=[]
    label_list=[]
    scenarios=os.listdir(obj_dir)
    for scenario in scenarios:
        if scenario[0]==".":
            continue
        if scenario not in scenario_list:
            continue
        obj_dir_s=obj_dir+"/"+scenario
        label_dir_s=label_dir+"/"+scenario
        sonars=os.listdir(label_dir_s)
        for sonar in sonars:
            if sonar[0]==".":
                continue
            obj_dir_sonar=obj_dir_s+"/"+sonar
            label_dir_sonar=label_dir_s+"/"+sonar
            files=os.listdir(label_dir_sonar)
            if ".DS_Store" in files: 
                files.remove(".DS_Store")
            for file in files:
                time=file.split('_')[1][:-4]
                obj_file=obj_dir_sonar+"/txt/"+file
                label_file=label_dir_sonar+"/"+file
                detect_data_our.append(obj_file)
                label_list.append(label_file)
    return detect_data_our,label_list
            
def eval_single(data_path,label_path,scenario=["111"],name=-1):
    label_dir=label_path
    data_dir=data_path
    wrong_list=[]
    miss_rate_list=[]
    IoU_list=[]
    IoU_label_list=[]
    path,label=data_label_tune(data_dir,label_dir,scenario)
    wrong_aver,miss_rate,IoU_aver,IoU_var,IoU,IoU_label,F_score,wrong_num,miss_num,obj_num,frame_num=compare_results(path,label,yolo=False,eval_format=False)
    
    wrong_list.append(wrong_aver)
    miss_rate_list.append(miss_rate)
    IoU_list.append(IoU)
    IoU_label_list.append(IoU_label)
    array_static=np.array([wrong_num,miss_num,obj_num,frame_num])
    IoU_aver=np.mean(IoU_list)
    if name!=-1:
        dir_create("./object_detection_results")
        np.save("./object_detection_results/"+name+"_detect.npy",array_static)
        np.save("./object_detection_results/"+name+"_IoU.npy",IoU)
    print(wrong_aver,miss_rate,IoU_aver,F_score)
    return wrong_aver,miss_rate,IoU_aver

    
if __name__=="__main__":
    pass
