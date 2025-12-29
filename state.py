import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import time
import argparse

##check fre remove -1 in time_start
#matplotlib.rc('text.latex', preamble=[r'\usepackage{libertine}',r'\usepackage[libertine]{newtxmath}',r'\usepackage{sfmath}',r'\usepackage[T1]{fontenc}'])
#matplotlib.rc('text', usetex=True)
matplotlib.rc('pdf', fonttype=42)
plt.rc('font', family='Times New Roman', size=14)

color = ['#E38D8C','#F2B74D','#67B1D7', '#84C2AE', '#999999','#dbe339','#4dcaf0', '#E99C93','#9FACD3','#7fc97f','#beaed4','#fdc086', '#8EBAD6']

def matrix2ratio(matrix):
    matrix_new=np.zeros((5,5))
    for i in range(len(matrix)):
        count=np.sum(matrix[i,:])
        for j in range(len(matrix[i])):
            matrix_new[i][j]=matrix[i][j]/count
    return matrix_new

def dir_create(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

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
    if not only_label:
        IoU=(cross)/(sum_area-cross)
    else:
        if obj2_area!=0:
            IoU=cross/obj2_area
        else:
            IoU=0
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
        if '\n' in lines:
            lines.remove('\n')
        #print(lines)
        for line in lines:
            arr = line.strip().split()
            # human object
            if len(arr) == 6:
                human_id = arr[0]
                state = arr[1]
                xmin = int(float(arr[2])/2.0)
                ymin = int(float(arr[3])/2.0)
                xmax = int(float(arr[4])/2.0)
                ymax = int(float(arr[5])/2.0)
            
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

def read_yolo_label(file_path, W=500, H=400):
    ''' Read one txt file in yolo format.
        human_id | state_str | x_center | y_center | width | height
    Args:
        file_path: path of label file
        W: width of sonar data, default 500 unit (20 m)
        H: height of sonar data, default 400 gradian
    Returns:
        obj_list: list of object, each object is a list of [ymin, ymax, xmin, xmax]
        states: list of state, each state is a string
    '''
    humans=[]
    obj_list = []
    states = []
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

def generate_gt_label(dir_gt,time_single,target_sce=[],label_type=0,gt_config=[30,50,60]):
    scenairos=os.listdir(dir_gt)
    #gt_record={}
    gt_generate=[]
    gt_dict={}
    if label_type==0:
        patting_time=gt_config[0]
        struggle_time=gt_config[1]
    else:
        patting_time=gt_config[0]
        struggle_time=gt_config[1]
    if ".DS_Store" in scenairos:
        scenairos.remove(".DS_Store")
    for scenario in scenairos:
        scenario_path=dir_gt+"/"+scenario
        sonars=os.listdir(scenario_path)
        if ".DS_Store" in sonars:
            sonars.remove(".DS_Store")
        for sonar in sonars:
            sonar_path=scenario_path+"/"+sonar
            files=os.listdir(sonar_path)
            #key=scenario+"_"+sonar
            if ".DS_Store" in files:
                files.remove(".DS_Store")
            
            files.sort(key=lambda x:int(x.split('_')[1][:-4]))
            count=0
            time_start=(files[0].split('_'))[0].split('-')
            key=scenario+"_"+sonar
            time_start=np.float128(time_start[-2])*60+np.float128(time_start[-1])
            for file in files:
                #timestamp=file
                time_file=(file.split('_'))[0].split('-')
                time_stamp_single=np.float128(time_file[-2])*60+np.float128(time_file[-1])-time_start
                key_single=key+"_"+str(time_stamp_single)
                #print(key_single)
                file_single=sonar_path+"/"+file
                if label_type==0:
                    h,s,o=read_yolo_label(file_single)
                else:
                    h,s,o=read_default_label(file_single)
                #new_state=[]
                gt_dict.update({key_single:[]})
                for i in range(len(h)):
                    if s[i]=="struggle":
                        if time_stamp_single<patting_time:
                            new_state='patting'
                        elif time_stamp_single<struggle_time: 
                            new_state='struggling'
                        else:
                            new_state='drowning'
                    elif s[i] == "moving" or s[i]=="swim" or s[i]=="swimming":
                        new_state="moving"
                    else:
                        if time_stamp_single<gt_config[2]: 
                            new_state="motionless"
                        else:
                            new_state="drowning"
                    gt_generate.append([h[i],new_state,o[i],file])
                    gt_dict[key_single].append([h[i],new_state,o[i]])
                count+=1
    return gt_generate,gt_dict               
    
#def read_GT_label(path):
def read_detection_file(path,time_single,time_thre):
    f=open(path,'r')
    lines=f.readlines()
    print(path)
    detect_motion_list=[]
    GT_motion_list=[]
    timestamp=[]
    detect_motion_list_old=[]
    time_start_one=time_single
    start=0
    count=0
    for line in lines:
        data_sample=line.split(',')
        count+=1
        detect_motion=int(data_sample[0])
        GT_motion=int(data_sample[1])
        time_stamp_one=data_sample[2].split("_")[0]
        #print(time_stamp_one,time_start_one)
        time_stamp=np.int32(time_stamp_one.split('-')[-2])*60+np.int32(time_stamp_one.split('-')[-1])-time_start_one
        detect_motion_list.append(detect_motion)
        detect_motion_list_old.append(detect_motion)
        GT_motion_list.append(GT_motion)
        timestamp.append(time_stamp)
    detect_motion_list_new=motion_state_smooth(detect_motion_list,timestamp,time_thre)
    return detect_motion_list_new,GT_motion_list,timestamp,detect_motion_list_old
    
def read_moving_file(path):
    f=open(path,'r')
    lines=f.readlines()
    moving=[]
    timestamp=[]
    GT=[]
    seg_list=[]
    start=0.0
    count=0
    for line in lines:
        data_sample=line.split()
        if count==0:
            start=float(data_sample[1])
        count+=1
        moving_detect=data_sample[0]
        time=float(data_sample[1])
        GT_moving=data_sample[2][:]
        seg=[float(data_sample[4]),float(data_sample[5]),float(data_sample[6]),float(data_sample[7])]
        moving.append(moving_detect)
        timestamp.append(time)
        GT.append(GT_moving)
        seg_list.append(seg)
    return moving,timestamp,GT,seg_list

def moving_generate_mark(moving_state):
    if moving_state=="moving":
        return "C"
    else:
        return "U"
    
def motion_mark_generate(motion):
    if motion==0 or motion=='0':
        return 'S'
    else:
        return 'M'

def re_smooth_start(motion_flag_list,motion_flag_list_old,moving_flag_list,timestamp,time_thre,path): # resmooth the first 2 states when their following state changes. This function works after smoothing motion and movement. This design is proposed due to lack of past information. To reduce the overhead, we have a strict re-smooth conditions to not easily trigger it. In most of cases, this will not be triggered so the latency of first 2 states will not increase.
    if len(motion_flag_list)<3:
        return motion_flag_list
    if (motion_flag_list[1]!=motion_flag_list_old[1] or motion_flag_list[2]!=motion_flag_list_old[2]) and timestamp[2]-timestamp[0]<=time_thre and motion_flag_list[0]!=motion_flag_list[1]: #and moving_flag_list[2]!="C" and moving_flag_list[1]!="C":
        #print(moving_flag_list[1],moving_flag_list[2])
        #return motion_flag_list
        change_first_flag=False
        if motion_flag_list[2]!=motion_flag_list[0]:
            if motion_flag_list[1]==motion_flag_list[2]:
                motion_flag_list[0]=motion_flag_list[2]
                change_first_flag=True
            else:
                pass
        if change_first_flag:
            print("changed_1",path)
        if change_first_flag: #and moving_flag_list[1]!="C":    
            if len(motion_flag_list)>3 and motion_flag_list[1]!=motion_flag_list[2] and motion_flag_list[1]!=motion_flag_list[3] :
                stat_diff=0
                diff_count=0
                same_count=0
                for i in range(0,4):
                    #print("re",i)
                    if motion_flag_list[i]==motion_flag_list[1]:
                        same_count+=1
                    else:
                        stat_diff=motion_flag_list[i]
                        diff_count+=1
                if diff_count>same_count:
                    motion_flag_list[1]=stat_diff
        return motion_flag_list
    else:
        #print("not re")
        return motion_flag_list
            

def read_har_file(path,time_thre):
    f=open(path,'r')
    lines=f.readlines()
    moving_flag_list=[]
    moving_GT_list=[]
    timestamp=[]
    motion_flag_list=[]
    motion_flag_list_old=[]
    motion_fea_list=[]
    motion_GT_list=[]
    seg_list=[]
    for line in lines:
        data=line.split(',')
        time_single=float(data[0])
        moving_single=moving_generate_mark(data[1])
        moving_single_GT=moving_generate_mark(data[2])
        #motion_single=motion_mark_generate(int(data[3]))
        motion_single=int(data[3])
        motion_single_GT=motion_mark_generate(int(data[4]))
        timestamp.append(time_single)
        moving_flag_list.append(moving_single)
        moving_GT_list.append(moving_single_GT)
        motion_flag_list.append(motion_single)
        motion_GT_list.append(motion_single_GT)
        motion_single_old=float(data[9])
        motion_flag_list_old.append(motion_single_old)
        seg_single=[float(data[5]),float(data[6]),float(data[7]),float(data[8])]
        seg_list.append(seg_single)
    motion_flag_list=re_smooth_start(motion_flag_list,motion_flag_list_old,moving_flag_list,timestamp,time_thre,path) # resmooth the first 2 states when their following state changes. This function works after smoothing motion and movement. This design is proposed due to lack of past information. To reduce the overhead, we have a strict re-smooth conditions to not easily trigger it. In most of cases, this will not be triggered so the latency of first 2 states will not increase.
    for i in range(len(motion_flag_list)):
        motion_feature_single=motion_mark_generate(motion_flag_list[i])
        motion_fea_list.append(motion_feature_single)
    return timestamp,moving_flag_list,moving_GT_list,motion_fea_list,motion_GT_list,seg_list

def read_detect_result(path):
    f=open(path,'r')
    lines=f.readlines()
    timestamp=[]
    detect_re=[]
    seg=[]
    for line in lines:
        data=line.split(' ')
        time_single=data[0]
        detect_single=data[1]
        seg_single=[float(data[3]),float(data[4]),float(data[5]),float(data[6])]
        timestamp.append(time_single)
        detect_re.append(detect_single)
        seg.append(seg_single)
        
    return timestamp,detect_re,seg
        
    
def generate_detect_label(detect_dir,target_sce,target_file=[]):
    files = os.listdir(detect_dir)
    if ".DS_Store" in files:
        files.remove(".DS_Store")
    detect_re=[]
    #print(target_file)
    for file in files:
        scenario=file.split('_')[0]
        if target_sce!=[]:
            if scenario not in target_sce:
                continue
        if target_file!=[]:
            if file not in target_file:
                continue
        detect_path=detect_dir+"/"+file
        time_single,detect_single,seg_single=read_detect_result(detect_path)
        for i in range(len(time_single)):
            re_single=[time_single[i],detect_single[i],seg_single[i],file]
            detect_re.append(re_single)
    return detect_re

def state_merge_motion_temporal(motion_dir, moving_dir,save_dir,start_cfg,smooth_cfg):
    files=os.listdir(motion_dir)
    if ".DS_Store" in files:
        files.remove(".DS_Store")
    dir_create(save_dir)
    for file in files:
        #print(file)
        scenario=file.split('_')[0]
        sonar=file.split('_')[4][:-4]
        if start_cfg==['0.0']:
            time_single=1000 #config, start time, min*60+second if in one hour.
        else:
            time_single=np.float32(start_cfg[0])            
        if smooth_cfg==['0.0']:
            time_single=3.0 #config smooth time figuration
            time_thre=time_single*2+np.int32(time_single)-1
            if time_thre==-1:
                print("error")
                
        else:
            time_thre=np.float32(smooth_cfg[0])
        
        motion_file_dir=motion_dir+"/"+file
        id_first=file.split('_')[2]
        id_second=file.split('_')[3]
        file_moving=scenario+"_"+sonar+"_"+id_first+"_"+id_second+".txt"
        moving_file_dir=moving_dir+"/"+file_moving
        save_dir_file=save_dir+"/"+file_moving
        moving_list,timestamp_moving,GT_moving,seg_list=read_moving_file(moving_file_dir)
        detect_motion_list,GT_motion_list,timestamp_motion,detect_motion_list_old=read_detection_file(motion_file_dir,time_single,time_thre)
        f=open(save_dir_file,'w')
        for i in range(len(timestamp_motion)):
            index_moving=-1
            for k in range(len(timestamp_moving)):
                if timestamp_motion[i]==timestamp_moving[k]:
                    index_moving=k
                    break
            if index_moving==-1:
                continue
            record=str(timestamp_moving[index_moving])+","+str(moving_list[index_moving])+","+str(GT_moving[index_moving])+","+str(detect_motion_list[i])+","+str(GT_motion_list[i])+","+str(seg_list[index_moving][0])+","+str(seg_list[index_moving][1])+","+str(seg_list[index_moving][2])+","+str(seg_list[index_moving][3])+","+str(detect_motion_list_old[i])+"\n"
            f.writelines(record)
        f.close()
                    
def read_state_list(path):
    f=open(path,'r')
    lines=f.readlines()
    state=[]
    real=[]
    label=[]
    for line in lines:
        data=line.split(",")
        state.append([np.float32(data[0]),np.float32(data[1])])
        real.append(np.float32(data[2]))
        label.append(np.float32(data[3][0]))
    state=np.array(state)
    real=np.array(real)
    label=np.array(label)
    return state,real,label

def motion_state_smooth(real_state,timestamp,time_thre,len_win=5): #a sliding window in a threshold, the main reason why latency is high.
    #print(time_thre)
    
    for i in range(len(real_state)):
        #print(i,real_state[i])
        #print(" ")
        if i == 0 and len(real_state)>=3:
            if real_state[1]==real_state[2] and real_state[0]!=real_state[1]:
                if timestamp[2]-timestamp[0]>time_thre or timestamp[1]-timestamp[0]>time_thre:
                    continue
                real_state[0]=real_state[1]
            else:
                continue
        if i == 1 and len(real_state)>=3:
            #print(i,"smooth")
            if timestamp[2]-timestamp[1]>time_thre:
                continue
            if real_state[0]==real_state[2] and real_state[0]!=real_state[1]:
                if len(real_state)>=4:
                    if timestamp[3]-timestamp[1]>time_thre:
                        continue
                    if real_state[3]==real_state[2]:
                        real_state[1]=real_state[0]
                else:
                    real_state[1]=real_state[0]
            else:
                continue
        if i+int(len_win/2.0)<len(real_state):
            Flag_skip=False
            if i!=0 and real_state[i]==real_state[i-1]:
                continue
            elif i==0 and real_state[i]==real_state[i+1]:
                continue
            else:
                if i+int(len_win/2.0)<len(real_state):
                    if real_state[i]==real_state[i-1]:
                        continue
                    else:
                        same_count=0
                        diff_index=-1
                        diff_count=0
                        for k in range(-1*int(len_win/2.0),int(len_win/2.0)+1):
                            #print(k)
                            if i+k<0:
                                continue
                            if k>0 and np.abs(timestamp[i+k]-timestamp[i])>=time_thre:
                                if k==1 and i>=3 and real_state[i-2]!=real_state[i-3]:
                                    #continue, you can try to skip this step
                                    diff_count=0
                                    same_count=0
                                    break
                                continue
                            if k>0 and timestamp[i+k]-timestamp[i]<0:
                                break
                            if real_state[i+k]==real_state[i]:
                                same_count+=1
                            else:
                                diff_count+=1
                                diff_index=i+k
                            #print(i+k,timestamp[i+k],real_state[i+k],diff_count,same_count)
                        if diff_count>same_count:
                            real_state[i]=real_state[diff_index]
                #print(" ")
        else:
            if real_state[i]==real_state[i-1]:
                #break
                continue
            else:
                same_count=0
                diff_index=-1
                diff_count=0
                for k in range(-1*int(len_win/2.0),len(real_state)-i):
                    if i+k<0:
                        #print(i+k,k,"skip")
                        continue
                    if k>0 and np.abs(timestamp[i+k]-timestamp[i])>=time_thre:
                        if k==1 and i>=3 and real_state[i-2]!=real_state[i-3]:
                            #continue, you can try to skip this step
                            diff_count=0
                            same_count=0
                            break
                        continue
                    #print(i+k,timestamp[i+k])
                    if k>0 and timestamp[i+k]-timestamp[i]<0:
                        break
                    if real_state[i+k]==real_state[i]:
                        same_count+=1
                    else:
                        diff_count+=1
                        diff_index=i+k
                    #print(i+k,timestamp[i+k],diff_count,same_count)
                if diff_count>same_count:
                    real_state[i]=real_state[diff_index]
    #print(" ")
    return real_state

                
def SSE(location):
    mean_pos=[0,0]
    for i in range(len(location)):
        mean_pos[0]+=location[i][0]
        mean_pos[1]+=location[i][1]
    mean_pos[0]/=len(location)
    mean_pos[1]/=len(location)
    SSE_array=np.zeros(len(location))
    count=0
    for i in range(len(location)):
        SSE_array[i]=np.abs(np.sqrt((location[i][0]-mean_pos[0])**2+(location[i][1]-mean_pos[1])**2))
        count+=1
    SSE_loss_var=np.std(SSE_array)
    SSE_loss_mean=np.mean(SSE_array)
    return SSE_loss_mean,SSE_loss_var

class swimmer_state:
    def __init__(self,state_list,length,time_per_frame=2.57,time_check=30.0):
        self.motion=[]                                                          
        self.location=[]
        self.motion_score=[]
        self.frequency_score=[]
        self.state_list=state_list
        self.timestamp=[]
        self.length=length
        self.duration=[]
        self.duration_ratio=[]
        self.list_full=False
        self.state_transfer_machine=None
        self.time_per_frame=time_per_frame
        self.location_info=[]
        self.SSE_mean=[]
        self.SSE_var=[]
        self.cur_move=None
        self.bias=0
        self.time_check=time_check
            
    def update_move(self,motion,location,timestamp):
        self.motion.append(motion)
        self.location.append(location)
        self.timestamp.append(timestamp)
        
    def state_smooth(self,windows,step): #useless methods, please drop it.
        motion_filter=[]
        re_motion=np.array(self.motion)
        for i in range(0,len(self.motion),step):
            if i-np.int32(windows/2.0)<0 or i-np.int32(windows/2.0):
                motion_filter.append(self.motion[i])
            else:
                data=[0,0]
                data[0]=np.mean(re_motion[i-np.int32(windows/2.0):i-np.int32(windows/2.0)][0])
                data[1]=np.mean(re_motion[i-np.int32(windows/2.0):i-np.int32(windows/2.0)][1])
                motion_filter.append(data)
                
        self.motion=motion_filter
        
    def update_timestamp(self,time):
        self.timestamp.append(time)
        while len(self.timestamp)>self.length:
            self.timestamp.remove()
        
    def update_state(self,state):
        self.state_list.append(state)
        while len(self.state_list)>self.length:
            self.state_list.remove(self.state_list[0])
    
    def update_motion_score(self):
        still_conf=self.motion[-1][0]
        motion_conf=self.motion[-1][1]
        diff=still_conf-motion_conf
        still_score=(1.0+diff)/2
        motion_score=1-still_score
        self.motion_score=[still_score,motion_score]
        return self.motion_score
        
    def update_location_score(self,times=5.0):
        if len(self.location)>1:
            diff_loc=self.location[len(self.location)-1]-self.location[len(self.location)-2]
            time_late=self.timestamp[len(self.location)-1]
            time_early=self.timestamp[len(self.location)-2]
            velocity=diff_loc[len(diff_loc)-1]/(time_late-time_early)
            distance=np.sqrt((self.location[len(self.location)-1][0]-self.location[len(self.location)-2][0])**2+(self.location[len(self.location)-1][1]-self.location[len(self.location)-2][1])**2)
            location_dict={}
            location_dict.update({"loc_diff":diff_loc})
            location_dict.update({"velocity":velocity})
            location_dict.update({"distance":distance})
            self.location_info.append(location_dict)
            #time_cur=self.timestamp[0]
            index_time=len(self.timestamp)-1
            while index_time>0:
                if self.timestamp[len(self.timestamp)-1]-self.timestamp[index_time]<times:
                    index_time-=1
                else:
                    break
            SSE_loss_mean,SSE_loss_var=SSE(self.location[index_time:len(self.timestamp)])
            #cur_index=0
            self.SSE_mean.append(SSE_loss_mean)
            self.SSE_var.append(SSE_loss_var)
        else:
            location_dict={}
            location_dict.update({"loc_diff":[0,0]})
            location_dict.update({"velocity":0})
            location_dict.update({"distance":0})
            self.location_info.append(location_dict)
            self.SSE_mean.append(0)
            self.SSE_var.append(0)
        while self.length<len(self.location):
            self.location_info.remove(self.location_info[0])
            self.location.remove(self.location[0])
            self.timestamp.remove(self.timestamp[0])
            self.SSE_mean.remove(self.SSE_mean[0])
            self.SSE_var.remove(self.SSE_var[0])
            
            
    def update_motion_frequency(self):
        bias=self.bias
        Motion_list=[]
        F=0
        if self.cur_move!="C":
            time_start=self.timestamp[len(self.timestamp)-len(self.motion)]
        else:
            time_start=self.timestamp[-1]
        time_end=0.0
        time_list=[]
        M_count=0
        S_count=0
        if len(self.motion)<=self.length:
            for i in range(len(self.motion)):
                #if i == 0 and self.motion[i][1]==1: #the first frame can be considered or not due to lack of movement patterns
                #    F=1
                if self.motion[i][1]==1:
                    M_count+=1
                else:
                    S_count+=1
                if i <= len(self.motion)-2:
                    if self.motion[i][1]==1 and self.motion[i+1][1] == 0:
                        Motion_list.append(F)
                        time_end=self.timestamp[i+bias]
                        time_list.append(time_end-time_start)
                        F=0
                    elif self.motion[i][1]==1 and self.motion[i+1][1]==1:
                        F+=1
                    elif self.motion[i][1]==0 and self.motion[i+1][1]==1:
                        F=1
                        time_start=self.timestamp[i+bias+1]
                    else:
                        continue
                if i==len(self.motion)-1 and F!=0:
                    Motion_list.append(F)
                    time_end=self.timestamp[i+bias]
                    time_list.append(time_end-time_start)
        else:
            for i in range(len(self.motion)-self.length,self.length):
                if i == 0 and self.motion[i][1]==1:
                    F=1
                if self.motion[i][1]==1:
                    M_count+=1
                else:
                    S_count+=1
                if i <= len(self.motion)-2:
                    if self.motion[i][1]==1 and self.motion[i+1][1] == 0:
                        Motion_list.append(F)
                        time_end=self.timestamp[i+bias]
                        time_list.append(time_end-time_start)
                        F=0
                    elif self.motion[i][1]==1 and self.motion[i+1][1]==1:
                        F+=1
                    elif self.motion[i][1]==0 and self.motion[i+1][1]==1:
                        F=1
                        time_start=self.timestamp[i+bias+1]
                    else:
                        continue
                if i==len(self.length)-1 and F!=0:
                    Motion_list.append(F)
                    time_end=self.timestamp[i+bias]
                    time_list.append(time_end-time_start)
        if self.motion!=[]:
            if self.motion[-1][1]==1:
                Motion_list.append(F)
                time_list.append(self.timestamp[-1]-time_start)
                    
        length_check=0
        for i in range(len(self.timestamp)):
            if self.timestamp[len(self.timestamp)-1]-self.timestamp[len(self.timestamp)-i-1]<=self.time_check:
                length_check+=1
            else:
                break
        if len(Motion_list)>0:
            frequency_dict={}
            frequency_dict.update({"frequency_score_list":Motion_list})
            frequency_dict.update({"latest":Motion_list[-1]})
            Motion_list=np.array(Motion_list)
            F_mean=np.mean(Motion_list)
            F_var=np.var(Motion_list)
            frequency_dict.update({"average":F_mean})
            frequency_dict.update({"var":F_var})
            frequency_dict.update({"len":len(self.motion)})
            frequency_dict.update({"latest_time":time_list[-1]})
            frequency_dict.update({"ratio":M_count/(M_count+S_count)})
            frequency_dict.update({"check_fre":length_check})
            self.frequency_score.append(frequency_dict)
        else:
            frequency_dict={}
            frequency_dict.update({"frequency_score_list":Motion_list})
            frequency_dict.update({"latest":0})
            Motion_list=np.array(Motion_list)
            F_mean=np.mean(Motion_list)
            F_var=np.var(Motion_list)
            frequency_dict.update({"average":F_mean})
            frequency_dict.update({"var":F_var})
            frequency_dict.update({"len":len(self.motion)})
            frequency_dict.update({"latest_time":0})
            frequency_dict.update({"ratio":0})
            frequency_dict.update({"check_fre":length_check})
            self.frequency_score.append(frequency_dict)
        while len(self.frequency_score)>self.length:
            self.frequency_score.remove(self.frequency_score[0])
        
    def update_duration(self):
        count=0
        if len(self.state_list)>=2:
            count=1
            length=len(self.state_list)
            for i in range(1,length):
                if i!=length and self.state_list[length-i]==self.state_list[length-i-1]:
                    count+=1
                else:
                    break
            self.duration.append([self.timestamp[len(self.timestamp)-1]-self.timestamp[len(self.timestamp)-count-1],self.timestamp[-1]-self.timestamp[0]])

        elif len(self.state_list)==1:
            self.duration.append([self.timestamp[-1],self.timestamp[-1]-0.0])
        else:
            self.duration.append([0,0.0])
        while len(self.duration)>self.length:
            self.duration.remove(self.duration[0])
    
    def update_duration_ratio(self): #useless 
        time_win=0.0
        if len(self.state_list)==0:
            self.duration_ratio=0.0
            return
        if self.state_list[-1]=="motionless":
            time_win=60
        elif self.state_list[-1]=="struggling":
            time_win=20
        else:
            self.duration_ratio=0.0
            return
        count=0
        total_count=0
        if self.timestamp[len(self.state_list)-1]-self.timestamp[0]<time_win:
            self.duration_ratio=0.0
            return
        for i in range(len(self.state_list)):
            if self.timestamp[len(self.state_list)-1]-self.timestamp[len(self.state_list)-1-i]<time_win:
                total_count+=1
                if self.state_list[len(self.state_list)-1]==self.state_list[len(self.state_list)-1-i]:
                    count+=1
        self.duration_ratio=count*1.0/total_count
        
    def update(self,motion,location,timestamp): #useless now
        self.update_move(motion,location,timestamp)
        self.update_motion_score()
        self.update_location_score()
        self.update_motion_frequency()
        self.update_duration()
        self.update_duration_ratio()
        if self.state_transfer_machine==None:
            self.state_transfer_machine=state_transfer()
        state=self.state_transfer_machine.transfer_state()
        self.state_list.append(state)
        while len(self.state_list)>self.length:
            self.state_list.remove(self.state_list[0])
        
    def update_features(self,state_new,timestamp,move='U'):
        new_state=[0,0]
        if state_new[0]=='S':
            new_state=[1,0]
        else:
            new_state=[0,1]
        self.cur_move=move
        self.motion.append(new_state)
        if self.cur_move=="C":
            self.bias+=len(self.motion)
            self.motion=[]
        self.timestamp.append(timestamp)
        while len(self.motion)>self.length:
            self.motion.remove(self.motion[0])
            self.timestamp.remove(self.timestamp[0])
        self.update_motion_frequency()
        self.update_duration()
        self.update_duration_ratio()
        return self.frequency_score,self.duration,self.duration_ratio
    
class state_transfer:
    def __init__(self,cur_state, motion_score=None, location_info=None, duration=None,duration_ratio=None, frequency=None,check_fre=10, state=['moving','motionless','patting','struggling','drowning'],time_frame=2.57,check_fre_time=30.0):
        self.swim_state=state
        self.motion_score=motion_score
        self.localtion_score=location_info
        self.frequency=frequency
        self.duration=duration
        self.duration_ratio=duration_ratio
        self.cur_state=cur_state
        self.check_fre=check_fre #useless, can delete
        self.dronwing_mark={
            "Motion":['S','M'],
            "Location":['U','C'],
            "Frequency":['S','F'], #here 'S' is the 'P', S is low motion change frequency, meaning motion is prolonged.
            #"Velocity":['F','S'],
            "Duration":['S','L']
        }
        self.weight=[0.25,0.25,0.25,0.25]
        self.state_list=state
        self.time_frame=time_frame
        self.check_fre_time=check_fre_time
    
    def update_cur_state(self,state):
        self.cur_state=state
        
    def transfer_state_mark(self,out_input=False, action=None):
        if out_input:
            state_new=action
        else:
            state_new=[self.check_motion_state,self.check_location_changed,self.check_frequency_switch,self.check_state_duration]
        if self.cur_state==None:
            if state_new[0]=='M':
                return self.state_list[2]
            else:
                return self.state_list[1]
        
        if self.cur_state==self.state_list[0]:
            if state_new[1]=='C':
                return self.state_list[0]
            elif state_new[0]=='M' and state_new[1]=='U':
                return self.state_list[2]
            elif state_new[0]=='S' and state_new[1]=='U':
                return self.state_list[1]
            else:
                return self.cur_state
            
        if self.cur_state==self.state_list[1]:
            if state_new[1]=='C':
                return self.state_list[0]
            elif state_new[0]=='M' and state_new[1]=='U':
                return self.state_list[2]
            elif state_new[0]=='S' and state_new[1]=='U':
                if state_new[0]=='S' and state_new[3]=='L':
                    return self.state_list[4]
                return self.state_list[1]
            else:
                return self.cur_state
        
        if self.cur_state==self.state_list[2]:
            if state_new[1]=='C':
                return self.state_list[0]
            elif state_new[0]=='M' and state_new[1]=='U' and state_new[2]=="S":
                return self.state_list[3]
            elif state_new[0]=='M' and state_new[1]=='U':
                return self.state_list[2]
            elif state_new[0]=='S' and state_new[1]=='U':
                return self.state_list[1]
            else:
                return self.cur_state
            
        if self.cur_state==self.state_list[3]:
            if state_new[3]=="L":
                return self.state_list[4]
            else:
                return self.cur_state
            
        if self.cur_state==self.state_list[4]:
            return self.cur_state
        
        
    def check_motion_state(self):
        motion_score=self.motion_score[-1]
        if motion_score[1]==1:
            return self.dronwing_mark['motion'][1]
        else:
            return self.dronwing_mark['motion'][0]
        
    def check_frequency_switch(self):

        frequency_score=self.frequency[-1]
        check_fre=frequency_score['check_fre']
        frequency_score_ratio=frequency_score['latest']      
        if frequency_score_ratio==0:
            return self.dronwing_mark["Frequency"][1]
        if frequency_score_ratio>=check_fre*0.95 and frequency_score['latest_time']>=self.check_fre_time*0.95:
            return self.dronwing_mark["Frequency"][0]
        else:
            return self.dronwing_mark["Frequency"][1]     
        
    def check_state_duration(self):
        duration_info=self.duration[-1]
        duration_ratio=self.duration_ratio #useless
        if self.cur_state=="motionless":
            if duration_info[0]>=60:
                return self.dronwing_mark["Duration"][1]
            else:
                return self.dronwing_mark["Duration"][0]
        else:
            if duration_info[0]>=20:
                return self.dronwing_mark["Duration"][1]
            else:
                return self.dronwing_mark["Duration"][0]
        
    def check_location_changed(self):
        location_ve=self.localtion_score['velocity']
        location_info=self.localtion_score['distance']
        if location_info>2.5 or location_ve>1.0:
            return self.dronwing_mark['Location'][1]
        else:
            return self.dronwing_mark['Location'][0]
        
def sample(time_stamp,state,scan_time,GT):
    new_state=[]
    new_time=[]
    new_GT=[]
    count=1
    new_state.append(state[0])
    new_time.append(time_stamp[0])
    new_GT.append(GT[0])
    for i in range(1,len(time_stamp)):
        if time_stamp[i-1]<=count*scan_time and time_stamp[i]>=count*scan_time:
            new_time.append(time_stamp[i])
            new_state.append(state[i])
            new_GT.append(GT[i])
            count+=1
    return new_state,new_time,new_GT
    
def confusion_metrix(GT,detect):
    state=['moving','motionless','patting','struggling','drowning']
    detect_re=np.zeros((len(state),len(state)))
    #dict_re=[[0,0],[0,0],[0,0],[0,0],[0,0]]
    count=0
    correct=0
    for i in range(len(GT)):
        if GT[i]=="none":
            continue
        if GT[i]==detect[i]:
            correct+=1
            #dict_re[state.index(GT[i])][0]+=1
        detect_re[state.index(GT[i])][state.index(detect[i])]+=1
        count+=1
    if count==0.0:
        ratio=0
    else:
        ratio=correct*1.0/count
    #class_num=np.zeros(len(state))
    detect_re_per=np.zeros((len(state),len(state)))
    for i in range(len(detect_re)):
        class_num=np.sum(detect_re[i])
        for j in range(len(detect_re[i])):
            detect_re_per[i][j]=detect_re[i][j]/class_num
    return correct,count,ratio,detect_re,detect_re_per
        
def eval_all(har_dir,save_dir,cfg_har=[]):
    dir_create(save_dir)
    state=['moving','motionless','patting','struggling','drowning']
    files=os.listdir(har_dir)
    if ".DS_Store" in files:
        files.remove(".DS_Store")
    detect_re=[]
    detect_gt=[]
    #print(files)
    for file in files:
        scenario=file.split('_')[0]
        sonar=file.split("_")[1]
        if cfg_har==['0.0']:
            time_thre=-1
            time_single=3.0 #config, time_thre for re_smooth of first activty, time_single is useless now, you can set it as 0
            time_thre=time_single*2+np.int32(time_single)-1
        else:
            time_single=3.0 #config, time_thre for re_smooth of first activty, time_single is useless now, you can set it as 0
            time_thre=np.float32(cfg_har[0])
            
        check_fre=np.int32(30.0/time_single)+1
        check_fre_time=30.0
        read_har_single=har_dir+"/"+file
        save_path=save_dir+'/'+file
        timestamp,moving_flag_list,moving_GT_list,motion_flag_list,motion_GT_list,seg_list=read_har_file(read_har_single,time_thre) # resmooth the first 2 states when their following state changes. This function works after smoothing motion and movement. This design is proposed due to lack of past information. To reduce the overhead, we have a strict re-smooth conditions to not easily trigger it. In most of cases, this will not be triggered so the latency of first 2 states will not increase.
        state_score=swimmer_state(state_list=[],length=40,time_per_frame=time_single,time_check=check_fre_time) 
        state_score_gt=swimmer_state(state_list=[],length=40,time_per_frame=time_single,time_check=check_fre_time) 
        state_detected=None
        state_detected_gt=None    
        single_list_detected=[]
        single_list_GT=[]    
        for i in range(len(timestamp)):

            detect_mark=motion_flag_list[i]
            GT_mark=motion_GT_list[i]
            detect_one=[detect_mark,moving_flag_list[i]]
            GT_one=[GT_mark,moving_GT_list[i]]


            state_score.update_features(detect_one,timestamp[i],moving_flag_list[i])
            fre_info,duration,duration_ratio=state_score.frequency_score,state_score.duration,state_score.duration_ratio
            state_transfer_re=state_transfer(cur_state=state_detected,frequency=fre_info,duration=duration,duration_ratio=duration_ratio,state=state,check_fre=check_fre,time_frame=time_single,check_fre_time=check_fre_time)
            state_fre=state_transfer_re.check_frequency_switch()
            state_dur=state_transfer_re.check_state_duration()
            state_detect_list=[detect_one[0],detect_one[1],state_fre,state_dur]

            state_final=state_transfer_re.transfer_state_mark(out_input=True,action=state_detect_list)
            # we use out_input since we generate motion and moving mark during read_har_file. 
            detect_re.append(state_final)
            state_score.update_state(state_final)  
            state_transfer_re.update_cur_state(state_final) 
            state_detected=state_final
                
            state_score_gt.update_features(GT_one,timestamp[i],moving_GT_list[i]) #only for reference, not used for inference and metric calculation, you can even randomly input the gt.
            fre_info_gt,duration_gt,duration_ratio_gt=state_score_gt.frequency_score,state_score_gt.duration,state_score_gt.duration_ratio
            state_transfer_re_gt=state_transfer(cur_state=state_detected_gt,frequency=fre_info_gt,duration=duration_gt,duration_ratio=duration_ratio_gt,state=state,check_fre=check_fre,time_frame=time_single,check_fre_time=check_fre_time)
            state_fre_gt=state_transfer_re_gt.check_frequency_switch()
            state_dur_gt=state_transfer_re_gt.check_state_duration()
            state_detect_list_gt=[GT_one[0],GT_one[1],state_fre_gt,state_dur_gt]
            ]
            state_final_gt=state_transfer_re_gt.transfer_state_mark(out_input=True,action=state_detect_list_gt) #useless now, just for reference
            detect_gt.append(state_final_gt)
            state_score_gt.update_state(state_final_gt)  
            state_transfer_re_gt.update_cur_state(state_final_gt) 
            state_detected_gt=state_final_gt
            #print(state_detected,state_detected_gt)
            
            single_list_detected.append(state_detected)
            single_list_GT.append(state_detected_gt)
            #print(state_detected,state_detected_gt)
        f=open(save_path,'w')
        for i in range(len(seg_list)):
            record=str(timestamp[i])+" "+str(single_list_detected[i])+" "+str(single_list_GT[i])+" "+str(seg_list[i][0])+" "+str(seg_list[i][1])+" "+str(seg_list[i][2])+" "+str(seg_list[i][3])+"\n"
            f.writelines(record)
        f.close()
    print(len(detect_re),len(detect_gt))
    return detect_re, detect_gt

def compare_swimming(gt_dir,detect_dir,time_single,target,target_file=[],type_label=0,time_sample=100,gt_config=[]):
    state=['moving','motionless','patting','struggling','drowning']
    #print(target)

    gt_dataset,gt_dict=generate_gt_label(gt_dir,time_single,target,type_label,gt_config)

    detect_label=generate_detect_label(detect_dir,target,target_file)
    appear_moving=[]
    gt_sort=[]
    detect_sort=[]
    #print(gt_dict)
    #print(" ")
    #print(detect_label)
    #print(len(gt_dict))
    #print(len(detect_label))
    for i in range(len(detect_label)):
        obj1=detect_label[i][2]
        max_iou_temp=0.0
        max_index=-1
        #print(detect_label[i])
        key_file=detect_label[i][3]
        scenario=key_file.split('_')[0]
        sonar=key_file.split('_')[1]
        time_one=detect_label[i][0]
        dict_key=scenario+"_"+sonar+"_"+str(time_one)
        #print(dict_key)
        if dict_key not in gt_dict:
        #    print("notfind")
            continue
        gt_single=gt_dict[dict_key]
        #print(dict_key,gt_single)
        #print(" ")
        if gt_single==[]:
            continue
        for j in range(len(gt_single)):
            #print(len(gt_dataset[i]))
            obj2=gt_single[j][2]
            iou_temp=cal_IoU(obj1,obj2)
            #print(iou_temp,obj2)
            if iou_temp>max_iou_temp:
                max_iou_temp=iou_temp
                max_index=j
        if gt_single[max_index][1]=="moving":
            if np.float128(time_one)>time_sample:
                continue
            if [detect_label[i][0],detect_label[i][1],detect_label[i][2]] not in appear_moving:
                if time_one==0.0 or time_one=="0.0": #not record the first moving results since we cannot recognize the first moving activity, it is only the start point of swimming, you can recognize both swimming and non-swimming.
                    continue
                if max_index!=-1:
                    gt_sort.append(gt_single[max_index][1])
                    detect_sort.append(detect_label[i][1])
                else:
                    gt_sort.append("none")
                    detect_sort.append(detect_label[i][1])
                appear_moving.append([detect_label[i][0],detect_label[i][1],detect_label[i][2]])
    return gt_sort,detect_sort

def positive_negative_cls(matrix):
    p_n_m=np.zeros((2,2))
    count_p_p=0
    for i in range(0,3):
        for j in range(0,3):
            count_p_p+=matrix[i][j]
    
    count_n_p=0
    for i in range(3,5):
        for j in range(0,3):
            count_n_p+=matrix[i][j]
    count_p_n=0
    for i in range(0,3):
        for j in range(3,5):
            count_p_n+=matrix[i][j]
    count_n_n=0
    for i in range(3,5):
        for j in range(3,5):
            count_n_n+=matrix[i][j]
    count_p=count_p_p+count_p_n
    count_n=count_n_n+count_n_p
    p_n_m[0][0]=count_p_p
    p_n_m[0][1]=count_p_n
    p_n_m[1][0]=count_n_p
    p_n_m[1][1]=count_n_n
    m_r=np.zeros((2,2))
    m_r[0][0]=count_p_p/count_p
    m_r[0][1]=count_p_n/count_p
    m_r[1][0]=count_n_p/count_n
    m_r[1][1]=count_n_n/count_n
    return p_n_m,m_r

def overall(martix):
    count_all=0
    correct=0
    for i in range(len(martix)):
        for j in range(len(martix[i])):
            count_all+=martix[i][j]
            if i==j:
                correct+=martix[i][j]
    return correct*1.0/count_all


def compare_metirc(gt_dir,detect_dir,moving_dir,time_single=2.57,target=[],target_file=[],dis=13.0,label_type=0,sampel_time=90,gt_config=[]): #time_single is useless now
    #dis is useless now. 
    state=['moving','motionless','patting','struggling','drowning']
    gt_dataset,gt_dict=generate_gt_label(gt_dir,time_single,target,label_type,gt_config)

    detect_label=generate_detect_label(detect_dir,target,target_file)
    appear_detect=[]
    match_list=[]
    gt_sort=[]
    detect_sort=[]
    #print(detect_label[0])
    for i in range(len(detect_label)):
        obj1=detect_label[i][2]
        max_iou_temp=0.0
        max_index=-1
        #print(detect_label[i])
        key_file=detect_label[i][3]
        scenario=key_file.split('_')[0]
        sonar=key_file.split('_')[1]
        time_one=detect_label[i][0]
        dict_key=scenario+"_"+sonar+"_"+str(time_one)
        #print(dict_key)
        #print()
        if dict_key not in gt_dict:
        #    print("notfind")
            continue
        gt_single=gt_dict[dict_key]
        #print()
        #print(" ")
        #print(dict_key,gt_single)
        #print(" ")
        if gt_single==[]:
            continue
        for j in range(len(gt_single)):
            #print(len(gt_dataset[i]))
            obj2=gt_single[j][2]
            iou_temp=cal_IoU(obj1,obj2)
            #print(iou_temp,obj2)
            if iou_temp>max_iou_temp:
                max_iou_temp=iou_temp
                max_index=j
        if np.float128(detect_label[i][0])>=sampel_time:
            continue
        if gt_single[max_index][1]=="moving": #not compare moving now, moving will be compared in compare_swimming.
            continue
        if [detect_label[i][0],detect_label[i][1],detect_label[i][2]] not in appear_detect: #not repeatedly record results.
            if max_index!=-1:
                gt_sort.append(gt_single[max_index][1])
                detect_sort.append(detect_label[i][1])
            else:
                gt_sort.append("none")   # None is not shown in final results. It is better to combine object detection and activity recognition together. Current results are shown seperately. Confusion metric contains the GT of detected subjects and object detection show the miss and falsed detection. 
                detect_sort.append(detect_label[i][1])
            match_list.append([detect_label[i],gt_single[max_index]])
            appear_detect.append([detect_label[i][0],detect_label[i][1],detect_label[i][2]])
    gt_sort_moving,detect_sort_moving=compare_swimming(gt_dir,moving_dir,time_single,target,target_file,label_type,sampel_time,gt_config)
    for i in range(len(gt_sort_moving)):
        gt_sort.append(gt_sort_moving[i])
        detect_sort.append(detect_sort_moving[i])
    for i in range(len(match_list)):
        #print(match_list[i])
        pass
    correct,count,ratio,detect_re_,detect_re_per=confusion_metrix(gt_sort,detect_sort)
    print(correct,count,ratio)
    #print(detect_re_)
    
    print("all_ratio")
    print(detect_re_per)
    print(correct,count,ratio)
    print("all_num")
    print(detect_re_)
    return detect_re_

def compare_results(gt_dir,detect_dir,moving_dir,target,dis,label_type,sample,gt_config):
    count_matrix=np.zeros((5,5))
    ratio_matrix=np.zeros((5,5))
    for i in range(len(target)):
        target_scenario_single=target[i]
        time_single=3.0 # config time_single here, time_single is disabled, so it is useless now.
        target_scenario=[]
        target_scenario.append(target_scenario_single)
        time_record=time.time()
        matrix_single=compare_metirc(gt_dir,detect_dir,moving_dir,time_single,target_scenario,dis=dis,label_type=label_type,sampel_time=sample,gt_config=gt_config)
        print(time.time()-time_record)
        count_matrix+=matrix_single
    ratio_matrix=matrix2ratio(count_matrix)
    return count_matrix
        
    
def save_result(har_dir,detect_dir,har_cfg):
    eval_all(har_dir,detect_dir,har_cfg)

def main_merge_results(motion_dir,moving_dir,har_save,start_cfg,smooth_cfg):
    motion_dir=motion_dir 
    moving_dir=moving_dir 
    save_dir=har_save 
    state_merge_motion_temporal(motion_dir, moving_dir,save_dir,start_cfg,smooth_cfg)
    
def gt_config_all(gt):
    gt_dict={"111":[29,50,60]} #replace 111 with dir_name, replace [] with gt time config
    cfg=gt_dict[gt]
    return cfg

def gt_config_base(gt):
    gt_dict={"111":[38,60,70],
             }
    cfg=gt_dict[gt]
    return cfg
        
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", type=str, required=True, help="motion")
    parser.add_argument("--moving", type=str, required=True, help="moving")
    parser.add_argument("--har", type=str, required=True, help="har")
    parser.add_argument("--detect", type=str, required=True, help="detect")
    parser.add_argument("--label", type=str, required=True, help="detect")
    parser.add_argument("--har_cfg", type=str, required=True, help="har_cfg")
    parser.add_argument("--smooth_cfg", type=str, required=True, help="smooth_cfg")
    parser.add_argument("--start_cfg", type=str, required=True, help="start_cfg")
    parser.add_argument("--gt_cfg", type=str, required=True, help="gt_cfg")
    parser.add_argument("--gt_mode", type=int, required=True, help="gt_cfg")
    parser.add_argument("--gt_trans",type=int, required=True, help="gt_trans")
    parser.add_argument("--gt_sc",type=str, nargs="+",required=True, help="gt_trans")
    parser.add_argument("--dis", type=int, required=True, help="distance")
    parser.add_argument("--label_type",type=int, required=True, help="label_type")
    parser.add_argument("--sample",type=int,required=True, help="sample")
    parser.add_argument("--save",type=str,required=True, help="save")
    parser.add_argument("--save_dir_all",type=str,required=True, help="save_dir_all")
    args = parser.parse_args()
    save_dir_all=args.save_dir_all
    sample_=args.sample
    motion_dir=save_dir_all+"/"+args.motion
    moving_dir=save_dir_all+"/"+args.moving
    har_dir=save_dir_all+"/"+args.har
    detect_dir=save_dir_all+"/"+args.detect
    label_gt_dir=args.label
    label_type=args.label_type
    dis_=args.dis #useless
    # arg.trans is disabled, you can use it to control the gt config
    
    har_cfg=[args.har_cfg]
    smooth_cfg=[args.smooth_cfg]
    start_cfg=[args.start_cfg]
    gt_trans=args.gt_trans
    main_merge_results(motion_dir,moving_dir,har_dir,start_cfg,smooth_cfg)
    save_result(har_dir,detect_dir,har_cfg)
    scenario=args.gt_sc
    if args.gt_mode==0:
        gt_config=gt_config_all(args.gt_cfg)
    else:
        gt_config=gt_config_base(args.gt_cfg)

    m_re=compare_results(label_gt_dir,detect_dir,detect_dir,scenario,dis=dis_,label_type=label_type,sample=sample_,gt_config=gt_config)
    print(m_re) 
    save_dir=args.save
    np.save(save_dir,m_re)
