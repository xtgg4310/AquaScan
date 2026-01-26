import numpy as np
import os
import time
import argparse

def calculate_iou_on_small(obj1, obj2):
    ymin = max(obj1[0], obj2[0])
    ymax = min(obj1[1], obj2[1])
    xmin = max(obj1[2], obj2[2])
    xmax = min(obj1[3], obj2[3])
    if ymin >= ymax or xmin >= xmax:
        return 0
    inter = (ymax - ymin) * (xmax - xmin)
    obj1_area = (obj1[1] - obj1[0]) * (obj1[3] - obj1[2])
    obj2_area = (obj2[1] - obj2[0]) * (obj2[3] - obj2[2])
    iou = inter / min(obj1_area, obj2_area)
    return iou

def calculate_iou(obj1, obj2):
    '''pos: [ymin, ymax, xmin, xmax]
    '''
    ymin = max(obj1[0], obj2[0])
    ymax = min(obj1[1], obj2[1])
    xmin = max(obj1[2], obj2[2])
    xmax = min(obj1[3], obj2[3])
    if ymin >= ymax or xmin >= xmax:
        return 0
    inter = (ymax - ymin) * (xmax - xmin)
    union = (obj1[1] - obj1[0]) * (obj1[3] - obj1[2]) + (obj2[1] - obj2[0]) * (obj2[3] - obj2[2]) - inter
    return inter / union
    
def dir_create(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
        
def readline(line):
    ''' read one line of txt file
    Args:
        line: one line of txt file. Format: angle data0 data1 ... dataX
        By default, each line has 1 + 500 data.
    Returns:
        angle: angle of this line
        data: sonar data
    '''
    line = line.split()
    line = list(map(float, line))
    angle = line[0]
    data = line[1:]
    return angle, data
        
def read_txt(path, angle_range=400,bias=0):
    ''' read sonar data from txt file
    Args:
        path: path of txt file
        angle_range: range of angle (1 gradian = 0.9 degree), default: 400
    Returns:
        sonar_data: 2d array, shape: (angle_range, 500)
        start_angle: start angle of sonar data
        end_angle: end angle of sonar data
    '''
    path_seg=path.split('/')[-1]
    if '_' in path_seg:
        file_name=str(path_seg.split('_')[1])
    else:
        file_name=path_seg
    file_order=file_name.split('.')[0]
    sonar_data = np.zeros((angle_range, 500))
    with open(path, 'r') as f:
        lines = f.readlines()
        start_angle = float(lines[0].split(' ')[0])
        end_angle = float(lines[-1].split(' ')[0])
        for line in lines:
            angle, data = readline(line)
            if len(data) == 500:
                if np.int32(file_order)%2==1:
                    sonar_data[(int(angle)+bias)%400] = data
                else:
                    sonar_data[int(angle)] = data
    return sonar_data, int(start_angle), int(end_angle)

def read_trace_single(file):
    f=open(file,'r')
    lines=f.readlines()
    pos=[]
    seg=[]
    timestamp=[]
    state=[]
    for line in lines:
        line=line.split(" ")
        time=float(line[0])
        pos_single=[float(line[1]),float(line[2])]
        seg_single=[float(line[3]),float(line[4]),float(line[5]),float(line[6])]
        state_single=""
        raw_state=line[7][:-1]
        if raw_state=="moving" or raw_state=="swimming":
            state_single="moving"
        elif raw_state=="none":
            state_single="none"
        else:
            state_single="non-moving"
        raw_state=line[7][:-1]
        state_single_list=[state_single,raw_state]
        timestamp.append(time)
        pos.append(pos_single)
        seg.append(seg_single)
        state.append(state_single_list)
    return timestamp,pos,seg,state

def read_yolo_label(file_path, W=500, H=400,bias=9):
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

def read_default_label_raw(file_path):
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
                xmin = int(float(arr[4])) #4
                ymin = int(float(arr[2])) #2
                xmax = int(float(arr[5])) #5
                ymax = int(float(arr[3])) #3
            
            # noise object
            elif len(arr) == 5:
                human_id = -2
                state = "noise"
                xmin = int(float(arr[3]))
                ymin = int(float(arr[1]))
                xmax = int(float(arr[4]))
                ymax = int(float(arr[2]))

            else:
                raise ValueError('label file format error: {}'.format(file_path))

            obj = [ymin, ymax, xmin, xmax]
            human_ids.append(human_id)
            states.append(state)
            objs.append(obj)
    return human_ids, states, objs

def calculate_iou(obj1, obj2):
    '''pos: [ymin, ymax, xmin, xmax]
    '''
    ymin = max(obj1[0], obj2[0])
    ymax = min(obj1[1], obj2[1])
    xmin = max(obj1[2], obj2[2])
    xmax = min(obj1[3], obj2[3])
    if ymin >= ymax or xmin >= xmax:
        return 0
    inter = (ymax - ymin) * (xmax - xmin)
    union = (obj1[1] - obj1[0]) * (obj1[3] - obj1[2]) + (obj2[1] - obj2[0]) * (obj2[3] - obj2[2]) - inter
    return inter / union

def generate_moving_BBox(human,states,label):
    merge_label=[]
    for j in range(len(label)):
        label_choose=j
        #label_new_single=[human[label_choose],states[label_choose],label[label_choose][0],label[label_choose][1],label[label_choose][2],label[label_choose][3]]
        #print(label_new_single)
        label_y=(label[label_choose][0]+label[label_choose][1])/2.0
        label_x=(label[label_choose][1]+label[label_choose][2])/2.0
        x=np.cos(np.deg2rad(label_y))*label_x*2.0
        y=np.sin(np.deg2rad(label_y))*label_x*2.0
        label_new_single=[human[label_choose],states[label_choose],label_y,label_x,y,x]
        merge_label.append(label_new_single)
    merge_label.sort(key=lambda x: np.int32(x[0]))
    localize=[]
    #print(human)
    #print(states)
    #print(label)
    #print(merge_label)
    #print(" ")
    return merge_label

def read_trace(dir):
    files=os.listdir(dir)
    remove_list=[]
    trace_dict={}
    for file in files:
        file_split=file.split('_')
        if file[0]==".":
            continue
        if file[:-4] in remove_list:
            continue
        #print(file)
        file_trace_single=dir+"/"+file
        timestamp,pos,seg,state=read_trace_single(file_trace_single)
        key=file[:-4]
        trace_dict.update({key:[timestamp,pos,seg,state]})
    return trace_dict

def merge_trace_dict(dir,save_dir):
    files=os.listdir(dir)
    scenario={}
    for file in files:
        file_split=file.split('_')
        if file_split[0] not in scenario:
            scenario.update({file_split[0]:{0.0:[]}})
    
def moving_center_dis_trace(move_list):
    x_c=0
    y_c=0
    for i in range(len(move_list)):
        x_c+=move_list[i][1][0]
        y_c+=move_list[i][1][1]
    #print(x_c,y_c)
    x_c/=len(move_list)
    y_c/=len(move_list)
    dis=np.sqrt((move_list[-1][1][0]-x_c)**2+(move_list[-1][1][1]-y_c)**2)*100
    #print(dis)
    dis_count=np.sqrt((move_list[-1][1][0]-move_list[-2][1][0])**2+(move_list[-1][1][1]-move_list[-2][1][1])**2)*100 #there is a mistake in the paper, descrption in paper should be fixed as "the distance 𝑑 from the centroid to the swimmer’s locations.", Here our idea is to only compare the current location with the centroid, which can reflect the currecnt location change compared with the centroid and judge whether it is moving. If you calculate the mean distance, the final moving accuracy will fluctuate around 2-5% (may be higher or lower) 
    #print(dis_count)
    return dis,dis_count
    

def motion_detect_trace(loc_dict,pre_config=[],dis_min=[30,30],dis_max=[60,60],IoU_max=[0.5,0.5],ratio=1.0): #unit of dis is cm
    fail_dict=[]
    sucess_dict=[]
    count_correct=0
    state_dict={}
    for key in loc_dict:
        moving_list=[]
        start_time=0.0
        scenario=key.split("_")[0]
        sonar=key.split("_")[1]
        #print(pre_config)
        if pre_config==[]:
            time_single_pre=3.0
            time_threshold=time_single_pre*5+1
            if time_threshold==-1:
                print("error")
            #print(time_threshold)
        else:
            time_threshold=np.float32(pre_config[0])
        #time_threshold=2*2+1
        #print(scenario,time_threshold)
        if key not in state_dict.keys():
            state_dict.update({key:[]})
        for i in range(len(loc_dict[key][0])):
            #print(len(moving_list))
            if moving_list==[]:
                start_time=loc_dict[key][0][i]
                moving_list.append([loc_dict[key][0][i],loc_dict[key][1][i],loc_dict[key][2][i],loc_dict[key][3][i]])#timestamp,pos,seg,state
            elif loc_dict[key][0][i]-start_time<time_threshold: #and len(moving_list)<6
                moving_list.append([loc_dict[key][0][i],loc_dict[key][1][i],loc_dict[key][2][i],loc_dict[key][3][i]])
            else:
                while loc_dict[key][0][i]-start_time>time_threshold: #len(moving_list)>6
                    if moving_list!=[]:
                        moving_list.pop(0)
                    if moving_list!=[]:
                        start_time=moving_list[0][0]
                    else:
                        start_time=loc_dict[key][0][i]
                if moving_list==[]:
                    start_time=loc_dict[key][0][i]
                    moving_list.append([loc_dict[key][0][i],loc_dict[key][1][i],loc_dict[key][2][i],loc_dict[key][3][i]])
                else:
                    moving_list.append([loc_dict[key][0][i],loc_dict[key][1][i],loc_dict[key][2][i],loc_dict[key][3][i]])
            obj_single=[loc_dict[key][0][i],loc_dict[key][1][i],loc_dict[key][2][i],loc_dict[key][3][i]]
            #print(obj_single)
            #if i>1:
                #print(moving_list[-1],moving_list[-2])
            if i == 1:
                dis=np.sqrt((loc_dict[key][1][i][1]-loc_dict[key][1][i-1][1])**2+(loc_dict[key][1][i][0]-loc_dict[key][1][i-1][0])**2)*100
                iou_s=calculate_iou_on_small(loc_dict[key][2][i],moving_list[0][2])
                iou=calculate_iou(loc_dict[key][2][i],moving_list[0][2])
                if dis<dis_min[0]:
                    state_dict[key].append(['non-moving',obj_single])
                else:
                    state_dict[key].append(['moving',obj_single])
                continue     
            if len(moving_list)<=1:
                fail_dict.append(obj_single)
                state_dict[key].append(['non-moving',obj_single])
            else:
                iou_s=calculate_iou_on_small(loc_dict[key][2][i],moving_list[-2][2])
                iou=calculate_iou(loc_dict[key][2][i],moving_list[-2][2])
                dis,dis_count=moving_center_dis_trace(moving_list)
                #print(moving_list)
                #print(obj_single)
                if len(moving_list)>6:
                    print(len(moving_list),"error")
                    break
                else:
                    #continue
                    pass
                if dis==0.0:
                    dis_ratio=0.0
                else:
                    dis_ratio=dis_count/dis
                #print(dis_count)
                if (iou>IoU_max[0] or iou_s>IoU_max[1]) and (dis<dis_min[0] or dis_count<dis_min[1]):#there is a mistake in the paper, description in paper should be fixed as "and 𝐷 or 𝑑 below 𝐷_{𝑚𝑖𝑛} or 𝑑_{𝑚𝑖𝑛}". Low dis and dis_count means slight movement, which does not require meeting the requirement at the same time. If using "or D and d below D_{min} and d_{min}", systems will prefer to predict non-moving since the condition of non-moving is too easy. To correctly infer the moving subjects, please use the code here.
                    fail_dict.append(obj_single)
                    state_dict[key].append(['non-moving',obj_single])
                else:   
                    if dis>dis_max[0] or (dis_count>dis_max[1] and dis_ratio>ratio):
                        count_correct+=1
                        sucess_dict.append(obj_single)
                        state_dict[key].append(['moving',obj_single])
                    else:
                        fail_dict.append(obj_single)
                        state_dict[key].append(['non-moving',obj_single])
    return count_correct,fail_dict,sucess_dict,state_dict
                

def save_results(save_path,obj_new):
    f=open(save_path,"w")
    for i in range(len(obj_new)):
        obj_single=str(obj_new[i][0])+" "+str(obj_new[i][1])+" "+str(obj_new[i][2])+" "+str(obj_new[i][3])+" "+str(obj_new[i][4])+" "+str(obj_new[i][5])+" \n"
        f.writelines(obj_single)
    f.close()
    
def state_smooth(state_dict,len_win,smooth_cfg=[]):
    for key in state_dict.keys():
        scenario=key.split("_")[0]
        sonar=key.split("_")[1]
        if smooth_cfg==[]:
            time_threshold=-1
            time_threshold_past=-1
            time_single=3.0
            time_threshold=time_single*2+np.int32(time_single)-1
            time_threshold_past=time_single*2+np.int32(time_single)-1
            if time_threshold==-1:
                print("error")
        else:
            time_threshold=np.float32(smooth_cfg[0])
            time_threshold_past=np.float32(smooth_cfg[1])
        #print(time_threshold,time_threshold_past)
        for i in range(1,len(state_dict[key])): 
            print(state_dict[key][i],i) 
            print(" ")  
            if len(state_dict[key])==3:
                continue        
            if i == 1 and len(state_dict[key])>=4:
                if np.float32(state_dict[key][3][1][0])-np.float32(state_dict[key][1][1][0])>time_threshold:
                    continue
                if state_dict[key][3][0]==state_dict[key][2][0]:
                    state_dict[key][1][0]=state_dict[key][3][0]
                continue
            if i==2 and len(state_dict[key])>=4:
                if state_dict[key][2][0]!=state_dict[key][1][0]:
                    if np.float32(state_dict[key][3][1][0])-np.float32(state_dict[key][2][1][0])>time_threshold:
                        continue
                    #if np.float32(state_dict[key][2][1][0])-np.float32(state_dict[key][1][1][0])>time_threshold_past and state_dict[key][1][1][0]=="moving": #use past information will not add extra overhead
                    #    continue
                    else:
                        if state_dict[key][1][0]==state_dict[key][3][0]:
                            if len(state_dict[key])>=5:
                                if np.float32(state_dict[key][4][1][0])-np.float32(state_dict[key][2][1][0])>time_threshold:
                                    continue
                                if state_dict[key][3][0]==state_dict[key][4][0]:
                                    state_dict[key][2][0]=state_dict[key][1][0]
                            else:
                                state_dict[key][2][0]=state_dict[key][1][0]
                    #continue here should have a continue, but raw code does not, so the i=2 will be smooth as the other ones in the following loop.
                    #another reason why there is no continue here: the first is non-moving (we cannot recognize moving at the beginning). If 1-4 is non, moving, non, moving, it seems that we should consider it as a non-moving subjects. Without continue, we can smooth this state.
                    #Without continue here will not add more overhead here since we still use the state in the same time windows.
                else:
                    continue
                        
            if i+int(len_win/2.0)<len(state_dict[key]):
   
                if state_dict[key][i][0]==state_dict[key][i-1][0]:
                    continue
                else:
                    same_count=0
                    diff_index=-1
                    diff_count=0
                    for k in range(-1*int(len_win/2.0),int(len_win/2.0)+1):
                        if i+k<0:
                            #print(i+k)
                            continue
                        #print(np.float32(state_dict[key][i][1][0])-np.float32(state_dict[key][i+k][1][0]),)

                        #if k<0 and np.float32(state_dict[key][i][1][0])-np.float32(state_dict[key][i+k][1][0])>time_threshold_past and state_dict[key][i][1][0]=="moving":
                        #    continue
                        if k>0 and np.float32(state_dict[key][i+k][1][0])-np.float32(state_dict[key][i][1][0])>time_threshold:
                            #print(k,np.float32(state_dict[key][i+k][1][0]))
                            if k==1 and state_dict[key][i-2][0]!=state_dict[key][i-3][0]:
                                continue
                                diff_count=0
                                same_count=0
                                break
                            continue
                        #print(k,np.float32(state_dict[key][i+k]))
                        if (np.float32(state_dict[key][i][1][0])-np.float32(state_dict[key][i+k][1][0])<0 and k<0) or (np.float32(state_dict[key][i+k][1][0])-np.float32(state_dict[key][i][1][0])<0 and k>0):
                            break
                        print(k,state_dict[key][i+k])
                        #print(" ")
                        if state_dict[key][i+k][0]==state_dict[key][i][0]:
                            same_count+=1
                        else:
                            diff_count+=1
                            diff_index=i+k
                    if diff_count>same_count:
                        state_dict[key][i][0]=state_dict[key][diff_index][0]                   
            else:
                if state_dict[key][i][0]==state_dict[key][i-1][0]:
                    continue
                else:
                    same_count=0
                    diff_index=-1
                    diff_count=0
                    for k in range(-1*int(len_win/2.0),len(state_dict[key])-i):
                        if i+k<0:
                            #print(i+k,k)
                            continue
                        #if k<0 and np.float32(state_dict[key][i][1][0])-np.float32(state_dict[key][i+k][1][0])>time_threshold_past and state_dict[key][i][1][0]=="moving":
                        #    continue
                        if k>0 and np.float32(state_dict[key][i+k][1][0])-np.float32(state_dict[key][i][1][0])>time_threshold:
                            if k==1 and state_dict[key][i-2][0]!=state_dict[key][i-3][0]:
                                continue
                                diff_count=0
                                same_count=0
                                break
                            continue
                        #print(k)
                        print(k,state_dict[key][i+k])
                        
                        if state_dict[key][i+k][0]==state_dict[key][i][0]:
                            same_count+=1
                        else:
                            diff_count+=1
                            diff_index=i+k
                    if diff_count>same_count:
                        state_dict[key][i][0]=state_dict[key][diff_index][0]
    return state_dict

def statistic_data(state_dict):
    correct_num={}
    correct_pos={}
    count_correct=0
    wrong_num={}
    wrong_pos={}
    count_wrong=0
    for key in state_dict.keys():
        correct_num.update({key:0})
        wrong_num.update({key:0})
        correct_pos.update({key:[]})
        wrong_pos.update({key:[]})
        for i in range(len(state_dict[key])):
            if state_dict[key][i][0]==state_dict[key][i][1][3]:
                count_correct+=1
                correct_num[key]+=1
                correct_pos[key].append(state_dict[key][i][1])
            else:
                count_wrong+=1
                wrong_num[key]+=1
                wrong_pos[key].append(state_dict[key][i][1])
    return correct_num,correct_pos,count_correct,wrong_num,wrong_pos,count_wrong

def read_data_trace(data_dir,print_flag=False,pre_config=[],smooth_cfg=[],dis_min=[30,30],dis_max=[60,60],IoU_max=[0.5,0.5],ratio=1.0):
    data_dict=read_trace(data_dir)
    count_correct,fail_dict,sucess_dict,state_dict=motion_detect_trace(data_dict,pre_config,dis_min=dis_min,dis_max=dis_max,IoU_max=IoU_max,ratio=ratio)


    state_dict_filter=state_smooth(state_dict,5,smooth_cfg)
    #state_dict_filter=state_dict
    correct_num,correct_pos,count_correct_num,wrong_num,wrong_pos,count_wrong_num=statistic_data(state_dict_filter)
    correct=0
    count_mov=0
    correct_mov=0
    count=0
    count_no=0
    correct_no=0
    appear_dict=[]
    for key in state_dict.keys():
        for i in range(len(state_dict[key])):
            if state_dict[key][i][1][0]==0.0 or state_dict[key][i][1][3][0]=="none":
                continue
            else:
                if state_dict[key][i][1][2] not in appear_dict:
                    count+=1
                    if state_dict[key][i][0]==state_dict[key][i][1][3][0]:
                        correct+=1
                        if state_dict[key][i][1][3][0]=="moving":
                            correct_mov+=1
                        else:
                            correct_no+=1
                    
                    if state_dict[key][i][1][3][0]=="moving":
                        count_mov+=1
                    else:
                        count_no+=1
                    appear_dict.append(state_dict[key][i][1][2])
    #print(correct,count,correct*1.0/count)
    #print(correct_mov,count_mov,correct_mov*1.0/count_mov)
    #print(correct_no,count_no,correct_no*1.0/count_no)
    return state_dict_filter
    
def save_moving_detection_result(state_dict, save_dir):
    dir_create(save_dir)
    for key in state_dict:
        file=key+".txt"
        save_file=save_dir+"/"+file
        f=open(save_file,'w')
        for i in range(len(state_dict[key])):
            record=str(state_dict[key][i][0])+" "+str(state_dict[key][i][1][0])+" "+str(state_dict[key][i][1][3][0])+" "+str(state_dict[key][i][1][3][1])+" "+str(state_dict[key][i][1][2][0])+" "+str(state_dict[key][i][1][2][1])+" "+str(state_dict[key][i][1][2][2])+" "+str(state_dict[key][i][1][2][3])+"\n"
            f.writelines(record)
        f.close()
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="data_path")
    parser.add_argument("--save", type=str, required=True, help="save_path")
    parser.add_argument("--save_dir_all", type=str, required=True, help="save_dir")
    parser.add_argument("--pre_cfg", type=float, nargs="+",required=True, help="pre_cfg")
    parser.add_argument("--smooth_cfg", type=float, nargs="+",required=True, help="smooth_cfg")
    args = parser.parse_args()
    save_dir_all=args.save_dir_all
    data_path=save_dir_all+"/"+args.data
    save_path=save_dir_all+"/"+args.save
    if args.pre_cfg==[0.0]:
        pre_cfg=[]
    else:
        pre_cfg=args.pre_cfg
    if args.smooth_cfg==[0.0]:
        smooth_cfg=[]
    else:
        smooth_cfg=args.smooth_cfg
    state_dict=read_data_trace(data_path,False,pre_cfg,smooth_cfg)
    save_moving_detection_result(state_dict,save_path) #not the final results
    print("finish")
    
if __name__=="__main__":
    main()
