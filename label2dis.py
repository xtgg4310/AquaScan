import numpy as np
import argparse
import os
import pre_sonar as ps

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
    #print(path)
    #s=path.split('/')[-1]
    path_seg=path.split('/')[-1]
    if '_' in path_seg:
        file_name=str(path_seg.split('_')[1])
    else:
        file_name=path_seg
    #print(file_name)
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

def read_txt_noangle(path,angle_range=400):
    path_seg=path.split('/')[-1]
    if '_' in path_seg:
        file_name=str(path_seg.split('_')[1])
    else:
        file_name=path_seg
    #print(file_name)
    file_order=file_name.split('.')[0]
    sonar_data = np.zeros((angle_range, 500))
    with open(path, 'r') as f:
        lines = f.readlines()
        start_angle = 0
        end_angle = 399
        angle=0
        for line in lines:
            data=line.split()
            data = list(map(float, data))
            #print(angle,data)
            if len(data) == 500:
                if np.int32(file_order)%2==1:
                    sonar_data[(int(angle))%400] = data
                else:
                    sonar_data[int(angle)] = data
            angle+=1
    return sonar_data, int(start_angle), int(end_angle)

def img_recover(sonar_image,start=100,end=298):
    data_new=np.zeros((400,500))
    for i in range(len(sonar_image)):
        data_new[start+i*3,:]=sonar_image[i,:]
    return data_new

def read_skip_data(path,angle_range=400,bias=0):
    path_seg=path.split('/')[-1]
    if '_' in path_seg:
        file_name=str(path_seg.split('_')[1])
    else:
        file_name=path_seg
    #print(file_name)
    file_order=file_name.split('.')[0]
    sonar_data = []
    with open(path, 'r') as f:
        lines = f.readlines()
        start_angle = float(lines[0].split(' ')[0])
        end_angle = float(lines[-1].split(' ')[0])
        for line in lines:
            angle, data = readline(line)
            sonar_data.append(data)
    if start_angle>end_angle:
        sonar_data.reverse()
    sonar_data=np.array(sonar_data)
    print(start_angle,end_angle)
    return sonar_data, int(start_angle), int(end_angle)

def read_rescale_results(file_path,ratio=1.0):
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
        #print(lines)
        for line in lines:
            arr = line.strip().split(',')
            xmin = int(float(arr[0])/ratio) #4
            ymin = int(float(arr[1])/ratio) #2
            xmax = int(float(arr[2])/ratio) #5
            ymax = int(float(arr[3])/ratio) #3
            raw_ymin=int(float(arr[1]))
            raw_xmin=int(float(arr[0]))
            raw_ymax=int(float(arr[3]))
            raw_xmax=int(float(arr[2]))
            raw_bbox=[raw_ymin,raw_ymax,raw_xmin,raw_xmax]
            obj = [ymin, ymax, xmin, xmax,raw_bbox]
            objs.append(obj)
    return human_ids,states,objs

def read_rescale_results_gt(file_path):
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
            human=arr[0]
            state=arr[1]
            xmin = int(float(arr[4])) #4
            ymin = int(float(arr[2])) #2
            xmax = int(float(arr[5])) #5
            ymax = int(float(arr[3])) #3
            
            obj = [ymin, ymax, xmin, xmax]
            objs.append(obj)
            human_ids.append(human)
            states.append(state)
    return human_ids,states,objs

def read_pos_results(file_path):
    objs=[]
    with open(file_path,'r') as f:
        lines=f.readlines()
        if '\n' in lines:
            lines.remove('\n')
        for line in lines:
            arr=line.strip().split()
            x_aver = int(float(arr[0])) #0
            y_aver = int(float(arr[1])) #1
            x_mid = int(float(arr[2])) #2
            y_mid = int(float(arr[3])) #3
            obj=[x_aver,y_aver,x_mid,y_mid]
            objs.append(obj)
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
    if not only_label:
        IoU=(cross)/(sum_area-cross)
    else:
        if obj2_area!=0:
            IoU=cross/obj2_area
        else:
            IoU=0
    return IoU

def get_min_val(data):
    min_value=200
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j]!=0:
                if data[i][j]<min_value:
                    min_value=data[i][j]
    return min_value

def generate_weight_matrix(sonar):
    weight=np.zeros_like(sonar)
    max_weight=np.max(sonar)
    min_weight=get_min_val(sonar)
    min_gap=(max_weight-min_weight)*0.50
    min_weight=min_weight+min_gap
    print(min_weight,max_weight)
    for i in range(len(weight)):
        for j in range(len(weight[i])):
            if sonar[i][j]!=0 and sonar[i][j]>min_weight:
                weight[i][j]=(sonar[i][j]-min_weight)/(max_weight-min_weight)
    return weight

def generate_loc(sonar,obj,dis=17.0):
    x_sum=0
    x_ener=0
    y_ener=0
    y_sum=0
    print(obj)
    x_mid=(obj[2]+obj[3])/2.0
    y_mid=(obj[1]+obj[0])/2.0
    x_real_mid=x_mid*np.cos(y_mid*0.9*np.pi/180.0)*dis/499 #499 gaps, 500->499, has little difference, 500 tested without any influence. #angle-y axis dis-x axis, read data [min_angle, max_angle, min_dis, max_dis], angle unit is gradian, 0.9*gradian=1 degree
    y_real_mid=x_mid*np.sin(y_mid*0.9*np.pi/180.0)*dis/499
    print(obj[0],obj[1],obj[2],obj[3])
    print(sonar.shape)
    if obj[0]==obj[1]:
        obj[0]-=1
        obj[1]+=1
    sonar_seg=sonar[obj[0]:obj[1],obj[2]:obj[3]]
    sonar_seg=generate_weight_matrix(sonar_seg)
    for i in range(obj[0],obj[1]):
        for j in range(obj[2],obj[3]):
            y_sum+=sonar_seg[i-obj[0]][j-obj[2]]*i
            y_ener+=sonar_seg[i-obj[0]][j-obj[2]]
            x_sum+=sonar_seg[i-obj[0]][j-obj[2]]*j
            x_ener+=sonar_seg[i-obj[0]][j-obj[2]]
    x_aver=x_sum/x_ener
    y_aver=y_sum/y_ener
    x_real_aver=x_aver*np.cos(y_aver*0.9*np.pi/180.0)*dis/499
    y_real_aver=x_aver*np.sin(y_aver*0.9*np.pi/180.0)*dis/499
    return x_real_aver,y_real_aver,x_real_mid,y_real_mid,[x_aver,y_aver,x_mid,y_mid],[obj[0],obj[1],obj[2],obj[3]],obj[4] #here, x y not represent dis-angle , but a coordinate.
    

def label2pos_seg(sonar_data,detect_obj,objs_gt,states,humans,dis):
    pos_list=[]
    for i in range(len(detect_obj)):
        iou_max=0.0
        iou_max_index=-1
        for j in range(len(objs_gt)):
            iou_temp=cal_IoU(objs_gt[j],detect_obj[i][4])
            if iou_temp>iou_max:
                iou_max_index=j
                iou_max=iou_temp
        x,y,x_mid,y_mid,polar_pos,obj,obj_scale=generate_loc(sonar_data,detect_obj[i],dis)
        if iou_max_index!=-1:
            pos_list.append([x,y,x_mid,y_mid,polar_pos,obj,obj_scale,states[iou_max_index],states[iou_max_index],humans[iou_max_index]]) #states[iou_max_index],states[iou_max_index],humans[iou_max_index]
        else:
            pos_list.append([x,y,x_mid,y_mid,polar_pos,obj,obj_scale,"none","-1"])
    return pos_list

def write_pos(pos,save_file):
    f=open(save_file,'w')
    for i in range(len(pos)):
        record=str(pos[i][0])+" "+str(pos[i][1])+" "+str(pos[i][2])+" "+str(pos[i][3])+" "+str(pos[i][4][0])+" "+str(pos[i][4][1])+" "+str(pos[i][4][2])+" "+str(pos[i][4][3])+" "+str(pos[i][5][0])+" "+str(pos[i][5][1])+" "+str(pos[i][5][2])+" "+str(pos[i][5][3])+" "+str(pos[i][6][0])+" "+str(pos[i][6][1])+" "+str(pos[i][6][2])+" "+str(pos[i][6][3])+" "+str(pos[i][7])+"\n"
        f.writelines(record)
    f.close()
    
def write_label(pos,save_file):
    f=open(save_file,"w")
    for i in range(len(pos)):
        record=str(pos[i][8])+" "+str(pos[i][7])+" "+str(pos[i][5][0])+" "+str(pos[i][5][1])+" "+str(pos[i][5][2])+" "+str(pos[i][5][3])+" "+"\n"
        f.writelines(record)
    f.close()

def label2pos(label_direct,data_direct,gt_direct,save_direct,save_eval_label_dir,read_type,data_config,dis,remove):
    scenario=os.listdir(label_direct)
    dir_create(save_direct)
    dir_create(save_eval_label_dir)
    if ".DS_Store" in scenario:
        scenario.remove(".DS_Store")
    for i in range(len(scenario)):    
        label_raw_path=label_direct+"/"+scenario[i]
        data_path_scenario=data_direct+"/"+scenario[i]
        gt_path_scenario=gt_direct+"/"+scenario[i]
        save_scenario=save_direct+"/"+scenario[i]
        save_eval=save_eval_label_dir+"/"+scenario[i]
        dir_create(save_scenario)
        dir_create(save_eval)
        sonars=os.listdir(gt_path_scenario)
        if ".DS_Store" in sonars:
            sonars.remove(".DS_Store")
        for sonar in sonars:
            label_path_single=label_raw_path+"/"+sonar+"/txt"
            data_path_sonar=data_path_scenario+"/"+sonar
            gt_path_sonar=gt_path_scenario+"/"+sonar
            save_sonar=save_scenario+"/"+sonar
            save_eval_sonar=save_eval+"/"+sonar
            dir_create(save_sonar)
            dir_create(save_eval_sonar)
            files = os.listdir(gt_path_sonar)
            if ".DS_Store" in files:
                files.remove(".DS_Store")
            files.sort(key=lambda x:np.int32(x.split('_')[1][:-4]))
            for file in files:
                label_path_one=label_path_single+"/"+file
                data_path_one=data_path_sonar+"/"+file
                if read_type!=0 and read_type!=1:
                    data_path_one=data_path_sonar+"/"+file[:-4]+".npy"
                save_file=save_sonar+"/"+file
                gt_file=gt_path_sonar+"/"+file
                save_eval_file=save_eval_sonar+"/"+file
                if read_type==0:
                    sonar_data,_,_=read_txt(data_path_one)
                elif read_type==1:
                    sonar_data,_,_=read_txt(data_path_one)
                else:
                    sonar_data=np.load(data_path_one)
                if read_type!=0 and read_type!=1:
                    pre=True
                else:
                    pre=False
                sonar_data=ps.data_pre(sonar_data,data_config[0],data_config[1],remove,pre)
                _,_,detect_obj=read_rescale_results(label_path_one,ratio=3.0)
                humans,states,objs_gt=read_rescale_results_gt(gt_file)
                print(data_path_one)
                pos_list=label2pos_seg(sonar_data,detect_obj,objs_gt,states,humans,dis)
                write_pos(pos_list,save_file)
                write_label(pos_list,save_eval_file)
                
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="data_path")
    parser.add_argument("--detect",type=str, required=True, help="label_path")
    parser.add_argument("--gt",type=str, required=True, help="gt_path")
    parser.add_argument("--type",type=int, required=True, help="data_type")
    parser.add_argument("--remove",type=int, required=True, help="data_type")
    parser.add_argument("--dis",type=int, required=True, help="dis")
    parser.add_argument("--parad",type=int, nargs="+",required=True, help="parad")
    parser.add_argument("--save_dir_all",type=str, required=True, help="label_path")
    args = parser.parse_args()
    save_pos_dir=args.save_dir_all
    data_dir=args.data
    label_dir=save_pos_dir+"/"+args.detect
    gt_dir=save_pos_dir+"/"+args.gt

    save_infer_label=save_pos_dir+"/reference_label_not_infe"
    save_dir=save_pos_dir+"/"+args.data+"_pos"
    parad=args.parad
    dis=args.dis
    remove=args.remove
    len_single=(len(parad))/2
    len_single=np.int32(len_single)
    
    if len_single!=(len(parad))/2.0:
        print("error parad")
        return
    dis_para=[]
    threshold_para=[]
    for i in range(len_single):
        threshold_para.append(parad[i])
    for i in range(len_single,(len(parad))):
        dis_para.append(parad[i])
    data_type=args.type
    parad_input=[threshold_para,dis_para]
    print(parad_input)
    label2pos(label_dir,data_dir,gt_dir,save_dir,save_infer_label,data_type,data_config=parad_input,dis=dis,remove=remove)
    
if __name__=="__main__":
    main()
