import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageDraw
from sklearn.cluster import DBSCAN
#from skimage import transform
import cmath
import math
import cv2
import os
import argparse
import copy
import time
import denoise_metric as dm


def read_final_results(scenario_prefix,data_dir):
    files=os.listdir(data_dir)
    re_dict={}
    for file in files:
        if file[0:8]!=scenario_prefix:
            continue
        f=open(data_dir+"/"+file,'r')
        lines=f.readlines()
        for line in lines:
            data=line.split(' ')
            time=np.float32(data[0])
            if time not in re_dict:
                re_dict[time]=[]
            else:
                re_dict[time].append([data[1],np.float32(data[3]),np.float32(data[4]),np.float32(data[5]),np.float32(data[6])])
    print(re_dict)
    return re_dict

                

def obj_recover(obj,start,end):
    if start>end:
        temp=start
        start=end
        end=temp
    obj_new=[]
    print(start,end)
    for obj_s in obj:
        print(obj_s)
        obj_x_min=obj_s[2]
        obj_y_min=(obj_s[0]+start)*3
        obj_x_max=obj_s[3]
        obj_y_max=(obj_s[1]+start)*3
        obj_new_single=[obj_y_min,obj_y_max,obj_x_min,obj_x_max]
        obj_new.append(obj_new_single)
        print(obj_new_single)
   # print(obj_new)
    return obj_new
    
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

def read_yolo_label(file_path, W=500, H=400,bias=9):
    ''' Read one txt file in yolo format.
        human_id | state_str | x_center | y_center | width | height
    Args:
        file_path: path of label file
        W: width of sonar data, default 500 unit
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

def calculate_iou_on_obj1(obj1, obj2):
    ''' Calculate the iou of two objects.
        obj1 should be the denoised object.
    '''
    ymin = max(obj1[0], obj2[0])
    ymax = min(obj1[1], obj2[1])
    xmin = max(obj1[2], obj2[2])
    xmax = min(obj1[3], obj2[3])
    if ymin >= ymax or xmin >= xmax:
        return 0
    inter = (ymax - ymin) * (xmax - xmin)
    obj1_area = (obj1[1] - obj1[0]) * (obj1[3] - obj1[2])
    iou = inter / obj1_area
    return iou

def fuse_objects(denoised_objs, raw_objs, iou_threshold=0.2):
    ''' Denoised objects are more accurate, but smaller than raw objects.
        So use denoised objects to filter raw objects.
    '''
    if len(denoised_objs) == 0:
        return []
    if len(raw_objs) == 0:
        return denoised_objs
    fused_objs = []
    for denoised_obj in denoised_objs:
        for raw_obj in raw_objs:
            if calculate_iou_on_obj1(denoised_obj, raw_obj) > iou_threshold:
                fused_objs.append(raw_obj)
    if len(fused_objs) == 0:
        fused_objs = denoised_objs
    return fused_objs

def fuse_denoised_objects(denoised_objs, raw_objs, iou_threshold=0.3): # save all possible objs. We also remove the repeated subjects, which avoid double detection.
    if len(denoised_objs) == 0:
        return raw_objs
    fused_objs = []
    for denoised_obj in denoised_objs: 
        for raw_obj in raw_objs:
            if calculate_iou_on_obj1(raw_obj,denoised_obj) > iou_threshold:
                fused_objs.append(raw_obj)
            elif calculate_iou_on_obj1(denoised_obj,raw_obj) > iou_threshold:
                fused_objs.append(denoised_obj)
    if len(fused_objs)==0:
        fused_objs = raw_objs
    return fused_objs

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
        print(lines)
        for line in lines:
            arr = line.strip().split()
            # human object
            if len(arr) == 6:
                human_id = arr[0]
                state = arr[1]
                xmin = int(float(arr[2])/1.0)
                ymin = int(float(arr[3])/1.0)
                xmax = int(float(arr[4])/1.0)
                ymax = int(float(arr[5])/1.0)
            
            # noise object
            elif len(arr) == 5:
                human_id = -2
                state = "noise"
                xmin = int(float(arr[1])/1.0)
                ymin = int(float(arr[2])/1.0)
                xmax = int(float(arr[3])/1.0)
                ymax = int(float(arr[4])/1.0)

            else:
                raise ValueError('label file format error: {}'.format(file_path))

            obj = [ymin, ymax, xmin, xmax]
            human_ids.append(human_id)
            states.append(state)
            objs.append(obj)
    return human_ids, states, objs

def read_default_label_noangle(file_path):
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
        print(lines)
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
                xmin = int(float(arr[1])/2.0)
                ymin = int(float(arr[2])/2.0)
                xmax = int(float(arr[3])/2.0)
                ymax = int(float(arr[4])/2.0)

            else:
                raise ValueError('label file format error: {}'.format(file_path))

            obj = [ymin, ymax, xmin, xmax]
            human_ids.append(human_id)
            states.append(state)
            objs.append(obj)
    return human_ids, states, objs

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
        print(lines)
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


def data_remove(data,threshold,distance):
    #define your pool setting
    pass

def remove_line(data,threshold,distance):
    #define your pool line
    pass

def data_rescale(data,threshold,distance=[500],re_size=[1200,1500],remove_type=0,pre=False):
    if len(threshold)!=len(distance):
        print("unalign-bg-noise-remove")
        return
    #use remove_type control the configuration
    if remove_type==0:
        data_noline=remove_line(data,threshold,distance,pre)

    if remove_type==0:
        data_new=data_remove(data_noline,threshold,distance,pre)

    data_new=cv2.resize(data_new,(re_size[1],re_size[0]))
    print(data_new.shape)
    data_new=np.uint8(data_new)
    ratio=re_size[1]/500.0
    distance_new=np.zeros_like(distance)
    for i in range(len(distance)):
        distance_new[i]=ratio*distance[i]
    threshold_new=np.zeros_like(threshold)
    for i in range(len(threshold)):
        threshold_new[i]=threshold[i]/3.0
    if remove_type==0:
        data_new=data_remove(data_new,threshold_new,distance_new,pre)
    return data_new

def localize_one_pic(temp_data, save_data,save_flag=False,threshold=45, min_samples=10, eps=12, blur_size=5, min_size=25):
    """
    Transform sonar data to point location, which can be used in cluster algorithm
    :param temp_data: One sonar picture.
    :param threshold: The lowest strength of point for filter.
    :return:
    """
    temp_data_new = temp_data.astype(np.uint8)
    if blur_size > 0:
        temp_data_new = cv2.medianBlur(temp_data_new, blur_size)
    
    dbscan = DBSCAN(min_samples=min_samples, eps=eps)
    object_poses = []
    #print(temp_data_new.shape)
    point_pos = sonar2pos(temp_data_new, threshold)
    #print(point_pos.shape)
    try:
        labels = dbscan.fit_predict(point_pos)
    except:
        return object_poses
    if save_flag:
        visualize_result_step(point_pos,labels,eps,min_samples,save_data)
    label_class = np.unique(labels)
    for label in label_class:
        if label < 0:
            continue
        obj_pos = np.where(labels == label)[0]
        temp_dis = []
        temp_angle = []

        for point in obj_pos:
            temp_dis.append(point_pos[point, 1])
            temp_angle.append(point_pos[point, 0])
        min_dis = min(temp_dis)
        max_dis = max(temp_dis)
        min_angle = min(temp_angle)
        max_angle = max(temp_angle)
        if (max_angle - min_angle) * (max_dis - min_dis) < min_size:
            continue
        object_poses.append([min_angle, max_angle, min_dis, max_dis])
    return object_poses
    
def save_edge_detection(sonar_data,save_path):
    edges = cv2.Canny(sonar_data,10,255)
    cv2.imwrite(save_path,edges)
    
def process_one_pic(sonar_data, save_path,eps=[15,15,15],min_sample=[25,25,25],blur_size=[9,9,9,19],human_size=4680,max_blur=25,ratio=3.0,latency=False): #ratio=1.0
    ''' 
    Use different parameters to localize objects in one pic
    :param sonar_data: One sonar data.
    :return: object_poses: A list of objects' poses. Each object is a list of [min_angle, max_angle, min_dis, max_dis]
    '''
    # Get denoised bbox
    time1=time.time()
    #print(min_sample)
    denoise_configs = {
        'denoise1' : {
            # general
            'threshold': 5,
            'min_samples': min_sample[0],
            'eps': eps[0],#15
            'blur_size':blur_size[0], #3 9
            'min_size': 15,
        },
        'denoise2' : {
            # for weak objects
            'threshold': 5,
            'min_samples': min_sample[1],
            'eps': eps[1],#17
            'blur_size': blur_size[1],#9
            'min_size': 15,#20
        },
        'denoise3' : {
            # for strong noise
            'threshold': 5,
            'min_samples': min_sample[2],
            'eps': eps[2],#10
            'blur_size': max_blur,#17
            'min_size': 15,
        },
    }
    save_data_single=""
    index=0
    denoised_object_poses = []
    raw_object_poses = []
    sonar_data = copy.deepcopy(sonar_data)    
    result_poses = []
    denoised_object_poses = localize_one_pic(sonar_data,save_data_single, **denoise_configs['denoise1'])
    #if len(denoised_object_poses) == 0:
    #    save_data_single=save_path+"_denoise_"+str(index)
    #    index+=1
    #    denoised_object_poses = localize_one_pic(sonar_data,save_data_single, **denoise_configs['denoise2'])
    '''
    K_size=blur_size[2] 
    while obj_physical_filter(denoised_object_poses,human_size*ratio*ratio,40*ratio,100*ratio):
        K_size+=2 
        denoise_configs['denoise2']['blur_size']=K_size
        denoised_object_poses = localize_one_pic(sonar_data,save_data_single, **denoise_configs['denoise2'])
        if K_size>=blur_size[3]: 
            break
            
    '''
    #above is another implementaion , it is also works
    
    #print(sonar_data.shape,"111")
    K_size=blur_size[2]
    while K_size<=blur_size[3]:
        denoise_configs['denoise2']['blur_size']=K_size
        denoised_object_poses = localize_one_pic(sonar_data,save_data_single, **denoise_configs['denoise2'])
        
        if obj_physical_filter(denoised_object_poses,human_size*ratio*ratio,40*ratio,100*ratio):
            K_size+=2
            #denoised_object_poses=denoised_object_poses_temp
        else:
            break
    #if objec

    print(K_size)
    objs_well_denoise= localize_one_pic(sonar_data,save_data_single,**denoise_configs['denoise3'])
    

    object_poses=fuse_denoised_objects(objs_well_denoise,denoised_object_poses) 
    
    #object_poses=denoised_object_poses
    print(object_poses)
    obj_idx=0
    while obj_idx<len(object_poses):

        if obj_tiny_filter(object_poses[obj_idx],human_size=120*ratio*ratio,angle_thre=10*ratio,length_threshold=10*ratio):
            object_poses.remove(object_poses[obj_idx])
            obj_idx-=1
        obj_idx+=1
    print(object_poses)
    #object_poses=denoised_object_poses
    filter_flag=False
    
    for obj in object_poses:
        result_poses.append(obj)
    if len(result_poses) == 0:
        result_poses = denoised_object_poses
        filter_flag=True
    if len(result_poses) == 0:
        result_poses = raw_object_poses
        filter_flag=True
    new_result=[]
    for obj in result_poses:
        if obj not in new_result:
            new_result.append(obj)
    latency=time.time()-time1
    print(latency)
    if not latency:
        return new_result
    else:
        return new_result,latency
    
def obj_physical_filter(obj_list, human_size, angle_thre, length_threshold):
    for obj in obj_list:
        if np.abs((obj[1]-obj[0])*(obj[3]-obj[2]))>human_size or np.abs(obj[1]-obj[0])>angle_thre or np.abs(obj[3]-obj[2])>length_threshold:
            #print((obj[1]-obj[0])*(obj[3]-obj[2]))
            return True
    return False

def obj_tiny_filter(obj,human_size=120*3,angle_thre=5*3,length_threshold=10*3):
    if 300<obj[2]<400:
        ratio_angle=1.0
    elif obj[2]>=400:
        ratio_angle=1.0
    else:
        ratio_angle=1.0
    if np.abs((obj[1]-obj[0])*(obj[3]-obj[2]))<=human_size or np.abs(obj[1]-obj[0])<=angle_thre*ratio_angle or np.abs(obj[3]-obj[2])<=length_threshold:
        return True
    return False

def visualize(swimmer, sonar_pics, result_dict, im, swimmer_idx):
    """
    Return and visualize the results.
    Only used in test
    """
    re = swimmer
    if im is None:
        sonar_data = sonar_pics
        im = Image.fromarray(sonar_data)
        im = im.convert('RGB')

    a = ImageDraw.ImageDraw(im)
    a.rectangle(((re[2], re[0]), (re[3], re[1])), fill=None, outline='blue', width=8)

    text = []
    return im, re[2], re[0], re[3], re[1]

def visualize_none_swimmer(data,im):
    if im is None:
        im = Image.fromarray(data)
        im = im.convert('RGB')
    return im

def vis_figure(data,obj,save_path,save_path_file):
    im=None
    with open(save_path_file, 'w') as f:
        if len(obj)==0:
            im_d=visualize_none_swimmer(data,im)
            im_d.save(save_path)
            f.close()
        else:
            for i, swimmer in enumerate(obj):
                result_dict=[]
                im, xmin, ymin, xmax, ymax= visualize(swimmer, data, result_dict, im, i)
                f.write(str(xmin)+ ','+ str(ymin) + ','+ str(xmax) + ',' + str(ymax) + ',\n')
            im.save(save_path)
            f.close()
            
def visualize_final(swimmer, sonar_pics, result_dict, im, swimmer_idx):
    re = swimmer
    if im is None:
        sonar_data = sonar_pics
        im = Image.fromarray(sonar_data)
        im = im.convert('RGB')
    color="blue"
    if re[0]:
        
        a = ImageDraw.ImageDraw(im)
        a.rectangle(((re[3], re[1]), (re[4], re[2])), fill=None, outline='blue', width=8)

    text = []
    return im, re[2], re[0], re[3], re[1]
            
def vis_figure_final(data,obj,save_path,save_path_file):
    im=None
    with open(save_path_file, 'w') as f:
        if len(obj)==0:
            im_d=visualize_none_swimmer(data,im)
            im_d.save(save_path)
            f.close()
        else:
            for i, swimmer in enumerate(obj):
                result_dict=[]
                im, xmin, ymin, xmax, ymax= visualize(swimmer, data, result_dict, im, i)
                f.write(str(xmin)+ ','+ str(ymin) + ','+ str(xmax) + ',' + str(ymax) + ',\n')
            im.save(save_path)
            f.close()

def vis_only_figure(data,obj,save_path):
    im=None
    if len(obj)==0:
        im_d=visualize_none_swimmer(data,im)
        im_d.save(save_path)
    else:
        for i, swimmer in enumerate(obj):
            result_dict=[]
            im, xmin, ymin, xmax, ymax= visualize(swimmer, data, result_dict, im, i)
        im.save(save_path)

def sonar2pos(sonar_data,threshold=10):
    pos_arr = np.where(sonar_data >= threshold)
    num = pos_arr[0].shape[0]
    pos_matrix = []
    for i in range(num):
        pos_matrix.append([pos_arr[0][i], pos_arr[1][i]])
    pos_matrix = np.array(pos_matrix)
    return pos_matrix

def visualize_result_step(data,labels,eps,min_samples,save):    
    dbscan = DBSCAN(min_samples=min_samples, eps=eps)
    db=dbscan.fit(data)
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    ax=plt.gca()
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = labels == k

        xy = data[class_member_mask & core_samples_mask]
        ax.plot(
            xy[:, 1],
            xy[:, 0],
            markerfacecolor=tuple(col),
            markeredgecolor="k",
        )

        xy = data[class_member_mask & ~core_samples_mask]
        ax.plot(
            xy[:, 1],
            xy[:, 0],
            markerfacecolor=tuple(col),
            markeredgecolor="k",
        )
    ax.invert_yaxis()
    plt.title(f"Estimated number of clusters: {n_clusters_}")
    plt.savefig(save+"_"+str(eps)+"_"+str(min_samples)+".png")
    plt.close()    

def visualize_dbscan(data,save_path,eps=15,min_samples=20):
    dbscan = DBSCAN(min_samples=min_samples, eps=eps)
    db=dbscan.fit(data)
    labels=dbscan.fit_predict(data)
    #print(labels)
    #labels=db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    ax=plt.gca()
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = labels == k

        xy = data[class_member_mask & core_samples_mask]
        ax.plot(
            xy[:, 1],
            xy[:, 0],
            markerfacecolor=tuple(col),
            markeredgecolor="k",
        )

        xy = data[class_member_mask & ~core_samples_mask]
        ax.plot(
            xy[:, 1],
            xy[:, 0],
            markerfacecolor=tuple(col),
            markeredgecolor="k",
        )
    ax.invert_yaxis()
    plt.title(f"Estimated number of clusters: {n_clusters_}")
    plt.savefig(save_path)
    plt.close()    
    object_poses = []
    label_class = np.unique(labels)
    for label in label_class:
        if label < 0:
            continue
        obj_pos = np.where(labels == label)[0]
        temp_dis = []
        temp_angle = []

        for point in obj_pos:
            temp_dis.append(data[point, 1])
            temp_angle.append(data[point, 0])
        min_dis = min(temp_dis)
        max_dis = max(temp_dis)
        min_angle = min(temp_angle)
        max_angle = max(temp_angle)
        object_poses.append([min_angle, max_angle, min_dis, max_dis])
        #print(label,[min_angle, max_angle, min_dis, max_dis])

def results_resize(H,W,label_single):
    H_ratio=H/400.0
    W_ratio=W/500.0
    #print(H_ratio,W_ratio,label_single)
    label_new=[0,0,0,0]
    label_new[0]=label_single[0]*H_ratio
    label_new[1]=label_single[1]*H_ratio
    label_new[2]=label_single[2]*W_ratio
    label_new[3]=label_single[3]*W_ratio      
    return label_new

def label_trans_scale(H,W,obj):
    for i in range(len(obj)):
        obj[i]=results_resize(H,W,obj[i])
    return obj

def label_polar_single(label,length):
    y_mean=0
    #print(label)
    left_up = cmath.rect(label[2], math.radians((label[0] - y_mean) * 0.9))
    left_down = cmath.rect(label[2], math.radians((label[1] - y_mean) * 0.9))
    right_up = cmath.rect(label[3], math.radians((label[0] - y_mean) * 0.9))
    right_down = cmath.rect(label[3], math.radians((label[1] - y_mean) * 0.9))
    
    x = [left_up.real+length, left_down.real+length, right_up.real+length, right_down.real+length]
    xmin = min(x)
    xmax = max(x)
    xmin = math.floor(xmin)
    xmax = math.ceil(xmax)
    xmax = xmax + 5
    xmin = xmin - 5
    
    y = [left_up.imag+length, left_down.imag+length, right_up.imag+length, right_down.imag+length]
    ymin = min(y)
    ymax = max(y)
    ymin = math.floor(ymin)
    ymax = math.ceil(ymax)
    ymax = ymax + 5
    ymin = ymin - 5
    
    return [ymin,ymax,xmin,xmax]
    

def label_polar(labels,length):
    #labels_new=[]
    for i in range(len(labels)):
        labels[i]=label_polar_single(labels[i],length)
    return labels

def label_save(h,s,obj,save_path):
    f=open(save_path,"w")
    for i in range(len(obj)):
        record_single=str(h[i])+" "+str(s[i])+" "+str(obj[i][0])+" "+str(obj[i][1])+" "+str(obj[i][2])+" "+str(obj[i][3])+"\n"
        f.writelines(record_single)
    f.close()

def save_data(data, name):
    np.save(name+".npy",data)
    
def save_data_img(data, name):
    cv2.imwrite(name+".png",data)
        
def image_shift(data, bias):
    new_data=np.zeros_like(data)
    print(len(data))
    for i in range(len(data)):
        new_data[(i+bias)%400,:]=data[i,:]
    return new_data

def label_shift(obj_single,bias): #for those images caused by hardware errors
    print(obj_single)
    for i in range(len(obj_single)):
        obj_single[i][0]+=bias
        obj_single[i][1]+=bias
    return obj_single
    
def save_process_data(data,data_path,image_path,image_save=False):
    np.save(data_path,data)
    if image_save:
        cv2.imwrite(image_path,data)
        
def read_data_process(dirc,dirc_label,save_process_npy,save_img,save_label,label_scenario): #for those images caused by hardware errors
    scenario=os.listdir(dirc_label)
    dir_create(save_process_npy)
    dir_create(save_img)
    dir_create(save_label)
    if ".DS_Store" in scenario:
        scenario.remove(".DS_Store")
    for i in range(len(scenario)): 
        if scenario[i] not in label_scenario: 
            continue   
        data_path=dirc+"/"+scenario[i]
        label_path=dirc_label+"/"+scenario[i]
        save_path_data=save_process_npy+"/"+scenario[i]
        save_img_data=save_img+"/"+scenario[i]
        save_label_data=save_label+"/"+scenario[i]
        sonars=os.listdir(data_path)
        dir_create(save_path_data)
        dir_create(save_img_data)
        dir_create(save_label_data)
        if ".DS_Store" in sonars:
            sonars.remove(".DS_Store")
        for sonar in sonars:
            data_path_single=data_path+"/"+sonar
            label_path_single=label_path+"/"+sonar
            save_path_single=save_path_data+"/"+sonar
            save_img_single=save_img_data+"/"+sonar
            save_label_single=save_label_data+"/"+sonar
            dir_create(save_path_single)
            dir_create(save_label_single)
            dir_create(save_img_single)
            files = os.listdir(data_path_single)
            if ".DS_Store" in files:
                files.remove(".DS_Store")
            for file in files:
                npy_save_path=save_path_single+"/"+file[:-4]+".npy"
                img_save_path=save_img_single+"/"+file[:-4]+".png"
                img_save_path_raw=save_img_single+"/"+file[:-4]+"_raw.png"
                label_save_path=save_label_single+"/"+file
                data_path_single_file=data_path_single+'/'+file
                label_path_single_file=label_path_single+"/"+file
                sonar_data,_,_=read_txt(data_path_single_file)
                #save_data_img(sonar_data, img_save_path_raw)
                h,s,obj=read_default_label(label_path_single_file)
                index=np.int32(file[:-4].split("_")[1])
                print(index,obj)
                if (index+1)%2==0:
                    sonar_data=image_shift(sonar_data,12)
                    obj=label_shift(obj,12)
                save_process_data(sonar_data,npy_save_path,img_save_path,True)
                label_save(h,s,obj,label_save_path)
                
def read_data_path(dirc,save_path,preprocess_flag=0,para_scale=[],para_polar=[500],process=False,blur_size=[9,9,9,19],human_size=4680,remove_type=0,bg="./no_img/no/sonar2",background=["no"],max_blur=27):
    scenario=os.listdir(dirc)
    count=0
    latency_denoise=0
    if ".DS_Store" in scenario:
        scenario.remove(".DS_Store")
    dir_create(save_path)
    for i in range(len(scenario)):   
        if scenario[i] in background:
            continue
        print(scenario[i])
        data_path=dirc+"/"+scenario[i]
        save_path_data=save_path+"/"+scenario[i]
        sonars=os.listdir(data_path)
        dir_create(save_path_data)
        if ".DS_Store" in sonars:
            sonars.remove(".DS_Store")
        for sonar in sonars:
            data_path_single=data_path+"/"+sonar
            save_path_single=save_path_data+"/"+sonar
            dir_create(save_path_single)
            files = os.listdir(data_path_single)
            if ".DS_Store" in files:
                files.remove(".DS_Store")
            files.sort(key=lambda x:np.int32(x.split('_')[1][:-4]))
            for file in files:
                print(file)
                data_detection_result=save_path_single+"/txt/"
                dir_create(data_detection_result)
                data_detection_single=data_detection_result+file[:-4]+".txt"
                data_path_single_file=data_path_single+"/"+file
                save_name_img=save_path_single+"/img/"+file[:-4]
                save_name_img_vis=save_path_single+"/img/"+file[:-4]+"_vis"
                dir_create(save_path_single+"/img/")
                save_dbscan_path=save_path_single+"/vis/"
                dir_create(save_dbscan_path)
                save_edge=save_path_single+"/edge/"
                save_edge_img=save_edge+file[:-4]+".png"
                dir_create(save_edge)
                save_dbscan_file=save_dbscan_path+file[:-4]+".png"
                save_dbscan_file_step_dic=save_dbscan_path+"/"+file[:-4]+"/"
                dir_create(save_dbscan_file_step_dic)
                save_dbscan_file_step=save_dbscan_file_step_dic+file[:-4]
                if process:
                    sonar_data=np.load(data_path_single_file)
                else:   
                    if remove_type==0 and process!=0:
                        sonar_data,start,end=read_txt(data_path_single_file)   
                    else:
                        sonar_data,start,end=read_txt(data_path_single_file)
                sonar_data_vis=sonar_data
                print(data_path_single_file)
                ratio_scale=para_polar[0]/500.0
                if remove_type==1 or remove_type==0:
                    start=time.time()
                    sonar_data=background_select(sonar_data,bg,process)
                    print(time.time()-start)
                else:
                    pass
                if preprocess_flag==0:
                    sonar_data=data_rescale(sonar_data,para_scale[0],para_scale[1],para_scale[2],remove_type,process)
                    #sonar_data=data_remove_baseline(sonar_data)
                if preprocess_flag==1:
                    sonar_data=sonar_data #ban this function
                if preprocess_flag==2:
                    sonar_data=data_rescale(sonar_data,para_scale[0],para_scale[1],para_scale[2],remove_type,process)
                    sonar_data=sonar_data #ban this function 
                obj,latency=process_one_pic(sonar_data,save_dbscan_file_step,[15,15,15],[20,20,20],blur_size,human_size,max_blur,ratio_scale,latency=True) #20 25 19 27
                count+=1
                latency_denoise+=latency
                save_edge_detection(sonar_data,save_edge_img)
                sonar_data_vis=cv2.resize(sonar_data_vis,(1500,1200))
                vis_figure(sonar_data_vis,obj,save_name_img_vis+".png",data_detection_single)
                print("finish")
    latency_average=latency_denoise/count
    print(" ")
    print("latency:",latency_average)
    print(" ")
                    
    
def label_transfer(label_direct,label_save_path,label_type,preprocess_flag,para,process):
    scenario=os.listdir(label_direct)
    if ".DS_Store" in scenario:
        scenario.remove(".DS_Store")
    dir_create(label_save_path)
    for i in range(len(scenario)): 
        label_raw_path=label_direct+"/"+scenario[i]
        label_save_path_scenario=label_save_path+"/"+scenario[i]
        sonars=os.listdir(label_raw_path)
        dir_create(label_save_path_scenario)
        if ".DS_Store" in sonars:
            sonars.remove(".DS_Store")
        for sonar in sonars:
            label_path_single=label_raw_path+"/"+sonar
            save_path_single=label_save_path_scenario+"/"+sonar
            dir_create(save_path_single)
            files = os.listdir(label_path_single)
            if ".DS_Store" in files:
                files.remove(".DS_Store")
            for file in files:
                label_path_one=label_path_single+"/"+file
                save_label_single=save_path_single+"/"+file
                if process:
                    h,s,obj=read_default_label_raw(label_path_one)
                else:
                    if label_type==0:
                        h,s,obj=read_yolo_label(label_path_one)
                    elif label_type==1:
                        h,s,obj=read_default_label_noangle(label_path_one)
                    else:
                        h,s,obj=read_default_label_raw(label_path_one)
                #print("obj",obj)
                if preprocess_flag==0:
                    obj=label_trans_scale(para[0],para[1],obj)
                elif preprocess_flag==1:
                    obj=label_polar(obj,para[0])
                else:
                    obj=label_trans_scale(para[0],para[1],obj)
                    obj=label_polar(obj,para[2])
                label_save(h,s,obj,save_label_single)
                
def background_remove(bg,bg_re,sonar_data):
    clean=sonar_data-bg
    clean[clean<0]=0
    clean_re=sonar_data-bg_re
    clean_re[clean_re<0]=0
    if np.sum(clean_re)>np.sum(clean):
        return clean
    else:
        return clean_re

def background_remove_offest(bg,sonar_data):
    least_ener=10000000
    optiaml_bg=np.zeros_like(bg)
    for i in range(0,10):
        sonar_data_new=sonar_data
        bg_data_new=np.zeros_like(bg)
        for k in range(len(bg)):
            if k-i>=0:
                bg_data_new[k,:]=bg[k-i,:]
                print
            else:
                continue
        clean=sonar_data_new-bg_data_new
        clean[clean<0]=0
        if np.sum(clean)<least_ener: 
            least_ener=np.sum(clean)
            optiaml_bg=bg_data_new
    clean_data=sonar_data-optiaml_bg
    clean_data[clean_data<0]=0
    return clean_data
    
def background_select(sonar_data, bg_path,process):
    files=os.listdir(bg_path)
    least_energy=1000000
    opti_file=files[0]
    for i in range(0,2):
        bg,_,_=read_txt(bg_path+"/"+files[i])
        clean=background_remove_offest(bg,sonar_data)
        #clean=sonar_data-bg
        if np.sum(clean)<least_energy:
            least_energy=np.sum(clean)
            opti_file=files[i]
    bg_opti,_,_=read_txt(bg_path+"/"+opti_file)
    clean=background_remove_offest(bg_opti,sonar_data)
    if not process:
        clean[100:300,30:400]=sonar_data[100:300,30:400] #for all
    else:
        clean[90:310,30:400]=sonar_data[90:310,30:400] #for pre
    #clean=sonar_data-bg_opti
    return clean
                
def obtain_pre_data(dir,savedir,preprocess,label_dir,label_save,label_type,para,parap,para_label,blur_size,human_size,remove_type,bg,bg_sc,max_blur,pre):
    read_data_path(dir,savedir,preprocess,para,parap,pre,blur_size,human_size,remove_type,bg,bg_sc,max_blur)
    label_transfer(label_dir,label_save,label_type,preprocess,para_label,pre)

            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre", type=int, required=True, help="preprocess type")
    parser.add_argument("--data", type=str, required=True, help="data_path")
    parser.add_argument("--label",type=str, required=True, help="label_path")
    parser.add_argument("--label_type",type=int, required=True, help="label type")
    parser.add_argument("--parad",type=int, nargs="+",required=True, help="parad")
    parser.add_argument("--parap",type=int, nargs="+",required=True, help="parap")
    parser.add_argument("--paral",type=int, nargs="+",required=True, help="paral")
    parser.add_argument("--blur_size",type=int, nargs="+",required=True, help="blur")
    parser.add_argument("--human_size",type=int,required=True, help="human")
    parser.add_argument("--remove",type=int, required=True, help="remove type")
    parser.add_argument("--bg_path",type=str, required=True, help="bg_path")
    parser.add_argument("--bg_sc",type=str, required=True, help="bg_sc")
    parser.add_argument("--max_blur",type=int, required=True, help="max_blur")
    parser.add_argument("--process",type=int, required=True, help="pre")
    parser.add_argument("--obj_detect",type=str, required=True, help="pre")
    parser.add_argument("--obj_type",type=str, required=True, help="pre")
    parser.add_argument("--save_dir_all",type=str, required=True, help="pre")
    args = parser.parse_args()
    pre_type=args.pre
    data_dir=args.data
    label_dir=args.label
    parad=args.parad
    remove_type=args.remove
    bg_path=args.bg_path
    bg_sc=[args.bg_sc]
    max_blur=args.max_blur
    pre=args.process
    prefix=args.obj_detect
    save_dir_file_all=args.save_dir_all
    dir_create(save_dir_file_all)
    #data_dir=data_dir
    if pre==0:
        pre=False
    else:
        pre=True
    ratio=1.0
    if pre_type==0 or pre_type==2:
        ratio=np.int32(parad[-1]/500.0)
    
    if pre_type==0:
        save_dir_data=save_dir_file_all+"/"+data_dir+"_localize_"+str(ratio)
        save_dir_label=save_dir_file_all+"/"+label_dir+"_localize_"+str(ratio)
    elif pre_type==1:
        save_dir_data=data_dir+"_polar"
        save_dir_label=label_dir+"_polar"
    else:
        save_dir_data=data_dir+"_mix_filter"+str(ratio)
        save_dir_label=label_dir+"_mix_filter"+str(ratio)

    len_single=(len(parad)-2)/2
    len_single=np.int32(len_single)
    
    if len_single!=(len(parad)-2)/2.0:
        print("error parad")
        return
    dis_para=[]
    threshold_para=[]
    for i in range(len_single):
        threshold_para.append(parad[i])
    for i in range(len_single,(len(parad)-2)):
        dis_para.append(parad[i])
    size_re=[parad[-2],parad[-1]]
    parad_input=[threshold_para,dis_para,size_re]
    paral=args.paral
    label_type=args.label_type
    parap=args.parap
    human=np.int32(args.human_size)
    blur_size=args.blur_size
    
    print(parad_input)
    print(paral)
    scenario=[args.obj_type]
    obtain_pre_data(data_dir,save_dir_data,pre_type,label_dir,save_dir_label,label_type,parad_input,parap,paral,blur_size,human,remove_type,bg_path,bg_sc,max_blur,pre)
    print(save_dir_data,save_dir_label)
    main_metric(save_dir_data,save_dir_label,scenario,"e2e_"+prefix)
    
def main_metric(data_path,label_path,scenario,name):
    re=dm.eval_single(data_path,label_path,scenario,name)
    print(re)
        
if __name__=="__main__":
    main()
