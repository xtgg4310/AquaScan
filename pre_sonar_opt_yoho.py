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

def fuse_denoised_objects(denoised_objs, raw_objs, iou_threshold=0.3):
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
    #refer to the pre_sonar.py
    data_new=data

    return data_new

def data_pre(data, threshold, distance):
    if len(threshold)!=len(distance):
        print("unalign-bg-noise-remove")
        return
    print(data.shape)
    data_new=data_remove(data,threshold,distance)
    #data_new=cv2.resize(data_new,(500,re_size[0]))
    return data_new

def data_rescale(data,threshold,distance=[500],re_size=[1200,1500]):
    if len(threshold)!=len(distance):
        print("unalign-bg-noise-remove")
        return
    print(data.shape)
    data_new=data_remove(data,threshold,distance)
    #print(data_new.shape)
    data_new=cv2.resize(data_new,(re_size[1],re_size[0]))
    print(data_new.shape)
    data_new=np.uint8(data_new)
    #data_new=cv2.medianBlur(data_new,17)
    ratio=re_size[1]/500.0
    distance_new=np.zeros_like(distance)
    for i in range(len(distance)):
        distance_new[i]=ratio*distance[i]
    threshold_new=np.zeros_like(threshold)
    for i in range(len(threshold)):
        threshold_new[i]=threshold[i]/3.0
    data_new=data_remove(data_new,threshold_new,distance_new)
    return data_new

def localize_one_pic(temp_data, save_data,save_flag=False,threshold=45, min_samples=10, eps=12, blur_size=5, min_size=25):
    """
    Transform sonar data to point location, which can be used in cluster algorithm
    :param temp_data: One sonar picture.
    :param threshold: The lowest strength of point for filter.
    :return:
    """
    temp_data_new = temp_data.astype(np.uint8)
    #print(id(temp_data_new))
    #print(id(temp_data))
    if blur_size > 0:
        temp_data_new = cv2.medianBlur(temp_data_new, blur_size)
    
    dbscan = DBSCAN(min_samples=min_samples, eps=eps)
    object_poses = []

    point_pos = sonar2pos(temp_data_new, threshold)
    print(point_pos.shape)
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
    
def process_one_pic(sonar_data, save_path,eps=[15,15,15],min_sample=[25,25,25],max_blur=25,ratio=3.0,latency=False): #ratio=1.0
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
            'blur_size': 3,
            'min_size': 15,
        },
        'denoise2' : {
            # for weak objects
            'threshold': 5,
            'min_samples': min_sample[1],
            'eps': eps[1],#17
            'blur_size': 5,#9
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
        'opti_de' : {
            # for strong noise
            'threshold': 5,
            'min_samples': min_sample[0],
            'eps': eps[0],#10
            'blur_size': 13,#17
            'min_size': 15,
        },
        'max_de' : {
            # for strong noise
            'threshold': 5,
            'min_samples': min_sample[0],
            'eps': eps[0],#10
            'blur_size': 19,#17
            'min_size': 15,
        },
    }
    save_data_single=""
    index=0
    denoised_object_poses = []
    raw_object_poses = []
    sonar_data = copy.deepcopy(sonar_data)
    #sonar_data[:, :60] = 0
    #save_data_single=save_path+"_denoise_"+str(index)
    #index+=1
    #print(id(sonar_data))
    result_poses = []
    denoised_object_poses = localize_one_pic(sonar_data,save_data_single, **denoise_configs['denoise1'])
    if len(denoised_object_poses) == 0:
        save_data_single=save_path+"_denoise_"+str(index)
        index+=1
        denoised_object_poses = localize_one_pic(sonar_data,save_data_single, **denoise_configs['denoise2'])
    pre_results=denoised_object_poses
    K_size=5
    if obj_physical_filter(denoised_object_poses,3975*ratio*ratio,40*ratio,100*ratio):
        denoised_object_poses_judge_max=localize_one_pic(sonar_data,save_data_single, **denoise_configs['max_de'])
        if obj_physical_filter(denoised_object_poses_judge_max,3975*ratio*ratio,40*ratio,100*ratio):
            denoised_object_poses=denoised_object_poses_judge_max
            K_size=19
        else:
            denoised_object_poses_judge = localize_one_pic(sonar_data,save_data_single, **denoise_configs['opti_de'])
            if obj_physical_filter(denoised_object_poses_judge,3975*ratio*ratio,40*ratio,100*ratio):
                K_size=13
                while obj_physical_filter(denoised_object_poses_judge,3975*ratio*ratio,40*ratio,100*ratio):#2600*ratio*ratio #7.4 2.77
                    K_size+=2 #2
                    denoise_configs['denoise2']['blur_size']=K_size
                    denoised_object_poses_judge = localize_one_pic(sonar_data,save_data_single, **denoise_configs['denoise2'])
                    #print(K_size)
                    #for raw_config in raw_configs.values():
                    #    raw_object_poses = localize_one_pic(sonar_data, **raw_config)
                    #    if len(raw_object_poses) > 0:
                    #        break
                    #object_poses = fuse_objects(denoised_object_poses, raw_object_poses)
                    if K_size>=19: #17
                        break
                denoised_object_poses=denoised_object_poses_judge
            else:
                K_size=5
                last_results=pre_results
                current_results=pre_results
                while obj_physical_filter(pre_results,3975*ratio*ratio,40*ratio,100*ratio):#2600*ratio*ratio #7.4 2.77
                    K_size+=2 #2
                    #last_results=current_results
                    #print(pre_results)
                    denoise_configs['denoise2']['blur_size']=K_size
                    pre_results = localize_one_pic(sonar_data,save_data_single, **denoise_configs['denoise2'])
                    if K_size>=13: #17
                        break
            #print(K_size)
                denoised_object_poses=pre_results
    else:
        pass
    if K_size!=denoise_configs['denoise3']['blur_size']:
        objs_well_denoise= localize_one_pic(sonar_data,save_data_single,**denoise_configs['denoise3'])
    #else:
    #    objs_well_denoise=denoised_object_poses
    #objs_well_denoise
        #print(objs_well_denoise,denoised_object_poses)
        object_poses=fuse_denoised_objects(objs_well_denoise,denoised_object_poses) #denoised_object_poses
    else:
        object_poses=denoised_object_poses
    ###
    
    #object_poses=denoised_object_poses
    obj_idx=0
    while obj_idx<len(object_poses):
        if obj_tiny_filter(object_poses[obj_idx],human_size=120*ratio*ratio,angle_thre=10*ratio,length_threshold=10*ratio):
            object_poses.remove(object_poses[obj_idx])
            obj_idx-=1
        obj_idx+=1
    for obj in object_poses:
        result_poses.append(obj)
        #pass
    if len(result_poses) == 0:
        result_poses = denoised_object_poses
    if len(result_poses) == 0:
        result_poses = raw_object_poses
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
        if(obj[1]-obj[0])*(obj[3]-obj[2])>human_size or np.abs(obj[1]-obj[0])>angle_thre or np.abs(obj[3]-obj[2])>length_threshold:
            return True
    return False

def obj_tiny_filter(obj,human_size=120*3,angle_thre=5*3,length_threshold=10*3):
    if 300<obj[2]<400:
        ratio_angle=1.0
    elif obj[2]>=400:
        ratio_angle=1.0
    else:
        ratio_angle=1.0
    if(obj[1]-obj[0])*(obj[3]-obj[2])<=human_size or np.abs(obj[1]-obj[0])<=angle_thre*ratio_angle or np.abs(obj[3]-obj[2])<=length_threshold:
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
    a.rectangle(((re[2], re[0]), (re[3], re[1])), fill=None, outline='blue', width=1)
    #print((re[2], re[0]), (re[3], re[1]))
    # text = [str(swimmer_idx)]

    text = []
   # for p in result_dict.items():
        # append %.2f of p[1]
        #print(p)
    #    text.append("{:.2f}".format(p[1]))
    #a.text((re[2] - 5, re[0] - 20), '\n'.join(text), fill='red')
    return im, re[2], re[0], re[3], re[1]

def visualize_none_swimmer(data,im):
    #re = swimmer.positions[-1]
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
                #swimmer.classify(model, sonar_pics, gpu_id, args.onnx)
                #print(swimmer.postprocess())
                #result_dict,detect_object = swimmer.postprocess()
                #print(result_dict,detect_object)
                result_dict=[]
                im, xmin, ymin, xmax, ymax= visualize(swimmer, data, result_dict, im, i)
                #if detect_object == 'people':
                f.write(str(xmin)+ ','+ str(ymin) + ','+ str(xmax) + ',' + str(ymax) + ',\n')
                                #im.save('denoise_bbox/medianblur/1117/'+str(scenario)+'/img/{}.png'.format(index))
                #im.save(save_path_scenairo_img+"denoise_"+files[idx][:-4]+".png")
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
    """
    Transform sonar data to point location, which can be used in cluster algorithm
    :param sonar_data: One sonar picture.
    :param threshold: The lowest strength of point.
    :return:
    """
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

def data_polar(data,thershold, distance, length,remove_flag=True):
    if remove_flag:
        data_new=data_remove(data,thershold,distance)
    y_mean=0
    if remove_flag:
        polar_data=cv2.warpPolar(data_new,(length*2,length*2),(length,length),length,cv2.WARP_FILL_OUTLIERS + cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        polar_data=cv2.warpPolar(data,(length*2,length*2),(length,length),length,cv2.WARP_FILL_OUTLIERS + cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    M = cv2.getRotationMatrix2D((length, length), y_mean * 0.9, 1)
    polar_data = cv2.warpAffine(polar_data, M, (length*2, length*2))
    return polar_data

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
    
    
def read_data_path(dirc,save_path,preprocess_flag=0,para_scale=[],para_polar=[500]):
    scenario=os.listdir(dirc)
    count=0
    latency_denoise=0
    if ".DS_Store" in scenario:
        scenario.remove(".DS_Store")
    dir_create(save_path)
    for i in range(len(scenario)):    
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
            
            for file in files:
                data_detection_result=save_path_single+"/txt/"
                dir_create(data_detection_result)
                data_detection_single=data_detection_result+file[:-4]+".txt"
                data_path_single_file=data_path_single+"/"+file
                #save_name_npy=save_path_single+"/"+file[:-4]
                #dir_create(save_path_single+"/npy/")
                save_name_img=save_path_single+"/img/"+file[:-4]
                save_name_img_vis=save_path_single+"/img/"+file[:-4]+"_vis"
                dir_create(save_path_single+"/img/")
                #print(data_path_single,save_name_npy)
                save_dbscan_path=save_path_single+"/vis/"
                dir_create(save_dbscan_path)
                save_edge=save_path_single+"/edge/"
                save_edge_img=save_edge+file[:-4]+".png"
                dir_create(save_edge)
                save_dbscan_file=save_dbscan_path+file[:-4]+".png"
                save_dbscan_file_step_dic=save_dbscan_path+"/"+file[:-4]+"/"
                dir_create(save_dbscan_file_step_dic)
                save_dbscan_file_step=save_dbscan_file_step_dic+file[:-4]
                sonar_data,_,_=read_txt(data_path_single_file)
                print(data_path_single_file)
                ratio_scale=para_polar[0]/500.0
                if preprocess_flag==0:
                    sonar_data=data_rescale(sonar_data,para_scale[0],para_scale[1],para_scale[2])
                    #sonar_data=sonar_data
                if preprocess_flag==1:
                    sonar_data=data_polar(sonar_data,para_scale[0],para_scale[1],para_polar[0])
                if preprocess_flag==2:
                    sonar_data=data_rescale(sonar_data,para_scale[0],para_scale[1],para_scale[2])
                    sonar_data=data_polar(sonar_data,para_scale[0],para_scale[1],para_polar[0],remove_flag=False)
                obj,latency=process_one_pic(sonar_data,save_dbscan_file_step,[15,15,15],[20,20,20],19,ratio_scale,latency=True) #20 25 19
                count+=1
                #save_edge_detection(sonar_data,save_edge_img)
                latency_denoise+=latency
                vis_figure(sonar_data,obj,save_name_img_vis+".png",data_detection_single)
                #save_data(sonar_data,save_name_npy)
                #save_data_img(sonar_data,save_name_img)
                print("finish")
    latency_average=latency_denoise/count
    print(" ")
    print("latency:",latency_average)
    print(" ")
    
def label_transfer(label_direct,label_save_path,label_type,preprocess_flag,para):
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
                if label_type==0:
                    h,s,obj=read_yolo_label(label_path_one)
                elif label_type==1:
                    h,s,obj=read_default_label(label_path_one)
                else:
                    h,s,obj=read_default_label_raw(label_path_one)
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
                                
def obtain_pre_data(dir,savedir,preprocess,label_dir,label_save,label_type,para,parap,para_label):
    read_data_path(dir,savedir,preprocess,para,parap)
    label_transfer(label_dir,label_save,label_type,preprocess,para_label)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre", type=int, required=True, help="preprocess type")
    parser.add_argument("--data", type=str, required=True, help="data_path")
    parser.add_argument("--label",type=str, required=True, help="label_path")
    parser.add_argument("--label_type",type=int, required=True, help="label type")
    parser.add_argument("--parad",type=int, nargs="+",required=True, help="parad")
    parser.add_argument("--parap",type=int, nargs="+",required=True, help="parap")
    parser.add_argument("--paral",type=int, nargs="+",required=True, help="paral")
    parser.add_argument("--obj_detect",type=str, required=True, help="pre")
    parser.add_argument("--obj_type",type=str, required=True, help="pre")
    parser.add_argument("--save_dir_all",type=str, required=True, help="pre")
    args = parser.parse_args()
    pre_type=args.pre
    data_dir=args.data
    label_dir=args.label
    parad=args.parad
    prefix=args.obj_detect
    ratio=1.0
    if pre_type==0 or pre_type==2:
        ratio=np.int32(parad[-1]/500.0)
    save_dir_file_all=args.save_dir_all
    dir_create(save_dir_file_all)
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
    
    print(parad_input)
    print(paral)
    scenario=["111"]
    obtain_pre_data(data_dir,save_dir_data,pre_type,label_dir,save_dir_label,label_type,parad_input,parap,paral)
    main_metric(save_dir_data,save_dir_label,scenario,"e2e_"+prefix)
    
def main_metric(data_path,label_path,scenario,name):
    re=dm.eval_single(data_path,label_path,scenario,name)
    print(re)
            
if __name__=="__main__":
    main()
    
