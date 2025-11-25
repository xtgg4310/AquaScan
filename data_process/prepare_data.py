import os
import numpy as np
from tqdm import tqdm
import cv2
import math, cmath
from copy import deepcopy
import random
import time

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
            #print(angle,data)
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

    

def int2str(num, length=6):
    ''' convert int to str, and add 0 before the number
    Args:
        num: int number
        length: length of str, default: 6
    Returns:
        str_num: str number
    '''
    str_num = str(num)
    while len(str_num) < length:
        str_num = '0' + str_num
    return str_num

def object_polar2cart(sonar_data, obj, rotate=True):
    ''' erase other objects; transform from polar to cartesian; slice the object with rectangle
    Args:
        sonar_data: 3d array, shape: (channels, angle_range, 500)
        obj: [ymin, ymax, xmin, xmax]
        rotate: whether rotate the sonar data to make the object at around 0 degree
    Returns:
        polar_data: 3d array, like sonar picture and the sonar is at center of pic, shape: (channels, 1000, 1000)
        [xmin, ymin, xmax, ymax]: the object's position in polar_data (maybe after rotation)
    '''
    # if sonar_data.ndim == 2:
    #     sonar_data = sonar_data[:, np.newaxis]
    empty_data = np.zeros_like(sonar_data)
    y_mean = 0
    if rotate:
        y_mean = int((obj[0] + obj[1]) / 2)
    empty_data[obj[0]:obj[1], obj[2]:obj[3]] = sonar_data[obj[0]:obj[1], obj[2]:obj[3]]

    # sonar should be at center of pic
    polar_data = cv2.warpPolar(empty_data, (1000, 1000), (500, 500), 500,
                               cv2.WARP_FILL_OUTLIERS + cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    # center rotate y_mean degrees
    M = cv2.getRotationMatrix2D((500, 500), y_mean * 0.9, 1)
    polar_data = cv2.warpAffine(polar_data, M, (1000, 1000))

    left_up = cmath.rect(obj[2], math.radians((obj[0] - y_mean) * 0.9))
    left_down = cmath.rect(obj[2], math.radians((obj[1] - y_mean) * 0.9))
    right_up = cmath.rect(obj[3], math.radians((obj[0] - y_mean) * 0.9))
    right_down = cmath.rect(obj[3], math.radians((obj[1] - y_mean) * 0.9))

    max_r = cmath.rect(obj[3], math.radians(0)).real

    x = [left_up.real+500, left_down.real+500, right_up.real+500, right_down.real+500, max_r+500]
    xmin = min(x)
    xmax = max(x)
    xmin = math.floor(xmin)
    xmax = math.ceil(xmax)
    xmax = xmax + 5
    xmin = xmin - 5
    
    y = [left_up.imag+500, left_down.imag+500, right_up.imag+500, right_down.imag+500]
    ymin = min(y)
    ymax = max(y)
    ymin = math.floor(ymin)
    ymax = math.ceil(ymax)
    ymax = ymax + 5
    ymin = ymin - 5

    polar_data = polar_data[ymin:ymax, xmin:xmax]
    return polar_data, [xmin, ymin, xmax, ymax]

def get_clean_data(sonar_data,objs):
    empty_data=np.zeros_like(sonar_data)
    for obj in objs:
        print(obj)
        empty_data[obj[0]:obj[1],obj[2]:obj[3]]=sonar_data[obj[0]:obj[1],obj[2]:obj[3]]
    return empty_data

def norm_distance(sonar_data):
    ''' normalize sonar data based on distance. The center distance is the anchor.
        n = (pixel_x - anchor_x) / total_x * 
        pixel_strength = pixel_strength * (1 + n)
    Args:
        sonar_data: 2d array, shape: (400, 500);
                    The 250 column is the center of sonar data, and they don't have change.

    Returns:
        norm_data: 2d array, shape: (400, 500)
    '''

class SonarData():
    ''' SonarData class
    '''
    def __init__(self, channels=3,concate_len=2):
        self.save_file_idx = 0

        self.past_objects = []
        self.past_sonar_datas = []
        self.humans_id=[]
        self.past_data_paths = []
        self.past_label_paths = []
        self.channels = channels
        self.concate_len=concate_len
        self.bias_x=0
        self.bias_y=0
        self.record_last_datas=None
        self.record_last_objs=None

    def __checklen__(self):
        ''' check the length of past_objects and past_sonar_datas
        '''
        assert len(self.past_objects) == len(self.past_sonar_datas), 'past_objects and past_sonar_datas have different length'
        while len(self.past_objects) > self.channels:
            #print("add")
            self.past_objects.pop(0)
            self.past_sonar_datas.pop(0)
            self.past_data_paths.pop(0)
            self.past_label_paths.pop(0)
            self.humans_id.pop(0)
            
    def __concatlen__(self):
        assert len(self.past_objects) == len(self.past_sonar_datas), 'past_objects and past_sonar_datas have different length'
        while len(self.past_objects) > self.concate_len:
            self.past_objects.pop(0)
            self.past_sonar_datas.pop(0)
            self.past_data_paths.pop(0)
            self.past_label_paths.pop(0)
            self.humans_id.pop(0)
    
    def __clean__(self):
        ''' clean the data
        '''
        self.past_objects = []
        self.past_sonar_datas = []
        self.past_data_paths = []
        self.past_label_paths = []
        self.humans_id=[]

    def __skip_scanning__(image, skip, start_angle,end_angle, start, step):
        while start<start_angle:
            start+=step
        for i in range(start, end_angle, step):
            for j in range(skip):
                image[i + j, :] = 0
        return image
    
    def __process_one_pic_multi_ch__(self, sonar_data_path, label_file_path, pic_save_dir_path, label_save_dir_path, sonar_str,file,scenario="2022",scenario_set="1.1.1.1",time_start=0.0,last_file=0,compare_method=1):
        ''' process one pic, and it will be split into many pics contains different objects.
            This function only works on multi channel (consider time axis)
            Use IoU to get same object instead of human id.
        Label file format:
            The label file is default format.
            For human object: human id | state_str | xmin | ymin | xmax | ymax
            For noise object: state_str | xmin | ymin | xmax | ymax
            Note: some non-target human's id is -1
        Args:
            sonar_data_path: path of sonar data
            label_file_path: path of label file
            pic_save_dir_path: path to save processed data
            label_save_dir_path: path to save label file
        The saved pics will be *pic_save_dir_path*/*pic_name*/*idx*-state.png
        The saved label file will be *label_save_dir_path*/*pic_name*.txt
        Saved label file format: human id | state_str | pos | sonar_str | sonar_data_path | label_file_path
        '''

        sonar_data, start_angle, end_angle = read_txt(sonar_data_path)
        ratio=np.abs(end_angle-start_angle)/200.0
        file_order_raw=file.split('.')[0]
        #=file_order_raw
        if np.int32(file_order_raw)==0:
            time_start=0
            self.__clean__()
        #time_start=0
        if scenario=="2022":
            #scenario_set_time={"1.1.1.5":}
            time_single_frame=scenario_set#6.18*ratio/17*25
        else:
            time_single_frame=6.18*ratio/17.0*25
        file_order=((np.int32(file_order_raw)-time_start)*time_single_frame)%110
        if file_order>0:
            former_order=((np.int32(np.int32(last_file))-time_start)*time_single_frame)%110
        else:
            former_order=0
        if former_order>=file_order or last_file>np.int32(file_order_raw):
            
            self.__clean__()
            if np.int32(file_order_raw)>=last_file:
                time_start=np.int32(file_order_raw)
            else:
                time_start=last_file
            file_order=((np.int32(file_order_raw)-time_start)*time_single_frame)%110
        last_file=np.int32(file_order_raw)
        human_id_list, state_list, obj_list = read_default_label(label_file_path)
        self.past_objects.append(obj_list)
        self.humans_id.append(human_id_list)
        self.past_sonar_datas.append(sonar_data)
        self.past_data_paths.append(sonar_data_path)
        self.past_label_paths.append(label_file_path)
        self.__checklen__()
        if compare_method==0:
            candidate_regions, candidate_datas = self.compare_objects_raw()
        else:
            candidate_regions, candidate_datas = self.compare_objects()
        if candidate_regions is None and candidate_datas is None:
            return time_start,last_file
        self.record_last_datas=candidate_datas
        self.record_last_objs=candidate_regions
        for human_id, state, obj, data in zip(human_id_list, state_list, candidate_regions, candidate_datas):
            polar_data, pos = object_polar2cart(data, obj)
            save_data_path = os.path.join(pic_save_dir_path, int2str(self.save_file_idx) + '.png')
            save_label_path = os.path.join(label_save_dir_path, int2str(self.save_file_idx) + '.txt')
            print(obj,state,file_order,time_start,polar_data.shape)
            with open(save_label_path, 'w') as f:
                f.write(str(human_id) + '\n' + str(state) + '\n' + str(obj[0]) + ',' + str(obj[1]) + ',' + str(obj[2]) + ',' + str(obj[3]) + '\n' +
                        sonar_str + '\n' + ', '.join(self.past_data_paths) + '\n' + ', '.join(self.past_label_paths) + '\n')
            # cv2 stores image in BGR format
            cv2.imwrite(save_data_path, polar_data)
            self.save_file_idx += 1
        print(" ")
        return time_start,last_file

    def __process_one_pic_multi_ch_datalist__(self, data, label, pic_save_dir_path, label_save_dir_path,s,file_name,sonar,time_single=2.57):
        file_order=(file_name.split('_')[1]).split('.')[0]
    
        file_order=(np.int32(file_order)*time_single)
        time_start=time.time()
        sonar_data,start_angle,end_angle=data[0],data[1],data[2]
        human_id_list, state_list, obj_list=label[0],label[1],label[2]
        self.past_objects.append(obj_list)
        self.past_sonar_datas.append(sonar_data)
        #print(type(sonar_data))
        #self.past_data_paths
        self.past_data_paths.append("/0229_pos")
        self.past_label_paths.append(file_name)
        self.humans_id.append(human_id_list)
        print("file print")
        print(self.past_label_paths)
        print("current file")
        print(file_name)
        self.__checklen__()
        print(self.past_label_paths)
        #print(human_id_list, state_list, obj_list)
        # Use IoU to get same object positions from past frames with the current frame
        candidate_regions, candidate_datas = self.compare_objects()
        #print(candidate_datas[-1].shape)
        if candidate_regions is None and candidate_datas is None:
            return
        for human_id, state, obj, data in zip(human_id_list, state_list, candidate_regions, candidate_datas):
            polar_data, pos = object_polar2cart(data, obj)
            save_data_path = os.path.join(pic_save_dir_path, str(s)+"_"+int2str(self.save_file_idx)+"_"+str(sonar) + '.png')
            save_label_path = os.path.join(label_save_dir_path, str(s)+"_"+int2str(self.save_file_idx) +"_"+str(sonar)+ '.txt')
            #if state=='moving' or state=='Moving':
            #    continue
            with open(save_label_path, 'w') as f:
                f.write(str(human_id) + '\n' + str(state) + '\n' + str(obj[0]) + ',' + str(obj[1]) + ',' + str(obj[2]) + ',' + str(obj[3]) + '\n' +str(sonar)+"\n"+ str(s)+"\n"+str(file_name))
            # cv2 stores image in BGR format
            #np.save(save_data_path, polar_data)
            cv2.imwrite(save_data_path, polar_data)
            self.save_file_idx += 1 
        print(time.time()-time_start)
        print("\n")
    
    def compare_objects_raw(self,bias=False,concat=False): #use for both training and inferences
        empty_data = np.zeros_like(self.past_sonar_datas[-1])
        if not concat:
            while len(self.past_objects) < self.channels:
                self.past_objects.insert(0, self.past_objects[-1])
                self.past_sonar_datas.insert(0, empty_data)
                self.past_data_paths.insert(0, 'empty')
                self.past_label_paths.insert(0, 'empty')
        else:
            if len(self.past_objects) < self.concate_len:
                while len(self.past_objects) < self.channels:
                    self.past_objects.insert(0, self.past_objects[-1])
                    self.past_sonar_datas.insert(0, empty_data)
                    self.past_data_paths.insert(0, 'empty')
                    self.past_label_paths.insert(0, 'empty')
            else:
                split_seed=random.random()*(0.8-0.2)+0.2
                image_center=self.past_sonar_datas[0]
                obj_center_list=[]
                if random.random()>0.5:
                    direction = 1
                else:
                    direction = -1
                #print(self.past_objects)
                if len(self.past_objects[0])!=len(self.past_objects[1]):
                    return None,None
                for index in range(len(self.past_objects[0])):
                    y_min_1=self.past_objects[1][index][0]
                    y_max_1=self.past_objects[1][index][1]
                    y_min_2=self.past_objects[0][index][0]
                    y_max_2=self.past_objects[0][index][1]
                    x_min_1=self.past_objects[1][index][2]
                    x_max_1=self.past_objects[1][index][3]
                    x_min_2=self.past_objects[0][index][2]
                    x_max_2=self.past_objects[0][index][3]
                    
                    if direction==1:
                        end_angle=int((y_max_1-y_min_1)*split_seed+y_min_1)
                        start_angle=y_min_1
                    else:
                        end_angle=int(y_max_1-(y_max_1-y_min_1)*split_seed)
                        start_angle=y_max_1
                    image1=self.past_sonar_datas[-1]
                    image_center[start_angle:end_angle,x_min_1:x_max_1]=image1[start_angle:end_angle,x_min_1:x_max_1]
                    obj_center=[min(y_min_1,y_min_2),max(y_max_1,y_max_2),min(x_min_1,x_min_2),max(x_max_1,x_max_2)]
                    obj_center_list.append(obj_center)
                self.past_sonar_datas.insert(1,image_center)
                self.past_objects.insert(1,obj_center_list)
                self.past_data_paths.insert(1, 'empty')
                self.past_label_paths.insert(1, 'empty')
        total_frames = len(self.past_objects)
        # TODO: candidate_data对于每个object都是不同的，所以最后并不是只有一个candidate_data
        # 但是candidate_data的中间结果不能直接输出，需要想个办法解决
        # 所有object的基准都是以current frame为主的，也就是说candidate region不会超过current frame的范围
        # 但是再往前的frame计算是否是同一个obj是基于past objects的，所以要在代码逻辑上注意
        # 第一次loop之后，len(candidata_datas) == len(candidate_regions) == len(self.past_objects[-1])
        max_IoU=0.05
        iou_threshold=0.3 #raw design:iou_threshold=0.3
        last_objects = deepcopy(self.past_objects[-1])
        detect_flags = [True for _ in range(len(last_objects))]
        candidate_datas = [self.past_sonar_datas[-1] for _ in range(len(last_objects))]
        candidate_regions = deepcopy(last_objects)
        #print(candidate_regions)
        if len(last_objects) == 0:
            #print("out")
            #self.record_last_objs.append(self.record_last_objs[-1])
            #empty_can_data=[]
            #for i in range(len(self.record_last_datas)):
            #    print(self.record_last_datas[-1].shape)
            #    empty_data=np.zeros_like(self.record_last_datas[i])
            #    empty_can_data.append(empty_data)
            #self.record_last_datas.append(empty_can_data)
            #self.record_last_datas.pop(0)
            #self.record_last_objs.pop(0)
            #print(self.record_last_objs)
            #return self.record_last_objs, self.record_last_datas
            #find_flag=False
            #for i in range(len(self.past_objects)): 
            #    last_objects = deepcopy(self.past_objects[len(self.past_objects)-i-1])
            #    if len(last_objects)!=0:
            #        find_flag=True
            #        break
            #if not find_flag:
            last_objects = deepcopy(self.record_last_objs)
            candidate_regions = deepcopy(last_objects)
            detect_flags = [True for _ in range(len(last_objects))]
            candidate_datas = [empty_data for _ in range(len(last_objects))]
        for i in range(total_frames - 1, 0, -1):
            objects_old = self.past_objects[i - 1]
            for idx, region in enumerate(candidate_regions):
                last_obj = last_objects[idx]
                last_candidate_data = candidate_datas[idx]
                if last_obj is None:
                    candidate_region, candidate_data = self.combine_objects(region, region, 
                                            last_candidate_data, empty_data)
                    candidate_regions[idx] = candidate_region
                    candidate_datas[idx] = candidate_data
                if len(objects_old) == 0:
                    candidate_region, candidate_data = self.combine_objects(region, region, 
                                            last_candidate_data, empty_data)
                    candidate_regions[idx] = candidate_region
                    candidate_datas[idx] = candidate_data
                    last_objects[idx] = last_obj
                    detect_flags[idx] = False
                flag=False
                max_IoU=0.0
                for object_old in objects_old:
                    if calculate_iou_on_small(last_obj, object_old)>max_IoU:
                        max_IoU=calculate_iou_on_small(last_obj, object_old)
                        max_IoU_obj=object_old
                        flag=True
                if flag:
                    x_seed=random.random()
                    y_seed=random.random()
    
                    if x_seed<0.5:
                        self.bias_x=-1
                    else:
                        self.bias_x=1
                    if y_seed<0.5:
                        self.bias_y=-1
                    else:
                        self.bias_y=1
                        
                    if not bias or i != total_frames-1:
                        temp_data = self.get_clean_data(self.past_sonar_datas[i - 1], max_IoU_obj)
                    else:
                        temp_data,obj_new = self.extend_data(self.past_sonar_datas[i - 1], max_IoU_obj)
                    if bias and i == total_frames-1:
                        candidate_region, candidate_data = self.combine_objects(region, obj_new,
                                        last_candidate_data, temp_data)
                    else:
                        candidate_region, candidate_data = self.combine_objects(region, max_IoU_obj,
                                        last_candidate_data, temp_data)
                    candidate_regions[idx] = candidate_region
                    candidate_datas[idx] = candidate_data
                    last_objects[idx] = max_IoU_obj
                    #break
                else:
                    candidate_region, candidate_data = self.combine_objects(region, region, 
                                        last_candidate_data, empty_data)
                    candidate_regions[idx] = candidate_region
                    candidate_datas[idx] = candidate_data
                    last_objects[idx] = last_obj
                    detect_flags[idx] = False
        
        return candidate_regions, candidate_datas
        
    def compare_objects(self): #only for inference
        empty_data = np.zeros_like(self.past_sonar_datas[-1])
        while len(self.past_objects) < self.channels:
            self.past_objects.insert(0, self.past_objects[-1])
            self.past_sonar_datas.insert(0, empty_data)
            self.past_data_paths.insert(0, 'empty')
            self.past_label_paths.insert(0, 'empty')
            self.humans_id.insert(0,[])   
        total_frames = len(self.past_objects)
        # TODO: candidate_data对于每个object都是不同的，所以最后并不是只有一个candidate_data
        # 但是candidate_data的中间结果不能直接输出，需要想个办法解决
        # 所有object的基准都是以current frame为主的，也就是说candidate region不会超过current frame的范围
        # 但是再往前的frame计算是否是同一个obj是基于past objects的，所以要在代码逻辑上注意
        # 第一次loop之后，len(candidata_datas) == len(candidate_regions) == len(self.past_objects[-1])

        iou_threshold=0.3 #raw design:iou_threshold=0.3
        last_objects = deepcopy(self.past_objects[-1])
        detect_flags = [True for _ in range(len(last_objects))]
        candidate_datas = [self.past_sonar_datas[-1] for _ in range(len(last_objects))]
        candidate_regions = deepcopy(last_objects)
        if len(last_objects) == 0:
            return None, None
        for i in range(total_frames - 1, 0, -1):
            # former frame
            objects_old = self.past_objects[i - 1]
            humans_old_id=self.humans_id[i - 1]
            for idx, region in enumerate(candidate_regions):
                # compare last_objects and previous objects (objects_old). combine regions
                # then last_objects => previous objects
                last_obj = last_objects[idx]
                last_candidate_data = candidate_datas[idx]
                human_id=self.humans_id[-1][idx]
                if last_obj is None:
                    assert detect_flags[idx] == False
                    candidate_region, candidate_data = self.combine_objects(region, region, 
                                            last_candidate_data, empty_data)
                    candidate_regions[idx] = candidate_region
                    candidate_datas[idx] = candidate_data
                    continue
                if len(objects_old) == 0:
                    candidate_region, candidate_data = self.combine_objects(region, region, 
                                            last_candidate_data, empty_data)
                    candidate_regions[idx] = candidate_region
                    candidate_datas[idx] = candidate_data
                    last_objects[idx] = None
                    detect_flags[idx] = False
                    continue
                #max_IoU=0.0
                max_IoU_obj=None
                for humam_idx in range(len(humans_old_id)):
                    if human_id==humans_old_id[humam_idx]:
                        max_IoU_obj=objects_old[humam_idx]
                        break

                if max_IoU_obj!=None and calculate_iou_on_small(last_obj, max_IoU_obj)>=0.00:
                    temp_data = self.get_clean_data(self.past_sonar_datas[i - 1], max_IoU_obj)
                    candidate_region, candidate_data = self.combine_objects(region, max_IoU_obj,
                                    last_candidate_data, temp_data)
                    candidate_regions[idx] = candidate_region
                    candidate_datas[idx] = candidate_data
                    last_objects[idx] = max_IoU_obj
                    #break
                else:
                    candidate_region, candidate_data = self.combine_objects(region, region, 
                                        last_candidate_data, empty_data)
                    candidate_regions[idx] = candidate_region
                    candidate_datas[idx] = candidate_data
                    last_objects[idx] = None
                    detect_flags[idx] = False
                    #print(candidate_data.shape)
        #print(" ")
        return candidate_regions, candidate_datas
    
    '''
    def compare_objects(self,bias=False,concat=False): for both training and inference
        empty_data = np.zeros_like(self.past_sonar_datas[-1])
        if not concat:
            while len(self.past_objects) < self.channels:
                self.past_objects.insert(0, self.past_objects[-1])
                self.past_sonar_datas.insert(0, empty_data)
                self.past_data_paths.insert(0, 'empty')
                self.past_label_paths.insert(0, 'empty')
                self.humans_id.insert(0,[])
        else:
            if len(self.past_objects) < self.concate_len:
                while len(self.past_objects) < self.channels:
                    self.past_objects.insert(0, self.past_objects[-1])
                    self.past_sonar_datas.insert(0, empty_data)
                    self.past_data_paths.insert(0, 'empty')
                    self.past_label_paths.insert(0, 'empty')
                    self.humans_id.insert(0,[])
            else:
                split_seed=random.random()*(0.8-0.2)+0.2
                image_center=self.past_sonar_datas[0]
                obj_center_list=[]
                if random.random()>0.5:
                    direction = 1
                else:
                    direction = -1
                #print(self.past_objects)
                if len(self.past_objects[0])!=len(self.past_objects[1]):
                    return None,None
                for index in range(len(self.past_objects[0])):
                    y_min_1=self.past_objects[1][index][0]
                    y_max_1=self.past_objects[1][index][1]
                    y_min_2=self.past_objects[0][index][0]
                    y_max_2=self.past_objects[0][index][1]
                    x_min_1=self.past_objects[1][index][2]
                    x_max_1=self.past_objects[1][index][3]
                    x_min_2=self.past_objects[0][index][2]
                    x_max_2=self.past_objects[0][index][3]
                    
                    if direction==1:
                        end_angle=int((y_max_1-y_min_1)*split_seed+y_min_1)
                        start_angle=y_min_1
                    else:
                        end_angle=int(y_max_1-(y_max_1-y_min_1)*split_seed)
                        start_angle=y_max_1
                    image1=self.past_sonar_datas[-1]
                    image_center[start_angle:end_angle,x_min_1:x_max_1]=image1[start_angle:end_angle,x_min_1:x_max_1]
                    obj_center=[min(y_min_1,y_min_2),max(y_max_1,y_max_2),min(x_min_1,x_min_2),max(x_max_1,x_max_2)]
                    obj_center_list.append(obj_center)
                self.past_sonar_datas.insert(1,image_center)
                self.past_objects.insert(1,obj_center_list)
                self.past_data_paths.insert(1, 'empty')
                self.past_label_paths.insert(1, 'empty')
                self.humans_id.insert(1,[])
        total_frames = len(self.past_objects)
        # TODO: candidate_data对于每个object都是不同的，所以最后并不是只有一个candidate_data
        # 但是candidate_data的中间结果不能直接输出，需要想个办法解决
        # 所有object的基准都是以current frame为主的，也就是说candidate region不会超过current frame的范围
        # 但是再往前的frame计算是否是同一个obj是基于past objects的，所以要在代码逻辑上注意
        # 第一次loop之后，len(candidata_datas) == len(candidate_regions) == len(self.past_objects[-1])

        iou_threshold=0.3 #raw design:iou_threshold=0.3
        last_objects = deepcopy(self.past_objects[-1])
        detect_flags = [True for _ in range(len(last_objects))]
        candidate_datas = [self.past_sonar_datas[-1] for _ in range(len(last_objects))]
        candidate_regions = deepcopy(last_objects)
        if len(last_objects) == 0:
            return None, None
        for i in range(total_frames - 1, 0, -1):
            # former frame
            objects_old = self.past_objects[i - 1]
            humans_old_id=self.humans_id[i - 1]
            for idx, region in enumerate(candidate_regions):
                # compare last_objects and previous objects (objects_old). combine regions
                # then last_objects => previous objects
                last_obj = last_objects[idx]
                last_candidate_data = candidate_datas[idx]
                human_id=self.humans_id[-1][idx]
                if last_obj is None:
                    assert detect_flags[idx] == False
                    candidate_region, candidate_data = self.combine_objects(region, region, 
                                            last_candidate_data, empty_data)
                    candidate_regions[idx] = candidate_region
                    candidate_datas[idx] = candidate_data
                    continue
                if len(objects_old) == 0:
                    candidate_region, candidate_data = self.combine_objects(region, region, 
                                            last_candidate_data, empty_data)
                    candidate_regions[idx] = candidate_region
                    candidate_datas[idx] = candidate_data
                    last_objects[idx] = None
                    detect_flags[idx] = False
                    continue
                #max_IoU=0.0
                max_IoU_obj=None
                for humam_idx in range(len(humans_old_id)):
                    if human_id==humans_old_id[humam_idx]:
                        max_IoU_obj=objects_old[humam_idx]
                        break

                if max_IoU_obj!=None and calculate_iou_on_small(last_obj, max_IoU_obj)>=0.00:
                    # combine the same object in different frames
                    x_seed=random.random()
                    y_seed=random.random()
    
                    if x_seed<0.5:
                        self.bias_x=-1
                    else:
                        self.bias_x=1
                    if y_seed<0.5:
                        self.bias_y=-1
                    else:
                        self.bias_y=1
                        
                    if not bias or i != total_frames-1:
                        temp_data = self.get_clean_data(self.past_sonar_datas[i - 1], max_IoU_obj)
                    else:
                        temp_data,obj_new = self.extend_data(self.past_sonar_datas[i - 1], max_IoU_obj)
                    if bias and i == total_frames-1:
                        candidate_region, candidate_data = self.combine_objects(region, obj_new,
                                        last_candidate_data, temp_data)
                    else:
                        candidate_region, candidate_data = self.combine_objects(region, max_IoU_obj,
                                        last_candidate_data, temp_data)
                    candidate_regions[idx] = candidate_region
                    candidate_datas[idx] = candidate_data
                    last_objects[idx] = max_IoU_obj
                    #break
                else:
                    candidate_region, candidate_data = self.combine_objects(region, region, 
                                        last_candidate_data, empty_data)
                    candidate_regions[idx] = candidate_region
                    candidate_datas[idx] = candidate_data
                    last_objects[idx] = None
                    detect_flags[idx] = False
                    #print(candidate_data.shape)
        #print(" ")
        return candidate_regions, candidate_datas
     '''
    def get_clean_data(self, sonar_data, obj):
        empty_data = np.zeros_like(sonar_data)
        empty_data[obj[0]:obj[1], obj[2]:obj[3]] = sonar_data[obj[0]:obj[1], obj[2]:obj[3]]
        return empty_data

    def combine_objects(self, obj1, obj2, sonar_data_new, sonar_data_old):
        ymin = min(obj1[0], obj2[0])
        ymax = max(obj1[1], obj2[1])
        xmin = min(obj1[2], obj2[2])
        xmax = max(obj1[3], obj2[3])
        if sonar_data_old.ndim == 2:
            sonar_data_old = np.expand_dims(sonar_data_old, axis=2)
        if sonar_data_new.ndim == 2:
            sonar_data_new = np.expand_dims(sonar_data_new, axis=2)
        
        combined_sonar_data = np.concatenate((sonar_data_old, sonar_data_new), axis=2)
        return [ymin, ymax, xmin, xmax], combined_sonar_data
        
    def extend_data(self, sonar_data, obj):
        y_center=(obj[0]+obj[1])/2
        x_center=(obj[2]+obj[3])/2
        offset_y=y_center-obj[0]
        offset_x=x_center-obj[2]
        y_max=obj[1]
        y_min=obj[0]
        x_max=obj[3]
        x_min=obj[2]
        if self.bias_x==1:
            x_max=int(obj[3]+offset_x)
            x_min=obj[2]
        else:
            x_min=int(obj[2]-offset_x)
            x_max=obj[3]
            
        if self.bias_y==1:
            y_max=int(obj[1]+offset_y)
            y_min=obj[0]
        else:
            y_max=obj[1]
            y_min=int(obj[0]-offset_y)
        empty_data = np.zeros_like(sonar_data)
        empty_data[obj[0]:obj[1], obj[2]:obj[3]] = sonar_data[obj[0]:obj[1], obj[2]:obj[3]]
        extend_obj=[y_min,y_max,x_min,x_max]
        return empty_data,extend_obj
        
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
        print(lines)
        for line in lines:
            arr = line.strip().split()
            # human object
            if len(arr) == 6:
                human_id = arr[0]
                state = arr[1]
                xmin = int(float(arr[4])/1.0) #42
                ymin = int(float(arr[2])/1.0) #23
                xmax = int(float(arr[5])/1.0) #54
                ymax = int(float(arr[3])/1.0) #35
            
            # noise object
            elif len(arr) == 5:
                human_id = -2
                state = "noise"
                xmin = int(float(arr[1])) #31
                ymin = int(float(arr[2])) #12
                xmax = int(float(arr[3])) #43
                ymax = int(float(arr[4])) #24
            else:
                raise ValueError('label file format error: {}'.format(file_path))

            obj = [ymin, ymax, xmin, xmax]
            human_ids.append(human_id)
            states.append(state)
            objs.append(obj)
    return human_ids, states, objs

def label2yolo(human_id,states,objs):
    label_content=[]
    print(states)
    for i in range(len(human_id)):
        print(states[i])
        #print(objs[i])
        label_single=[states[i],objs[i][0],objs[i][1],objs[i][2],objs[i][3]]
        label_content.append(label_single)
    return label_content
        
def square2yolo(obj,W=500,H=400):
    obj_yolo=[]
    for i in range(len(obj)):
        ymin=np.int32(obj[i][0])
        ymax=np.int32(obj[i][1])
        xmin=np.int32(obj[i][2])
        xmax=np.int32(obj[i][3])
        x=(xmin+xmax)/(2.0*W)
        y=(ymin+ymax)/(2.0*H)
        w=(xmax-xmin)*1.0/W
        h=(ymax-ymin)*1.0/H
        new_obj=[x,y,w,h]
        obj_yolo.append(new_obj)
    return obj_yolo
        

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

def get_time(x):
    time_file=x.split('/')[-1].split('_')[0]
    time_data=np.int32(time_file.split('-')[-3])*3600+np.int32(time_file.split('-')[-2])*60+np.int32(time_file.split('-')[-1])
    return time_data

def split_dataset_prepare_data(aug_p,raw_data_path,label_path,save_path,train_scenario,test_scenario,val_scenario,scenario_data="2022"):
    data_save_path = os.path.join(save_path, 'data')
    #print(data_save_path)
    label_save_path = os.path.join(save_path, 'label')
    dir_create(data_save_path)
    dir_create(label_save_path)
    data_save_train_path=os.path.join(data_save_path, 'train_new_5cls')
    data_save_test_path=os.path.join(data_save_path, 'test_new_5cls')
    data_save_val_path=os.path.join(data_save_path, 'val_new_5cls')
    label_save_train_path=os.path.join(label_save_path, 'train_new_5cls')
    label_save_test_path=os.path.join(label_save_path, 'test_new_5cls')
    label_save_val_path=os.path.join(label_save_path, 'val_new_5cls')

    dir_create(data_save_train_path)
    dir_create(data_save_test_path)
    dir_create(data_save_val_path)
    dir_create(label_save_train_path)
    dir_create(label_save_test_path)
    dir_create(label_save_val_path)
    print(aug_p)
    save_file_idx = 0
    start_time=0
    sonar_data = SonarData(channels=30)
    print(sonar_data.channels)
    for scenario in tqdm(train_scenario):
        scenario = str(scenario)
        data_scenario_path = os.path.join(raw_data_path, scenario)
        label_scenario_path = os.path.join(label_path, scenario)
        if not os.path.exists(data_scenario_path):
            print("raw data path {} not exists".format(data_scenario_path))
            continue
        if scenario[0]==".":
            continue
        sonars = os.listdir(label_scenario_path)
        for sonar in sonars:
            #start_time=0  
            #current_time=0
            data_dir_path = os.path.join(data_scenario_path, sonar)
            label_dir_path = os.path.join(label_scenario_path, sonar)
            assert os.path.exists(data_dir_path), "sonar path {} not exists".format(data_dir_path)
            #print(sonar, scenario)
            if sonar[0]==".":
                continue
            #if sonar!="sonar2":
            #    continue
            files = os.listdir(label_dir_path)
            if len(files) == 0:
                continue
            if aug_p=="random":
                random.shuffle(files)
            else:
                if '_' in files[0]:
                    files.sort(key=get_time,reverse=False)#lambda x: int(x.split('/')[-1].split('-')), reverse=False)
                else:
                    files.sort(key=lambda x: int(x.split('.')[0]), reverse=False)
            sonar_data.__clean__()
            if '_' in files[0]:
                current_time=np.int32(files[0].split('_')[1])
            else:
                current_time=np.int32(files[0].split('.')[0])
            if '_' in files[0]:
                file_name_start=str(files[0].split('_')[1])+".txt"
            else:
                file_name_start=files[0]
            last_file=np.int32(file_name_start.split('.')[0])
            if '_' in files[0]:
                time_first=get_time(files[0])
                time_last=get_time(files[19])
                #print(files[0],files[-1],len(files))
                time_frame=(time_last-time_first)/(19)
            else:
                time_frame=6.18
            for file in files:
                if '_' in file:
                    file_name=str(file.split('_')[1])+".txt"
                else:
                    file_name=file
                data_file_path = os.path.join(data_dir_path, file)
                label_file_path = os.path.join(label_dir_path, file)
                print(label_file_path)
                assert os.path.exists(data_file_path), "data file {} not exists".format(data_file_path)
                if file[0]==".":
                    continue
                start_time=current_time
                #last_file=np.int32(file_name.split('.')[0])
                last_file_time=last_file
                print(start_time,last_file_time)
                if aug_p=="none" or aug_p=="random":
                    current_time,last_file = sonar_data.__process_one_pic_multi_ch__(data_file_path, label_file_path, data_save_train_path, label_save_train_path, sonar,file_name,five_class=True,scenario=scenario_data,scenario_set=time_frame,time_start=start_time,last_file=last_file_time)
                elif aug_p=="concat":
                    save_file_idx = sonar_data.__process_one_concat_multi_ch__(data_file_path, label_file_path, data_save_train_path, label_save_train_path, sonar)
                elif aug_p=="bias":
                    save_file_idx = sonar_data.__process_one_bias_multi_ch__(data_file_path, label_file_path, data_save_train_path, label_save_train_path, sonar)
                elif aug_p=="yolo":
                    save_file_idx = sonar_data.__process_one_pic_multi_yolo__(data_file_path, label_file_path, data_save_train_path, label_save_train_path, sonar,file_name)
                else:
                    print("type error")
    start_time=0   
    current_time=0        
    for scenario in tqdm(test_scenario):
        scenario = str(scenario)
        data_scenario_path = os.path.join(raw_data_path, scenario)
        label_scenario_path = os.path.join(label_path, scenario)
        if not os.path.exists(data_scenario_path):
            print("raw data path {} not exists".format(data_scenario_path))
            continue
        if scenario[0]==".":
            continue
        sonars = os.listdir(label_scenario_path)
        for sonar in sonars:
            start_time=0  
            current_time=0
            data_dir_path = os.path.join(data_scenario_path, sonar)
            label_dir_path = os.path.join(label_scenario_path, sonar)
            assert os.path.exists(data_dir_path), "sonar path {} not exists".format(data_dir_path)
            #print(sonar, scenario)
            if sonar[0]==".":
                continue
            files = os.listdir(label_dir_path)
            if len(files) == 0:
                continue
            if aug_p=="random":
                random.shuffle(files)
            else:
                if '_' in files[0]:
                    files.sort(key=get_time,reverse=False)
                else:
                    files.sort(key=lambda x: int(x.split('.')[0]), reverse=False)
            
            sonar_data.__clean__()
            if '_' in files[0]:
                current_time=np.int32(files[0].split('_')[1])
            else:
                current_time=np.int32(files[0].split('.')[0])
            if '_' in files[0]:
                file_name_start=str(files[0].split('_')[1])+".txt"
            else:
                file_name_start=files[0]
            last_file=np.int32(file_name_start.split('.')[0])
            if '_' in files[0]:
                time_first=get_time(files[0])
                time_last=get_time(files[19])
                time_frame=(time_last-time_first)/(19)
            else:
                time_frame=6.18
            for file in files:
                if '_' in file:
                    file_name=str(file.split('_')[1])+".txt"
                else:
                    file_name=file
                data_file_path = os.path.join(data_dir_path, file)
                label_file_path = os.path.join(label_dir_path, file)
                print(label_file_path)
                assert os.path.exists(data_file_path), "data file {} not exists".format(data_file_path)
                if file[0]==".":
                    continue
                start_time=current_time
                last_file_time=last_file
                if aug_p=="none" or aug_p=="random":
                    current_time,last_file = sonar_data.__process_one_pic_multi_ch__(data_file_path, label_file_path, data_save_test_path, label_save_test_path, sonar,file_name,five_class=True,scenario=scenario_data,scenario_set=time_frame,time_start=start_time,last_file=last_file_time)
                elif aug_p=="concat":
                    save_file_idx = sonar_data.__process_one_concat_multi_ch__(data_file_path, label_file_path, data_save_test_path, label_save_test_path, sonar)
                elif aug_p=="bias":
                    save_file_idx = sonar_data.__process_one_bias_multi_ch__(data_file_path, label_file_path, data_save_test_path, label_save_test_path, sonar)
                elif aug_p=="yolo":
                    save_file_idx = sonar_data.__process_one_pic_multi_yolo__(data_file_path, label_file_path, data_save_test_path, label_save_test_path, sonar,file_name)
                else:
                    print("type error")
    start_time=0
    current_time=0
    if aug_p=="none" or aug_p=="yolo":
        for scenario in tqdm(val_scenario):
            scenario = str(scenario)
            data_scenario_path = os.path.join(raw_data_path, scenario)
            label_scenario_path = os.path.join(label_path, scenario)
            if not os.path.exists(data_scenario_path):
                print("raw data path {} not exists".format(data_scenario_path))
                continue
            if scenario[0]==".":
                continue
            sonars = os.listdir(label_scenario_path)
            for sonar in sonars:
                start_time=0  
                current_time=0
                data_dir_path = os.path.join(data_scenario_path, sonar)
                label_dir_path = os.path.join(label_scenario_path, sonar)
                assert os.path.exists(data_dir_path), "sonar path {} not exists".format(data_dir_path)
                #print(sonar, scenario)
                if sonar[0]==".":
                    continue
                files = os.listdir(label_dir_path)
                if len(files) == 0:
                    continue
                if '_' in files[0]:
                    files.sort(key=get_time,reverse=False)
                else:
                    files.sort(key=lambda x: int(x.split('.')[0]), reverse=False)
                if '_' in files[0]:
                    current_time=np.int32(files[0].split('_')[1])
                else:
                    current_time=np.int32(files[0].split('.')[0])
                if '_' in files[0]:
                    file_name_start=str(files[0].split('_')[1])+".txt"
                else:
                    file_name_start=files[0]
                last_file=np.int32(file_name_start.split('.')[0])
                if '_' in files[0]:
                    time_first=get_time(files[0])
                    time_last=get_time(files[19])
                    time_frame=(time_last-time_first)/(19)
                else:
                    time_frame=6.18
                sonar_data.__clean__()
                for file in files:
                    if '_' in file:
                        file_name=str(file.split('_')[1])+".txt"
                    else:
                        file_name=file
                    data_file_path = os.path.join(data_dir_path, file)
                    label_file_path = os.path.join(label_dir_path, file)
                    print(label_file_path)
                    assert os.path.exists(data_file_path), "data file {} not exists".format(data_file_path)
                    if file[0]==".":
                        continue
                    start_time=current_time
                    last_file_time=last_file
                    if aug_p=="yolo":
                        save_file_idx = sonar_data.__process_one_pic_multi_yolo__(data_file_path, label_file_path, data_save_val_path, label_save_val_path, sonar,file_name)
                    else:
                        current_time,last_file = sonar_data.__process_one_pic_multi_ch__(data_file_path, label_file_path, data_save_val_path, label_save_val_path, sonar,file_name,five_class=True,scenario=scenario_data,scenario_set=time_frame,time_start=start_time,last_file=last_file_time)


def make_prepare_data(data_dir,label_dir,save_dir,s):
    data_path=os.path.join(data_dir,'txt')
    label_path= label_dir #os.path.join(label_dir,'img')
    dir_create(save_dir)
    save_file_dir=os.path.join(save_dir,"data")
    save_label_dir=os.path.join(save_dir,"label")
    dir_create(save_file_dir)
    dir_create(save_label_dir)
    data_list_files=os.listdir(data_path)
    label_list_files=os.listdir(label_path)
    if '_' in label_list_files[0]:
        label_list_files.sort(key=lambda x: int(x[:-4].split('_')[1]), reverse=True)
    else:
        label_list_files.sort(key=lambda x: int(x.split('.')[0]), reverse=True)
    Sonar_data = SonarData(channels=3)
    for i in range(len(label_list_files)):
        if data_list_files[i][0]=='.':
            continue
        file_path=os.path.join(data_path,label_list_files[i])
        label_file_path=os.path.join(label_path,label_list_files[i])
        sonar_data,start,end=read_txt_noangle(file_path)
        human_id_list, state_list, obj_list = read_yolo_label(label_file_path)
        data=[sonar_data,start,end]
        label=[human_id_list,state_list,obj_list]
        Sonar_data.__process_one_pic_multi_ch_datalist__(data,label,save_file_dir,save_label_dir,s,label_list_files[i])
        print(s,file_path,label_list_files[i])
        

def sonar_datalist_generate(data_path,label_path,save_path,eval=False):
    data_scenario_path=os.listdir(data_path)
    label_scenario_path=os.listdir(label_path)
    se_list=[]
    noangle_format=[]
    dir_create(save_path)
    print(data_scenario_path)
    print(label_scenario_path)
    for i in range(len(data_scenario_path)):
        if data_scenario_path[i][0]=='.':
            continue
        if data_scenario_path[i] not in label_scenario_path:
            continue
        print(data_scenario_path[i])
        #if data_scenario_path[i] not in target_list:
        #    continue
        if data_scenario_path[i] not in se_list:
            npy_file=False
        else:
            npy_file=True
        if data_scenario_path[i] not in noangle_format:
            noangle=False
        else:
            noangle=True
        scenario_path_data=os.path.join(data_path,str(data_scenario_path[i]))
        scenario_path_label=os.path.join(label_path,str(data_scenario_path[i]))

        if not eval:
            make_prepare_data(scenario_path_data,scenario_path_label,save_path,data_scenario_path[i])
        else:
            make_eval_dataset(scenario_path_data,scenario_path_label,save_path,data_scenario_path[i],npy_file,noangle)

def make_eval_dataset(data_dir,label_dir,save_dir,s,npy_file=False,noangle_format=False):
    data_path = data_dir
    label_path= label_dir #os.path.join(label_dir,'img')
    dir_create(save_dir)
    save_file_dir=os.path.join(save_dir,"data")
    save_label_dir=os.path.join(save_dir,"label")
    dir_create(save_file_dir)
    dir_create(save_label_dir)
    sonars=os.listdir(label_path)
    for sonar in sonars:
        if sonar[0]=='.':
            continue
        data_path_one=os.path.join(data_path,sonar)
        #data_path_one=data_path+"/txt/"
        label_path_one=os.path.join(label_path,sonar)
        data_list_files=os.listdir(data_path_one)
        label_list_files=os.listdir(label_path_one)
        if len(label_list_files)==0:
            continue
        if ".DS_Store" in label_list_files:
            label_list_files.remove(".DS_Store")
        if ".DS_Store" in data_list_files:
            data_list_files.remove(".DS_Store")
        if '_' in label_list_files[0]:
            label_list_files.sort(key=lambda x: int(x[:-4].split('_')[1]), reverse=False)
        else:
            label_list_files.sort(key=lambda x: int(x.split('.')[0]), reverse=False)
        Sonar_data = SonarData(channels=3)
        Sonar_data.__clean__()
        for i in range(len(label_list_files)):
            #print(label_list_files[i])
            if '_' in label_list_files[i]:
                file_name=label_list_files[i]#.split('_')[1]
            else:
                file_name=label_list_files[i]
            if data_list_files[i][0]=='.':
                continue
            #data_path_one=data_path_one+"/txt/"
            file_path=os.path.join(data_path_one,label_list_files[i])
            label_file_path=os.path.join(label_path_one,label_list_files[i])
            if not npy_file:
                if not noangle_format:
                    sonar_data,start,end=read_txt(file_path)
                else:
                    sonar_data,start,end=read_txt(file_path)
            else:
                file_path=file_path[:-4]+".npy"
                sonar_data=np.load(file_path)
                start=0
                end=399
            human_id_list, state_list, obj_list = read_default_label(label_file_path)
            data=[sonar_data,start,end]
            label=[human_id_list,state_list,obj_list]
            Sonar_data.__process_one_pic_multi_ch_datalist__(data,label,save_file_dir,save_label_dir,s,file_name,sonar)
            print(s,file_path,label_list_files[i])
                
def generate_specific_list(data_path,label_path,save_path,path_s,eval=False):
    scenario_path_data=os.path.join(data_path,str(path_s))
    scenario_path_label=os.path.join(label_path,str(path_s))
    if not eval:
        make_prepare_data(scenario_path_data,scenario_path_label,save_path,path_s)
    else:
        make_eval_dataset(scenario_path_data,scenario_path_label,save_path,path_s)
    
    
if __name__ == '__main__':
    pass
