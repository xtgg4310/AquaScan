import generate_datalist as gdata
import prepare_data as pdata
import split_datalist as sdata
import argparse
import os
import numpy as np
from sklearn.model_selection import train_test_split
import psutil
import time

def dir_create(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def generate_data_label_list(path_list):
    datalist_none=[]
    index_none=[]
    #print(len(path_list_none),len(path_list))
    for i in range(len(path_list)):
        #for j in range(len(label_list)):
        data_list_path=path_list[i]+"/data"
        label_list_path=path_list[i]+"/label"
        datalist=gdata.walk_file_list(data_list_path,label_list_path)
        datalist_none.append(datalist)
    
    for i in range(len(datalist_none)):
        index_none.append(i)
    return datalist_none,index_none

def file_combine(path_1,path_2,new_file,file_name):
    data_label_list=[]
    if not os.path.exists(new_file):
        os.makedirs(new_file)
    datalist_1 = [line.split() for line in open(path_1).readlines()]
    #auglist=[line.split() for line in open(aug_dir).readlines()]
    data_paths_1 = [info[0] for info in datalist_1]
    label_paths_1 = [info[1] for info in datalist_1]
    #labels = [str(open(label_path).readlines()[1].strip()) for label_path in label_paths]
    datalist_2 = [line.split() for line in open(path_2).readlines()]
    #auglist=[line.split() for line in open(aug_dir).readlines()]
    data_paths_2 = [info[0] for info in datalist_2]
    label_paths_2 = [info[1] for info in datalist_2]

    save_path= os.path.join(new_file, file_name+".txt")
    
    with open(save_path,"w") as f:
        for idx in range(len(data_paths_1)):
            f.write("{} {}\n".format(data_paths_1[idx], label_paths_1[idx]))
        for idx in range(len(data_paths_2)):
            f.write("{} {}\n".format(data_paths_2[idx], label_paths_2[idx]))
    f.close()
                            
def main_eval(data_read_path,label_read_path,save_read_path,save_file_name): #modify the name of variable
    data_path=data_read_path
    label_path=label_read_path  
    save_data=save_read_path
    save_path="./data_process/data_processed/"+save_data
    path_list=[]
    data_path_one=data_path
    save_path_one=save_path
    start=time.time()
    pdata.sonar_datalist_generate(data_path_one,label_path,save_path_one,True)     
    print((time.time()-start))
    path_list.append(save_path_one)
    dataset_list=[]
    for i in range(len(path_list)):
        data_list_path=path_list[i]+"/data"
        label_list_path=path_list[i]+"/label"
        datalist=gdata.walk_file_list(data_list_path,label_list_path)
        dataset_list.append(datalist)
    save_data_dir="./data_process/datalist_"+save_file_name+"/"
    dir_create(save_data_dir)
    save_data_path="./data_process/datalist_"+save_file_name+"/datalist.txt"
    gen_data_dir="./data_process/datalist_generate_"+save_file_name
    dir_create(gen_data_dir)
    gen_data_path="./data_process/datalist_generat_"+save_file_name
    gdata.save_list_all(save_data_path,dataset_list)
    sdata.split_datalist_save(save_data_path,gen_data_path,save_file_name)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="data_path")
    parser.add_argument("--label", type=str, required=True, help="label_path")
    parser.add_argument("--save", type=str, required=True, help="save_path")
    parser.add_argument("--file", type=str, required=True, help="file")
    parser.add_argument("--save_dir_all", type=str, required=True, help="save_dir")
    args = parser.parse_args()
    data_dir=args.data
    save_dir_all=args.save_dir_all
    label_dir=save_dir_all+"/"+args.label
    save=args.save
    file=args.file
    main_eval(data_dir,label_dir,save,file)

