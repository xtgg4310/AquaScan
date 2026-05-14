import os
import argparse
import subprocess
    
def all_datalist_save(datalist_dir,save_dir,save_name):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    datalist = [line.split() for line in open(datalist_dir).readlines()]
    #auglist=[line.split() for line in open(aug_dir).readlines()]
    data_paths = [info[0] for info in datalist]
    label_paths = [info[1] for info in datalist]
    #labels = [str(open(label_path).readlines()[1].strip()) for label_path in label_paths]

    save_path= os.path.join(save_dir, save_name+".txt")
    
    with open(save_path,"w") as f:
        for idx in range(len(data_paths)):
            f.write("{} {}\n".format(data_paths[idx], label_paths[idx]))
    f.close()

if __name__ == "__main__":
    pass
