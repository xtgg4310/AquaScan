import numpy as np
import os
import argparse
import random

def dir_create(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def read_pos_results(file_path):
    objs=[]
    with open(file_path,'r') as f:
        lines=f.readlines()
        if '\n' in lines:
            lines.remove('\n')
        for line in lines:
            arr=line.strip().split()
            x_aver = float(arr[0]) #0
            y_aver = float(arr[1]) #1
            x_mid = float(arr[2]) #2
            y_mid = float(arr[3]) #3
            raw_obj_y_min=float(arr[8])
            raw_obj_y_max=float(arr[9])
            raw_obj_x_min=float(arr[10])
            raw_obj_x_max=float(arr[11])
            raw_obj_y_min_sc=float(arr[12])
            raw_obj_y_max_sc=float(arr[13])
            raw_obj_x_min_sc=float(arr[14])
            raw_obj_x_max_sc=float(arr[15])
            state=arr[16] #useless,just for reference
            seg_raw_sc=[raw_obj_y_min_sc,raw_obj_y_max_sc,raw_obj_x_min_sc,raw_obj_x_max_sc]
            raw_obj=[raw_obj_y_min,raw_obj_y_max,raw_obj_x_min,raw_obj_x_max]
            obj=[x_aver,y_aver,x_mid,y_mid,raw_obj,seg_raw_sc,state]
            objs.append(obj)
    return objs

def label_read(label_dir,save_dir):
    dir_create(save_dir)
    obj_scenario={}
    scenario_key=[]
    save_label_file={}
    timestamp_list={}
    scenario=os.listdir(label_dir)
    if ".DS_Store" in scenario:
        scenario.remove(".DS_Store")
    for i in range(len(scenario)):   
        label_raw_path=label_dir+"/"+scenario[i]
        save_dir_scenario=save_dir+"/"+scenario[i]
        dir_create(save_dir_scenario)
        sonars=os.listdir(label_raw_path)
        if ".DS_Store" in sonars:
            sonars.remove(".DS_Store")
        for sonar in sonars:
            label_path_single=label_raw_path+"/"+sonar
            save_dir_sonar=save_dir_scenario+"/"+sonar
            dir_create(save_dir_sonar)
            files = os.listdir(label_path_single)
            if ".DS_Store" in files:
                files.remove(".DS_Store")
            count=0
            files.sort(key=lambda x: int(x.split('_')[1][:-4]), reverse=False)
            time_start=(files[0].split('_'))[0].split('-')
            time_start=np.float128(time_start[-2])*60+np.float128(time_start[-1])
            count=0
            save_key=(scenario[i],sonar)
            single_scenario=[]
            for file in files:
                time_file=(file.split('_'))[0].split('-')
                time_stamp_single=np.float128(time_file[-2])*60+np.float128(time_file[-1])-time_start
                label_path_one=label_path_single+"/"+file
                save_label_single=save_dir_sonar+"/"+file
                single_scenario.append(save_label_single)
                objs=read_pos_results(label_path_one)
                scenario_sonar_key=(scenario[i],sonar)
                if scenario_sonar_key not in obj_scenario:
                    obj_scenario.update({scenario_sonar_key:[[objs,time_stamp_single]]})
                else:
                    obj_scenario[scenario_sonar_key].append([objs,time_stamp_single])
                count+=1
            scenario_key.append(scenario_sonar_key)
            save_label_file.update({save_key:single_scenario})
            if scenario_sonar_key not in timestamp_list:
                timestamp_list.update({scenario_sonar_key:-1})
            else:
                timestamp_list[scenario_sonar_key]=-1
    return obj_scenario,scenario_key,save_label_file,timestamp_list
                    

class swimmer():
    def __init__(self, positions,seg,id):
        self.last_obj=None
        self.speed=np.array([0.0,0.0])
        self.positions=positions
        self.direction=None
        self.swim_score=-1
        self.delta_direction=0
        self.delta_speed=np.array([0.0,0.0])
        self.seg=seg
        self.timestamp=0.0
        self.state_label="none"
        self.frame_id=id
        
    def update_last(self,last_obj):
        self.last_obj=last_obj
        
    def update_timestamp(self,time):
        self.timestamp=time
        
    def update_speed(self):
        positions_cur=np.array(self.positions)
        position_last=np.array(self.last_obj.positions)
        time=self.timestamp-self.last_obj.timestamp
        self.speed=(positions_cur-position_last)/time
        
    def update_direction(self):
        self.direction=np.arctan2(self.speed[1],self.speed[0])*180/np.pi
                
    def update_obj(self,last_obj):
        self.last_obj=last_obj
        
    def update_state(self, state):
        self.state_label=state
    
    def update_delta(self):
        if self.last_obj!=None:
            self.delta_direction=self.direction-self.last_obj.direction
            self.delta_speed=self.speed-self.last_obj.speed
        
    def obtain_data(self):
        return self.positions
        
    def obtain_obj(self):
        return [self.positions,self.speed,self.swim_score,self.delta_direction,self.delta_speed]

    def obtain_timestamp(self):
        return self.timestamp
        
    def update_all(self,last_obj,time,timestamp):
        self.update_obj(last_obj)
        self.update_speed(time)
        self.update_direction()
        self.update_delta()
        self.update_timestamp(timestamp)
        
class Swimmer_score():
    def __init__(self,cur_obj,last_obj,weight):
        self.score=0
        self.last_obj=last_obj
        self.cur_obj=cur_obj
        self.weight=weight
        self.score=0.0
    
    def update_obj(self,new_obj,last_obj): 
        self.last_obj=last_obj
        self.cur_obj=new_obj
    
    def update_score(self,new_obj,last_obj,time):
        self.update_obj(new_obj,last_obj)
        self.score=self.motion_score(time)
        #update_seg(self,new_seg)
        return self.overall_score()
        
    def motion_score(self,time):
        cur_loc=self.last_obj.positions
        cur_speed=self.last_obj.speed
        delta_speed=self.last_obj.delta_speed
        pred_speed=cur_speed+delta_speed
        new_loc=self.cur_obj.positions
        pred_loc=cur_loc+pred_speed*time
        delta_loc=new_loc-pred_loc
        delta_dis=np.sqrt(delta_loc[0]**2+delta_loc[1]**2)
        return delta_dis
        
    def overall_score(self):
        score=self.weight[0]*self.score
        return score
        
class treenode():
    def __init__(self, obj):
        self.obj=obj
        self.child=[]
        self.parent=None
        self.score=0
        self.depth=0
        self.nearby_flag=True
        
    def obtain_obj(self):
        return self.obj
        
    def add_child(self,child):
        self.child.append(child)
        
    def check_child(self,obj):
        return obj in self.child
        
    def remove_child(self,child):
        self.child.remove(child)
            
    def update_score(self,score):
        self.score=score

class TrackTree():
    def __init__(self, root,id):
        self.len_tree=1
        self.root_node=root
        self.tree_dict=[root]
        self.max_tree=1
        self.bottom=[root]
        self.id=id
        self.latest_time=root.obj.timestamp
        self.end_flag=False
        
    def add_Node(self, Node, parent):
        if parent not in self.tree_dict:
            print("fail")
            return
        if self.len_tree==0:
            Node.ID=self.id
            self.tree_dict.append(Node)
            self.len_tree=1
            self.max_tree=1
        else:
            Node.ID=self.id
            self.tree_dict.append(Node)
            self.len_tree+=1
            Node.parent=parent
            Node.depth=parent.depth+1
            parent.add_child(Node)
            if Node.depth>self.max_tree:
                self.max_tree=Node.depth
                self.bottom.append(Node)
            elif Node.depth<=self.max_tree:
                self.bottom.append(Node)
            if parent in self.bottom:
                self.bottom.remove(parent)
        if  Node.obj.timestamp>=self.latest_time:
            self.latest_time=Node.obj.timestamp
       
    def update_end(self):
        self.end_flag=True
            
    def search_bottom_node(self,node):
        for i in range(len(self.bottom)):
            if self.bottom[i].positions == node.positions and self.bottom[i].timestamp == node.timestamp:
                return i
            else:
                continue
        return -1
    
    def print_Tree(self):
        print(self.latest_time)
        for i in range(len(self.tree_dict)):
            if self.tree_dict[i].parent!=None:
                print(self.tree_dict[i].obj.positions,"-",self.tree_dict[i].parent.obj.positions)
            else:
                print(self.tree_dict[i].obj.positions,"-")
                
    def obtain_all_trace(self):
        trace=[]
        check_none_trace=[]
        for i in range(len(self.bottom)):
            bottom_node=self.bottom[i]
            if bottom_node!=None and bottom_node.parent!=None:
                print(bottom_node.obj.timestamp,bottom_node.obj.positions,bottom_node.obj.state_label)
                print(bottom_node.parent.obj.timestamp,bottom_node.parent.obj.positions,bottom_node.parent.obj.state_label)
                print(" ")
                
            check_none_flag=True # if you want to ignore the trace where all subjects are none (false detection), you can set it as False. This paper evaluates the detection and recognition seperately so the confusion metric shows the accuracy of detected subjects and the miss and false detection is evaluated seperately. A better way to is combine two metric together.
            temp_child=bottom_node
            trace_single=[]
            while temp_child!=None:
                trace_single.append([temp_child.obj.timestamp,temp_child.obj.positions,temp_child.obj.seg,temp_child.obj.state_label,temp_child.obj.frame_id])
                if temp_child.obj.state_label!="None" and temp_child.obj.state_label!="none":
                    check_none_flag=True
                temp_child=temp_child.parent
            if check_none_flag:
                trace_single.reverse()
                trace.append(trace_single)
        
        return trace
            
                
def check_the_same_node(obj_list):
    single_list=[]
    conflict=[]
    for i in range(len(obj_list)):
        confilct_flag=False
        conflict_index=-1
        for j in range(len(obj_list)):
            if i==j:
                continue
            else:
                if obj_list[i][1]==obj_list[j][1] and obj_list[i][3]==obj_list[j][3]:
                    confilct_flag=True
                    conflict_index=j
                    break
                else:
                    continue
        if not confilct_flag:
            single_list.append(obj_list[i])
    return single_list,conflict

def obj_filter(obj_list,node_list,dis_filter,repeat_flag=False):
    obj_cleaned=[]
    for i in range(len(obj_list)):
        obj=obj_list[i]
        if (obj[0]<=dis_filter and obj[1] not in node_list) or (obj[0]<=dis_filter and repeat_flag):
            obj_cleaned.append(obj)
    obj_cleaned.sort(key=lambda x:x[0])
    return obj_cleaned

def obj_filter_sinle_match(obj_list):
    single_list=[]
    for i in range(len(obj_list)):
        repeat_flag=False
        for j in range(len(obj_list)):
            if i!=j and obj_list[i][2].obj.positions == obj_list[j][2].obj.positions:
                repeat_flag=True
                break
        if not repeat_flag:
            single_list.append(obj_list[i])
    return single_list

def remove_repeat(obj):
    obj_new=[]
    for i in range(len(obj)):
        repeat=False
        for j in range(len(obj_new)):
            if obj_new[j].obj.positions==obj[i].obj.positions:
                repeat=True
                break
        if not repeat:
            obj_new.append(obj[i])
    return obj_new
                
def obj_list_dict(obj):
    obj_node_dict={}
    for i in range(len(obj)):
        if obj[i][1] not in obj_node_dict:
            obj_node_dict.update({(obj[i][1],obj[i][3]):[[obj[i][0],obj[i][2]]]})
        else:
            obj_node_dict[(obj[i][1],obj[i][3])].append([obj[i][0],obj[i][2]])
    return obj_node_dict

def remove_obj(obj,pos):
    obj_new=[]
    for i in range(len(obj)):
        if obj[i][2].obj.positions!=pos:
            obj_new.append(obj[i])
    return obj_new

def check_list_only(node_list, conflict_list):
    count=0
    for i in range(len(conflict_list)):
        if conflict_list[i][2].obj.positions==node_list[2].obj.positions:
            count+=1
    if count==1:
        return True
    else:
        return False

class Tracker():
    def __init__(self,time):
        self.tracktree_dict={}
        self.time_stamp=time
        self.max_id=0
        
    def add_tree(self,root, id):
        new_track=TrackTree(root,id)
        self.tracktree_dict.update({id:new_track})
    
    def update_max_id(self):
        key_list=list(self.tracktree_dict.keys())
        if len(key_list)==0:
            return 0
        return max(key_list)
                
    def update_tree(self,id,obj,parent):
        tree_trace=self.tracktree_dict[id]
        tree_trace.add_Node(obj,parent)

    def single_node_tree_build(self,obj):
        obj_nearest_new=obj
        obj_track=[]
        bulid_new_tree=[]
        track_tree=[]
        obj_nearest_new.sort(key=lambda x:x[0])
        for i in range(len(obj_nearest_new)):
            add_list=obj_nearest_new[i]
            if add_list[0]>3.1 and add_list[2] not in bulid_new_tree and add_list[2] not in track_tree:
                max_id=self.update_max_id()
                self.add_tree(add_list[2],max_id+1)
                bulid_new_tree.append(add_list[2])
            else:
                exist_flag=False
                for k in range(len(obj_track)):
                    if add_list[2] == obj_track[k][2]:
                        exist_flag=True
                if not exist_flag and add_list[2] not in bulid_new_tree:
                    obj_track.append(add_list)
                    track_tree.append(add_list[2])
                    print(obj_nearest_new[i][0],obj_nearest_new[i][1].obj.positions,obj_nearest_new[i][2].obj.positions)
        return obj_track
        
    def tracking(self,obj_list,timestamp,stop_flag=False,time_threshold=12.0,dis_threshold=3.0): 
        scorer=Swimmer_score(None,None,[1.0])
        obj_wait_list=obj_list
        updated_node=[]
        for key in self.tracktree_dict:
            tree_single=self.tracktree_dict[key]
            if not self.tracktree_dict[key].end_flag:
                if (timestamp-self.tracktree_dict[key].latest_time)>time_threshold:
                    self.tracktree_dict[key].update_end()
                else:
                    for j in range(len(tree_single.bottom)):
                        updated_node.append([key,tree_single.bottom[j]])
        obj_wait_list=remove_repeat(obj_wait_list)
        obj_nearest=[]
        node_list=[]
        node_useful=[]
        node_len=len(updated_node)
        for node_id in range(node_len):
            N_leaf=updated_node[node_id][1]              
            if N_leaf.parent!=None:
                last_obj_leaf=N_leaf.parent.obj#.last_obj
                N_leaf.obj.update_last(last_obj_leaf)
                N_leaf.obj.update_speed()
            else:
                pass
        for obj in obj_wait_list:
            near_dis=10000
            near_node=None
            tree_index=-1
            for j in range(len(updated_node)):
                N_leaf=updated_node[j][1]
                timestamp_node=N_leaf.obj.obtain_timestamp()
                obj_time=obj.obj.obtain_timestamp()
                time_delta=obj_time-timestamp_node
                time_delta=0
                dis_temp=scorer.update_score(obj.obj,N_leaf.obj,time_delta)
                if dis_temp<near_dis:
                    near_dis=dis_temp
                    near_node=N_leaf
                    tree_index=updated_node[j][0]
                if dis_temp<dis_threshold and N_leaf not in node_useful:
                    node_useful.append(N_leaf)
            
            obj_nearest.append([near_dis,near_node,obj,tree_index])
        node_len=len(node_useful)
        obj_nearest_new=obj_nearest
        obj_track=[]
        bulid_new_tree=[]
        track_tree=[]
        obj_nearest_new.sort(key=lambda x:x[0])
        for i in range(len(obj_nearest_new)):
            add_list=obj_nearest_new[i]
            if (add_list[0]>=dis_threshold and add_list[2] not in bulid_new_tree and add_list[2] not in track_tree): 
                max_id=self.update_max_id()
                self.add_tree(add_list[2],max_id+1)
                bulid_new_tree.append(add_list[2])
                for obj_single in obj_wait_list:
                    if obj_single.obj.positions == add_list[2].obj.positions:
                        obj_wait_list.remove(obj_single)
            else:
                exist_flag=False
                for k in range(len(obj_track)):
                    if add_list[2] == obj_track[k][2]:
                        exist_flag=True
                        break
                if not exist_flag and add_list[2] not in bulid_new_tree:
                    obj_track.append(add_list)
                    track_tree.append(add_list[2])
                
        obj_nearest=obj_track
        obj_nearest_new,_=check_the_same_node(obj_nearest)
            
        obj_nearest_new.sort(key=lambda x:x[0])
        for i in range(len(obj_nearest_new)):
            add_list=obj_nearest_new[i]
            self.tracktree_dict[add_list[3]].add_Node(add_list[2],add_list[1])
            for obj_single in obj_wait_list:
                if obj_single.obj.positions == add_list[2].obj.positions:
                    obj_wait_list.remove(obj_single)
            node_list.append(add_list[1])
            node_len-=1
        index_count=0
        while len(obj_wait_list)!=0:
            if index_count>15:
                break
            obj_conflict=[]
            for obj in obj_wait_list:
                near_dis=10000     
                near_node=None
                tree_index=-1
                for k in range(len(updated_node)):
                    N_leaf=updated_node[k][1]
                    timestamp_node=N_leaf.obj.obtain_timestamp()
                    obj_time=obj.obj.obtain_timestamp()
                    time_delta=obj_time-timestamp_node
                    dis_temp=scorer.update_score(obj.obj,N_leaf.obj,time_delta)
                    obj_conflict.append([dis_temp,N_leaf,obj,updated_node[k][0]])
    
            obj_rest=obj_conflict
            obj_conflict=obj_filter(obj_conflict,node_list,dis_filter=dis_threshold) #3.0
            if obj_conflict==[]:
                obj_conflict=obj_filter(obj_rest,node_list,dis_threshold,True)  #3.0
            added_list=[]
            for obj_c in obj_conflict:
                #print(obj_wait_list)
                #print(obj_conflict)
                add_list=obj_c
                if add_list[2].obj.positions in added_list:
                    continue
                remove_flag=False
                if index_count<5:
                    if node_len>=len(obj_wait_list):
                        if (add_list[1] not in node_list and add_list[0]<dis_threshold) or (add_list[0]<dis_threshold and check_list_only(add_list,obj_conflict)):
                            self.tracktree_dict[add_list[3]].add_Node(add_list[2],add_list[1])
                            for obj_single in obj_wait_list:
                                if obj_single.obj.positions == add_list[2].obj.positions:
                                    obj_wait_list.remove(obj_single)
                            node_list.append(add_list[1])
                            node_len-=1
                            remove_flag=True
                        elif add_list[0]>dis_threshold:
                            max_id=self.update_max_id()
                            self.add_tree(add_list[2],max_id+1)
                            bulid_new_tree.append(add_list[2])
                            for obj_single in obj_wait_list:
                                if obj_single.obj.positions == add_list[2].obj.positions:
                                    obj_wait_list.remove(obj_single)
                            remove_flag=True
                    else:
                        self.tracktree_dict[add_list[3]].add_Node(add_list[2],add_list[1])
                        for obj_single in obj_wait_list:
                            if obj_single.obj.positions == add_list[2].obj.positions:
                                obj_wait_list.remove(obj_single)
                        remove_flag=True
                else:
                    if add_list[0]<dis_threshold:
                        self.tracktree_dict[add_list[3]].add_Node(add_list[2],add_list[1])
                        for obj_single in obj_wait_list:
                            if obj_single.obj.positions == add_list[2].obj.positions:
                                obj_wait_list.remove(obj_single)
                        remove_flag=True
                    else:
                        max_id=self.update_max_id()
                        self.add_tree(add_list[2],max_id+1)
                        bulid_new_tree.append(add_list[2])
                        for obj_single in obj_wait_list:
                            if obj_single.obj.positions == add_list[2].obj.positions:
                                obj_wait_list.remove(obj_single)
                        remove_flag=True
                if remove_flag:
                    obj_conflict=remove_obj(obj_conflict,add_list[2].obj.positions)
                    print(len(obj_conflict))
                    added_list.append(add_list[2].obj.positions)
                if len(obj_wait_list)==0:
                    break
            index_count+=1
            if len(obj_wait_list)==0:
                break
            
    def print_trace(self):
        for key in self.tracktree_dict:
            self.tracktree_dict[key].print_Tree()
            print(" ")
            
    def print_one_trace(self,key):
        self.tracktree_dict[key].print_Tree()
        
    def save_trace(self, trace_path):
        f=open(trace_path,"w")
        for key in self.tracktree_dict:
            for i in range(len(self.tracktree_dict[key].tree_dict)):
                if self.tracktree_dict[key].tree_dict[i].parent!=None:
                    record="["+str(self.tracktree_dict[key].tree_dict[i].obj.positions[0])+","+str(self.tracktree_dict[key].tree_dict[i].obj.positions[1])+"],["+str(self.tracktree_dict[key].tree_dict[i].obj.seg[0])+","+str(self.tracktree_dict[key].tree_dict[i].obj.seg[1])+","+str(self.tracktree_dict[key].tree_dict[i].obj.seg[2])+","+str(self.tracktree_dict[key].tree_dict[i].obj.seg[3])+","+str(self.tracktree_dict[key].tree_dict[i].obj.timestamp)+"]"+","+str(self.tracktree_dict[key].tree_dict[i].obj.state_label)+","+" -- "+"["+str(self.tracktree_dict[key].tree_dict[i].parent.obj.positions[0])+","+str(self.tracktree_dict[key].tree_dict[i].parent.obj.positions[1])+"],["+str(self.tracktree_dict[key].tree_dict[i].parent.obj.seg[0])+","+str(self.tracktree_dict[key].tree_dict[i].parent.obj.seg[1])+","+str(self.tracktree_dict[key].tree_dict[i].parent.obj.seg[2])+","+str(self.tracktree_dict[key].tree_dict[i].parent.obj.seg[3])+","+str(self.tracktree_dict[key].tree_dict[i].parent.obj.timestamp)+"]"+","+str(self.tracktree_dict[key].tree_dict[i].parent.obj.state_label)+"\n"
                else:
                    record="["+str(self.tracktree_dict[key].tree_dict[i].obj.positions[0])+","+str(self.tracktree_dict[key].tree_dict[i].obj.positions[1])+"],["+str(self.tracktree_dict[key].tree_dict[i].obj.seg[0])+","+str(self.tracktree_dict[key].tree_dict[i].obj.seg[1])+","+str(self.tracktree_dict[key].tree_dict[i].obj.seg[2])+","+str(self.tracktree_dict[key].tree_dict[i].obj.seg[3])+","+str(self.tracktree_dict[key].tree_dict[i].obj.timestamp)+"]"+","+str(self.tracktree_dict[key].tree_dict[i].obj.state_label)+"\n"
                f.writelines(record)
            f.writelines("\n")
    
    def split_all_tree_trace(self, save_file):
        for key in self.tracktree_dict:
            tree_single=self.tracktree_dict[key]
            trace_single=tree_single.obtain_all_trace()
            for i in range(len(trace_single)):
                save_file_single=save_file+"_"+str(key)+"_"+str(i)+".txt"
                f=open(save_file_single,"w")
                for k in range(len(trace_single[i])):
                    record=str(trace_single[i][k][0])+" "+str(trace_single[i][k][1][0])+" "+str(trace_single[i][k][1][1])+" "+str(trace_single[i][k][2][0])+" "+str(trace_single[i][k][2][1])+" "+str(trace_single[i][k][2][2])+" "+str(trace_single[i][k][2][3])+" "+str(trace_single[i][k][3])+"\n"
                    f.writelines(record)
                f.close()
                
    def re_id_all_trace(self, save_file_list):
        dict_id={}
        one_single_frame_time=self.time_stamp
        for key in self.tracktree_dict:
            tree_single=self.tracktree_dict[key]
            trace_single=tree_single.obtain_all_trace()
            for i in range(len(trace_single)):
                key_trace=str(key)+"_"+str(i)
                dict_id.update({key_trace:trace_single[i]})
        for i in range(len(save_file_list)):
            f=open(save_file_list[i],"w")
            for key in dict_id:
                if len(dict_id[key])==0:
                    continue
                time=dict_id[key][0][4]
                frame_id=time 
                if frame_id==i:
                    record=str(key)+" "+str(dict_id[key][0][3])+" "+str(dict_id[key][0][2][0])+" "+str(dict_id[key][0][2][1])+" "+str(dict_id[key][0][2][2])+" "+str(dict_id[key][0][2][3])+"\n"
                    f.writelines(record)
                    dict_id[key].remove(dict_id[key][0])
            f.close()
                
                
def convert_pos_obj_node(obj_list,timestamp,id=0):
    obj_node_list=[]
    for i in range(len(obj_list)):
        obj_=swimmer(obj_list[i][0],obj_list[i][1],id)
        obj_.update_timestamp(timestamp)
        obj_.update_state(obj_list[i][2])
        obj_node=treenode(obj_)
        obj_node_list.append(obj_node)
    return obj_node_list
    
def eval_track_tree(obj_list,time_list,flag=False,time_threshold=12.0,dis_threshold=3.0):
    tracker=Tracker(-1)
    for i in range(len(obj_list[0])):
        obj_=swimmer(obj_list[0][i][0],obj_list[0][i][1],0)
        obj_.update_timestamp(time_list[0])
        obj_.update_state(obj_list[0][i][2])
        obj_node=treenode(obj_)
        tracker.add_tree(obj_node,i)
    for i in range(1,len(obj_list)):
        time_frame=time_list[i]
        obj_node_list=convert_pos_obj_node(obj_list[i],time_frame,i)
        tracker.tracking(obj_node_list,time_frame,flag,time_threshold,dis_threshold)
    return tracker
    
def read_and_track(label_path,save_dir_label):
    save_dir=save_dir_label
    obj,keys,save_file_list,time_list=label_read(label_path,save_dir)
    invalid_data=[]
    for i in range(len(keys)):
        key=keys[i]
        for k in range(len(obj[key])):
            invalid_flag=False
            for j in range(len(obj[key][k][0])):
                if obj[key][k][0][j][0]==np.NaN:
                    invalid_flag=True
                    break
            if invalid_flag:
                invalid_data.append(obj[key][k])
    return obj,keys,save_file_list,time_list

def tracking(objs, keys, results_dir,save_trace_dir,save_list,thre_cfg=[]):
    for i in range(0,len(keys)):
        if thre_cfg==[]:
            # you can config the thre_cfg manually here,replace 111 with dir.
            time_threshold={"111":12.0}
            dis_threshold={"111":3.0}
            
            dis_single=dis_threshold[keys[i][0]]
            time_single_thre=time_threshold[keys[i][0]]
        else:
            dis_single=thre_cfg[0]
            time_single_thre=thre_cfg[1]
        obj_single=objs[keys[i]]
        obj=[]
        time=[]
        for j in range(len(obj_single)):
            pos=[]
            for k in range(len(obj_single[j][0])):
                pos.append([[obj_single[j][0][k][0],obj_single[j][0][k][1]],obj_single[j][0][k][4],obj_single[j][0][k][6]])
            obj.append(pos)
            time.append(obj_single[j][1])
            print(pos,obj_single[j][1])
        tracker=eval_track_tree(obj,time,True,time_single_thre,dis_single)
        print(keys[i][0],keys[i][1])
        results_file=results_dir+"/"+str(keys[i][0])+"_"+str(keys[i][1])+".txt"
        tracker.save_trace(results_file)
        save_trace_single=save_trace_dir+"/"+str(keys[i][0])+"_"+str(keys[i][1])
        tracker.split_all_tree_trace(save_trace_single)
        tracker.re_id_all_trace(save_list[keys[i]])
        
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="data_path")
    parser.add_argument("--label",type=str, required=True, help="label_path")
    parser.add_argument("--track",type=str, required=True, help="tracking")
    parser.add_argument("--track_re",type=str, required=True, help="track_results")
    parser.add_argument("--cfg",type=int, nargs="+", required=True, help="cfg")
    parser.add_argument("--save_dir_all",type=str, required=True, help="save_dir")
    args = parser.parse_args()
    save_dir=args.save_dir_all
    save_label=save_dir+"/"+args.label
    objs,keys,save_list,_=read_and_track(save_dir+"/"+args.data,save_label)
    results_dir=save_dir+"/"+args.track_re
    trace_dir=save_dir+"/"+args.track
    if args.cfg!=[0]:
        cfg_track=args.cfg
    else:
        cfg_track=[]
    print(cfg_track)
    dir_create(results_dir)
    dir_create(trace_dir)
    tracking(objs,keys,results_dir,trace_dir,save_list,cfg_track)
