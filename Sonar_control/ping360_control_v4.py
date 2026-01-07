from brping import Ping360
from brping import definitions
import numpy as np
import time
import argparse
from sonar_display import show_sonar
from data import *
import matplotlib.pyplot as plt
import matplotlib
import Track_scan
import os
from sonar_display import *
from datetime import datetime

matplotlib.use('Agg')
min_human = 0.4
max_human = 1.5
sample_fixed_duration = 25e-9
_firmwareMaxTransmitDuration = 500
_firmwareMinTransmitDuration = 5
_firmwareMaxNumberOfPoints = 1200
_firmwareMinSamplePeriod = 80
_firmwareMaxSamplePeriod = 40000
speed_of_sound = 1500
background_path = ""

def cal_sample_period(dis, number_samples):
    return 2 * dis / (number_samples * speed_of_sound * sample_fixed_duration)


def samplePeriod(sample_period):
    return sample_period * sample_fixed_duration


def transmitDurationMax(sample_period):
    return min(_firmwareMaxTransmitDuration, samplePeriod(sample_period) * 64e6)


def adjustTransmitDuration(distance, sample_period):
    transmit_duration = round(8000 * distance / speed_of_sound)
    transmit_duration = max(2.5 * sample_period / 1000, transmit_duration)
    return max(_firmwareMinTransmitDuration, min(transmitDurationMax(sample_period), transmit_duration))


def rescan(former_object, distance, number_sample, k, images):
    # change the setting to suit different object
    temp_distance = max(former_object[4], 8)
    temp_number_sample = round(number_sample * temp_distance / distance)
    temp_sample_period = cal_sample_period(temp_distance, temp_number_sample)
    temp_sample_period = round(temp_sample_period)
    temp_transmit_duration = adjustTransmitDuration(temp_distance, temp_sample_period)

    p.set_sample_period(temp_sample_period)
    p.set_number_of_samples(temp_number_sample)
    p.set_transmit_duration(temp_transmit_duration)

    for l in range(k):
        image = np.zeros((number_sample, former_object[2] - former_object[1] + 1))
        for i in range(former_object[1], former_object[2] + 1):
            p.control_transducer(
                0,
                p._gain_setting,
                i,
                p._transmit_duration,
                p._sample_period,
                p._transmit_frequency,
                p._number_of_samples,
                1,
                0
            )
            p.wait_message([definitions.PING360_DEVICE_DATA], 0.5)
            data = [int(n) for n in p._data]
            image[:temp_number_sample, i - former_object[1]] = data
        images = np.concatenate((images, image[:, :, np.newaxis]), axis=2)
    # return images[int(number_sample*former_object[3]/distance):int(number_sample*former_object[4]/distance), :, :]
    return images


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Control the sonar')
    parser.add_argument('--mode', type=int, default=3, help="0-time control scan result, 1-count control scan result, 2-tracking schemes")
    parser.add_argument('--udp', action="store", required=False, type=str, default="192.168.1.13:12345", help="host:port")
    parser.add_argument('--speed', default=1500, required=False, help="define the speed of sound underwater")
    parser.add_argument('--dis', default=15, type=int, required=False, help="define the sonar ranging")
    parser.add_argument('--trans_freq', default=750, type=int, required=False, help="define sonar frequency")
    parser.add_argument('--gain_setting', default=1, type=int, required=False, help="sonar gain setting")
    parser.add_argument('--number_samples', default=500, type=int, required=False, help="number of samples per scanning angle")

    parser.add_argument('--b', default=False, required=False, help="Whether we need to remove the background echoes")
    parser.add_argument('--background', default="", required=False, help="background_echoes record path")
    parser.add_argument('--save', default="mode_", required=False, help="data storage path")
    parser.add_argument('--fig',default="figure_sonar_",required=False,help="data figure storage path")
    parser.add_argument('--display', default=False, type=bool, help="display the scan result(only for mode 0)")
    parser.add_argument('--m', default = 8, type = int, help="Execute time-month")
    parser.add_argument('--D', default = 0, type = int, help="Execute time-day")
    parser.add_argument('--H', default = 0, type = int, help="Execute time-hour")
    parser.add_argument('--M', default = 0, type = int, help="Execute time-minute")

    ## repeat  time or cycles
    parser.add_argument('--time', default=120,type=int, help="one scanning duration")
    parser.add_argument('--count',default=3,type=int,help="total scanning counts")

    parser.add_argument('--sonar_num', default='4', type=str)
    parser.add_argument('--batch_id', default='02221002', type=str)

    parser.add_argument('--start',default=100,type=int)
    parser.add_argument('--end',default= 200,type=int)
    parser.add_argument('--skip', required=False, type=int, default=2, help="start angle")
    parser.add_argument('--step', required=False, type=int, default=3, help="scan step")

    parser.add_argument('--polar_pic_path', required=False, type=str, default="polar_pic",
                        help="sonar polar picture save path ")
    parser.add_argument('--xy_pic_path', required=False, type=str, default="xy_pic",
                        help="sonar xy picture save path")

    args = parser.parse_args()
    threshold = [10, 200, 3, 100]  # object filter
    baudrate = 115200
    start_time = datetime(2024,args.m,args.D, args.H,args.M,0)
    delay = (start_time-datetime.now()).total_seconds()
    time.sleep(delay)
    dis = args.dis
    duration = args.time
    total_count=args.count
#    dis = args.dis
    speed_of_sound = args.speed
    start_angle = args.start
    stop_angle = args.end
    scan_speed = args.step

    # sonar setting
    sonar_num = args.sonar_num
    batch_id = args.batch_id
    scan_mode = args.mode
    transmit_frequency = args.trans_freq
    number_samples = args.number_samples
    gain_setting = args.gain_setting


    sample_period = round(cal_sample_period(dis, number_samples))
    transmit_duration = adjustTransmitDuration(dis, sample_period)
    path_save_data =args.save

    (host, port) = args.udp.split(':')
    baudrate = 115200
    p = Ping360()
    p.connect_udp(host, int(port))
    p.initialize()
    # device = "/dev/ttyUSB0"
    # p = Ping360()
    # p.connect_serial(device, baudrate)
    # p.initialize()
    time_start = time.time()
    p.set_gain_setting(gain_setting)
    time_1_end = time.time()
    time_1 = time_1_end - time_start
    print(time_1)

    p.set_transmit_frequency(transmit_frequency)
    time_2_end = time.time()
    time_2 = time_2_end - time_1_end
    print(time_2)

    p.set_sample_period(sample_period)
    time_3_end = time.time()
    time_3 = time_3_end - time_2_end
    print(time_3)

    p.set_number_of_samples(number_samples)
    time_4_end = time.time()
    time_4 = time_4_end - time_3_end
    print(time_4)

    p.set_transmit_duration(transmit_duration)
    
    time_end = time.time()
    count_time = time_end - time_start
    print(count_time)


    if args.b:
        sonar_image_ref = np.zeros((number_samples, 400))
        f_background = open(background_path, "w")
        lines = f_background.readlines()
        for line in lines:
            angle, data = readline(line)
            if len(data) == number_samples:
                sonar_image_ref[:, angle] = data
        f_background.close()

    if args.mode == 0:
        # file_save=open(local_time+"_"+path_save_data,"w")
        start_time = time.time()
        x_angle = start_angle
        count = 0
        start = True
        while time.time() < start_time + duration:
            if start:
                local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                file_save = open("mode_"+str(args.mode)+"/"+local_time + "_" + str(count) + "_" + ".txt", "w")
                sonar_img = np.zeros((number_samples, int(400 / scan_speed) + 1))
                fig=plt.figure(figsize=(8,6),dpi=200)
                start = False
            if x_angle > stop_angle:
                x_angle = x_angle - stop_angle - 1 + start_angle
            else:
                p.control_transducer(
                    0,
                    p._gain_setting,
                    x_angle,
                    p._transmit_duration,
                    p._sample_period,
                    p._transmit_frequency,
                    p._number_of_samples,
                    1,
                    0,
                )
                p.wait_message([definitions.PING360_DEVICE_DATA], 0.5)
                data = [j for j in p._data]
                if len(data) == 0:
                    x_angle += 1
                else:
                    if args.b:
                        data = abs(data - sonar_image_ref[:, angle])
                    file_save.write(str(x_angle) + " ")
                    for j in range(len(data)):
                        file_save.write(str(data[j]) + " ")
                    file_save.write("\n")
                    if len(data) > 0:
                        sonar_img[:, int(x_angle / scan_speed)] = data
                    x_angle += scan_speed
                if x_angle > stop_angle:
                    start = True
                    count += 1
                    file_save.close()
                    if args.display:
                        show_sonar(sonar_img, dis)
                        #plt.savefig("mode_0/"+local_time+".png",dpi=200,bbox_inches = 'tight')
                        plt.axis('equal')
                        #plt.savefig("mode_"+str(args.mode)+"/"+local_time + "_" + str(count) +"figure.png")
                        plt.close()
    elif args.mode==1:
        x_angle = start_angle
        count = 0
        start = True
        while count < total_count:
            if start:
                local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                # file_save = open("mode_" + str(args.mode) + "/" + local_time + "_" + str(count) + "_" + path_save_data,
                #                  "w")

                txt_save_path= os.path.join(args.save + str(args.mode), batch_id, 'sonar' + sonar_num)
                os.makedirs(txt_save_path, exist_ok=True)
                file_save = open("mode_"+ str(args.mode) +"/" + batch_id + '/sonar' + sonar_num + '/' + local_time + '_' + str(count) + '.txt', 'w')

                sonar_img = np.zeros((number_samples, int(400 / scan_speed) + 1))
                # fig=plt.figure(figsize=(8,6),dpi=200)
                print(count)
                start = False
            if x_angle > stop_angle:
                x_angle = x_angle - stop_angle - 1 + start_angle
            else:
                p.control_transducer(
                    0,
                    p._gain_setting,
                    x_angle,
                    p._transmit_duration,
                    p._sample_period,
                    p._transmit_frequency,
                    p._number_of_samples,
                    1,
                    0,
                )
                p.wait_message([definitions.PING360_DEVICE_DATA], 0.5)
                data = [j for j in p._data]
                if len(data) == 0:
                    x_angle += 1
                else:
                    if args.b:
                        data = abs(data - sonar_image_ref[:, angle])
                    file_save.write(str(x_angle) + " ")
                    for j in range(len(data)):
                        file_save.write(str(data[j]) + " ")
                    file_save.write("\n")
                    if len(data) > 0:
                        sonar_img[:, int(x_angle / scan_speed)] = data
                    x_angle += scan_speed
                if x_angle > stop_angle:
                    start = True
                    count += 1
                    file_save.close()
                    if args.display:
                        show_sonar(sonar_img, dis)
                        # plt.savefig("figure/mode_1/"+local_time+".png",dpi=200,bbox_inches = 'tight')
                        plt.axis('equal')
                        plt.savefig("figure/mode_" + str(args.mode) + "/" + local_time + "_" + str(count) + ".png")
                        plt.close()

    elif args.mode == 2:
        start_angle = args.start
        end_angle = args.end
        scan_continue = args.step
        skipping = args.skip
        scanning_angle = Track_scan.ignore_scanning_scheme(start_angle, stop_angle, scan_continue, skipping)
        count = 0
        while count < total_count:
            print(count)
            local_time=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            txt_save_path = os.path.join(path_save_data + str(args.mode) +"/" ,  batch_id, 'sonar' + sonar_num)
            os.makedirs(txt_save_path, exist_ok=True)
            fileObject = open("mode_"+ str(args.mode) +"/"+ batch_id + '/sonar'+ sonar_num + '/'+ local_time+ '_'+ str(count) + '.txt', 'w')
            for i in range(len(scanning_angle)):
                # if args.data:
                    p.control_transducer(
                        0,  # reserved
                        p._gain_setting,
                        scanning_angle[i],
                        p._transmit_duration,
                        p._sample_period,
                        p._transmit_frequency,
                        p._number_of_samples,
                        1,
                        0
                    )
                    p.wait_message([definitions.PING360_DEVICE_DATA], 0.5) #wait_message works to allow sonar to transmit the data within the largest time(here is 0.5). It is a function in ROS. there is no docu for wait_message now, if you worry about this function , you can use ping360 document to check new functions
                    new_message = [int(j) for j in p._data]
                    fileObject.write(str(scanning_angle[i])+" ")
                    for j in range(len(new_message)):
                        fileObject.write(str(new_message[j])+" ")
                    fileObject.write("\n")
                #if len(new_message)>0:
                #    sonar_img[:,int(x/step)]=new_message
            fileObject.close()
            count = count + 1
    elif args.mode == 3:
        start_angle = args.start
        end_angle = args.end
        scan_step = args.step
        count = 0
        while count < total_count:
            print(count)
            local_time=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            txt_save_path = os.path.join(path_save_data + str(args.mode) +"/" ,  batch_id, 'sonar' + sonar_num)
            os.makedirs(txt_save_path, exist_ok=True)
            fileObject = open("mode_"+ str(args.mode) +"/"+ batch_id + '/sonar'+ sonar_num + '/'+ local_time+ '_'+ str(count) + '.txt', 'w')
            scanning_angle = Track_scan.back_forth_scanning_scheme(start_angle, end_angle, scan_step, count)
            for i in range(len(scanning_angle)):
                # if args.data:
                    p.control_transducer(
                        0,  # reserved
                        p._gain_setting,
                        scanning_angle[i],
                        p._transmit_duration,
                        p._sample_period,
                        p._transmit_frequency,
                        p._number_of_samples,
                        1,
                        0
                    )
                    p.wait_message([definitions.PING360_DEVICE_DATA], 0.5)
                    new_message = [int(j) for j in p._data]
                    fileObject.write(str(scanning_angle[i])+" ")
                    for j in range(len(new_message)):
                        fileObject.write(str(new_message[j])+" ")
                    fileObject.write("\n")
                #if len(new_message)>0:
                #    sonar_img[:,int(x/step)]=new_message
            fileObject.close()
            count = count + 1
        ## start to draw 10 frames figures
        sonar_data_path = os.path.join('mode_' + str(args.mode), batch_id, 'sonar' + sonar_num)
        polar_pic_path = os.path.join(args.polar_pic_path, batch_id, 'sonar' + sonar_num)
        xy_pic_path = os.path.join(args.xy_pic_path, batch_id, 'sonar' + sonar_num)
        all_files = os.listdir(sonar_data_path)
        print("Start to draw some pictures")
        if len(all_files) >= 10:
            files = all_files[:10]
        else:
            files = all_files
        for file in files:
            if file[0] == '.':
                continue
            else:
                sonar_data_file = os.path.join(sonar_data_path, file)
                f = open(sonar_data_file, 'r')
                temp_data, start_angle, end_angle = read_txt(sonar_data_file)

                ## draw polar pictures
                sonar_tmp_data = np.transpose(temp_data)
                show_sonar(sonar_tmp_data, args.dis)
                os.makedirs(polar_pic_path, exist_ok=True)
                plt.savefig(os.path.join(polar_pic_path, str(file[:-4]) + '.png'))
                # plt.show()
                plt.clf()

                ## draw xy sonar pictures
                os.makedirs(xy_pic_path, exist_ok=True)
                draw_one_pic(temp_data, os.path.join(xy_pic_path, str(file[:-4]) + '.png'))
                # plt.show()
                plt.clf()

    elif args.mode == 4:
        start_angle = args.start
        end_angle = args.end
        scan_continue = args.step
        skipping = args.skip
        count = 0

        while count < total_count:
            print(count)
            local_time=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            txt_save_path = os.path.join(path_save_data + str(args.mode) +"/" ,  batch_id, 'sonar' + sonar_num)
            os.makedirs(txt_save_path, exist_ok=True)
            fileObject = open("mode_"+ str(args.mode) +"/"+ batch_id + '/sonar'+ sonar_num + '/'+ local_time+ '_'+ str(count) + '.txt', 'w')
            if count%2 == 0:
                scanning_angle = Track_scan.ignore_scanning_scheme(start_angle, stop_angle, scan_continue, skipping)
            else:
                scanning_angle = Track_scan.ignore_scanning_scheme(start_angle, stop_angle, scan_continue, skipping)
                scanning_angle.reverse()

            for i in range(len(scanning_angle)):
                # if args.data:
                    p.control_transducer(
                        0,  # reserved
                        p._gain_setting,
                        scanning_angle[i],
                        p._transmit_duration,
                        p._sample_period,
                        p._transmit_frequency,
                        p._number_of_samples,
                        1,
                        0
                    )
                    p.wait_message([definitions.PING360_DEVICE_DATA], 0.5)
                    new_message = [int(j) for j in p._data]
                    fileObject.write(str(scanning_angle[i])+" ")
                    for j in range(len(new_message)):
                        fileObject.write(str(new_message[j])+" ")
                    fileObject.write("\n")
                #if len(new_message)>0:
                #    sonar_img[:,int(x/step)]=new_message
            fileObject.close()
            count = count + 1

    else:
        local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        # file_save=open(local_time+"_"+path_save_data,"w")
        start_time = time.time()
        x_angle = start_angle
        angle_former = (start_angle - 1) % 400
        count = 0
        start = True
        num_rescan = 3
        object_former = []
        object_record = {}
        peaks_record = [[[], []]] * 400
        sonar_img = np.zeros((number_samples, 400))
        while time.time() < start_time + duration:
            if start:
                file_save = open("mode_"+str(args.mode)+"/"+local_time + "_" + str(count) + "_" + path_save_data, "w")
                sonar_img = np.zeros((number_samples, int(400 / scan_speed) + 1))
                object_record = {}
                peaks_record = [[]] * 400
                local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                start = False
            if x_angle > stop_angle:
                x_angle = x_angle - stop_angle - 1 + start_angle
            else:
                p.control_transducer(
                    0,
                    p._gain_setting,
                    x_angle,
                    p._transmit_duration,
                    p._sample_period,
                    p._transmit_frequency,
                    p._number_of_samples,
                    1,
                    0,
                )
                p.wait_message([definitions.PING360_DEVICE_DATA], 0.5)
                data = [j for j in p._data]
                if len(data) == 0:
                    x_angle += 1
                    print("wrong angle")
                else:
                    if args.b:
                        data = abs(data - sonar_image_ref[:, x_angle])
                    file_save.write(str(x_angle) + " ")
                    for j in range(len(data)):
                        file_save.write(str(data[j]) + " ")
                    file_save.write("\n")
                    if len(data) > 0:
                        sonar_img[:, int(x_angle / scan_speed)] = data
                    len_sample = dis / number_samples
                    data_filter = smooth(data, len_sample, 0)
                    local_var = smooth(abs(data - data_filter), len_sample, 1)
                    peaks, dict_re = detect(data_filter, len_sample, local_var)
                    new_object, overlap = update_record(peaks_record, object_record, dict_re, x_angle, angle_former,
                                                        len_sample)
                    sonar_img[:, x_angle] = data
                    rmax = 0
                    angle_add = fast_scan_speed
                    for o in object_former:
                        if o[1] not in overlap:
                            if filter(object_former[o], threshold):
                                if object_former[o][4] > rmax:
                                    rmax = object_former[o][4]
                                    r = object_former[o]
                        else:
                            angle_add = scan_speed
                    if rmax != 0:
                        # print(r)
                        images = sonar_img[:, r[1]: r[2] + 1, np.newaxis]
                        images = rescan(r, dis, number_samples, num_rescan, images)
                        info = str(r[1]) + '-' + str(r[2]) + '-' + str(int(r[3] / len_sample)) + '-' + str(
                            int(r[4] / len_sample)) + '_'
                        local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                        np.save("mode_"+str(args.mode)+"/"+"rescan_" + info + local_time + '.npy', images)
                        # images = test_transform(resize(images, (51, 11, 3)))
                        # return to former setting
                        sample_period = cal_sample_period(dis, number_samples)
                        sample_period = round(sample_period)
                        transmit_duration = adjustTransmitDuration(dis, sample_period)
                        p.set_sample_period(sample_period)
                        p.set_number_of_samples(number_samples)
                        p.set_transmit_duration(transmit_duration)
                        # do classification
                        # with torch.no_grad():
                        #     images = torch.unsqueeze(images, 0)
                        #     output = net(images.to(device, dtype=torch.float))
                        #     print(output.data)
                    angle_former = x_angle
                    object_former = new_object
                    for i in range(1, angle_add):
                        sonar_img[:, (x_angle + i) % 400] = data

                    x_angle += angle_add
                if x_angle > stop_angle:
                    start = True
                    count += 1
                    file_save.close()
                    if args.display:
                        show_sonar(sonar_img, dis)
                        plt.savefig("mode_"+str(args.mode)+"/"+local_time + "_" + str(count) +"figure.png")
                        plt.close()
