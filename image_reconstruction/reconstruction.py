import os
import argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

np.set_printoptions(threshold=np.inf)

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


def read_txt(path, angle_range=400):
    ''' read sonar data from txt file
    Args:
        path: path of txt file
        angle_range: range of angle (1 gradian = 0.9 degree), default: 400
    Returns:
        sonar_data: 2d array, shape: (angle_range, 500)
        start_angle: start angle of sonar data
        end_angle: end angle of sonar data
    '''
    sonar_data = np.zeros((angle_range, 500))
    with open(path, 'r') as f:
        lines = f.readlines()
        lines.sort(key = lambda x : float(x.split(" ")[0]))
        
        start_angle = float(lines[0].split(' ')[0])
        end_angle = float(lines[-1].split(' ')[0])
        for line in lines:
            angle, data = readline(line)
            if len(data) == 500:
                sonar_data[int(angle)] = data
                # print(angle)
                # print(sonar_data[int(angle)])
            # print(sonar_data.shape)
    return sonar_data, int(start_angle), int(end_angle)

def read_mse_file(file_path):
    """
    Reads a file containing MSE values and converts them to a list of floats.

    Args:
        file_path (str): Path to the file containing MSE values.

    Returns:
        list: A list of MSE values as floats.
    """
    # Open the file for reading
    with open(file_path, 'r') as f:
        mse = []
        lines = f.readlines()

        # Convert each line to a float and append to the list
        for line in lines:
            mse.append(np.float64(line))

    return mse
    
def __skip_scanning__(image, skip, start_angle, end_angle, start, step):
    """
    Performs skip scanning on the given image data.

    Args:
        image (ndarray): The input image data to be processed.
        skip (int): Number of angles to skip during scanning.
        start_angle (int): Starting angle of the image.
        end_angle (int): Ending angle of the image.
        start (int): Initial scanning start index.
        step (int): Step size for scanning.

    Returns:
        tuple: The modified image and a list of skipped angles.
    """
    # Adjust the start index to ensure it is within the valid range
    while start < start_angle:
        start += step

    # Initialize a list to store skipped angles
    skip_angle = []

    # Perform skip scanning
    for i in range(start, end_angle, step + skip):
        for j in range(skip):
            try:
                # Set skipped rows in the image to 0
                #image[i + j, :] = 0 #test version, can remove
                skip_angle.append(i + j) 
            except IndexError:
                # Safely handle out-of-bounds access
                continue

    return image, skip_angle


def read_data(filename, skip, start, step):
    """
       Reads sonar data from a file and processes it based on the given parameters.

       Args:
           filename (str): Path to the input file.
           skip (int): Skipping step for data scanning.
           start (int): Starting index for scanning.
           step (int): Step size for scanning.

       Returns:
           tuple: Processed data including skip_data, ground_truth, info, scheme, start_angle, and end_angle.
       """
    # Read sonar data and angles
    sonar_data, start_angle, end_angle = read_txt(filename)
    # end_angle = start_angle + 199
    # if end_angle > 399:
    #     end_angle = 399

    # Initialize ground truth array
    sonar_gt = np.zeros((len(sonar_data), len(sonar_data[0]))) #useless now. this is for the test
    #for i in range(len(sonar_gt)):
    #    for j in range(len(sonar_gt[0])):
    #        sonar_gt[i][j] = np.float64(sonar_data[i][j])

    # Perform scanning with skipping
    data_skip, info_total = __skip_scanning__(sonar_data, skip, start_angle, end_angle, start + start_angle, step) #obtain the skipped angle

    # Initialize containers for results
    skip_data = []
    ground_truth = []
    info = []
    scheme = []

    # Define the scanning object parameters
    obj = [start_angle, end_angle, 0, 500]
    polar_data = sonar_data[obj[0]:obj[1], obj[2]:obj[3]]
    sonar_gt = sonar_gt[obj[0]:obj[1], obj[2]:obj[3]]

    # Append processed data to respective lists
    skip_data.append(polar_data)
    ground_truth.append(sonar_gt)

    # Process the info data
    info_temp = []
    for i in range(len(info_total)):
        if obj[0] < info_total[i] < obj[1]:
            info_temp.append(info_total[i])
    info.append(info_temp)

    # Define the scanning scheme
    scheme_temp = [skip, obj[0], step]
    scheme.append(scheme_temp)

    return skip_data, ground_truth, info, scheme, start_angle, end_angle


def obtain_kernal(data, flag, idx, kernel_size, x, y):
    """
    Extracts a kernel (sub-matrix) from the given data and flag matrices based on the index and kernel size.

    Args:
        data (ndarray): The input data matrix.
        flag (ndarray): The input flag matrix.
        idx (tuple): The center index (row, column) for the kernel.
        kernel_size (int): The radius of the kernel (half the size).
        x (tuple): The range of valid row indices (min, max).
        y (tuple): The range of valid column indices (min, max).

    Returns:
        tuple: The extracted kernel from `data`, the corresponding flag values,
               the start indices, and the end indices of the kernel.
    """
    # Calculate the start and end indices for the kernel
    start_id = [idx[0] - kernel_size, idx[1] - kernel_size]
    end_id = [idx[0] + kernel_size, idx[1] + kernel_size]

    # Ensure indices are within the valid range for rows
    if idx[0] - kernel_size < x[0]:
        start_id[0] = x[0]
    if idx[0] + kernel_size > x[1]:
        end_id[0] = x[1]

    # Ensure indices are within the valid range for columns
    if idx[1] - kernel_size < y[0]:
        start_id[1] = y[0]
    if idx[1] + kernel_size > y[1]:
        end_id[1] = y[1]

    # Extract the kernel and the corresponding flag sub-matrix
    seg_k = data[start_id[0]:end_id[0] + 1, start_id[1]:end_id[1] + 1]
    seg_f = flag[start_id[0]:end_id[0] + 1, start_id[1]:end_id[1] + 1]

    return seg_k, seg_f, start_id, end_id


def image_reconstrution(image, info, scheme, mode, print_flag=False):
    """
    Reconstructs an image based on provided info, scheme, and mode.

    Args:
        image (ndarray): The input image to be reconstructed.
        info (list): List of indices containing information about skipped rows.
        scheme (list): Scanning scheme containing [skip, start].
        mode (str): Reconstruction mode.
        print_flag (bool): If True, prints detailed debug information.

    Returns:
        ndarray: The reconstructed image.
    """
    # Extract the scanning scheme parameters
    skip = scheme[0]
    start = scheme[1]

    # Initialize the recovered image and kernel size
    image_recover = np.zeros((len(image), len(image[0])))
    kernel_size = skip

    # Initialize the flag array
    flag = np.zeros((len(image), len(image[0])))
    for i in range(len(info)):
        # Mark rows specified in the info list
        flag[info[i] - start, :] = 1

    # Define the valid range for row and column indices
    x = [0, len(image) - 1]
    y = [0, len(image[0]) - 1]

    # Perform image reconstruction
    for i in range(len(image)):
        for j in range(len(image[i])):
            if i + start in info:
                # Obtain the kernel and flag sub-matrix
                data_k, flag_k, s_id, e_id = obtain_kernal(image, flag, [i, j], kernel_size, x, y)

                # Reconstruct the pixel value using the kernel
                image_recover[i][j] = add_pixel_value(
                    data_k, flag_k, i - s_id[0], j - s_id[1], kernel_size, mode, print_flag
                )
            else:
                # Retain the original pixel value if not in the info list
                image_recover[i][j] = image[i][j]

    return image_recover


def add_pixel_value(data, flag, i_id, j_id, k, mode="kernel", print_flag=False):
    """
    Reconstructs the pixel value at a given position using surrounding data and flags.

    Args:
        data (ndarray): The input data matrix.
        flag (ndarray): The flag matrix indicating valid/invalid pixels.
        i_id (int): Row index of the pixel to reconstruct.
        j_id (int): Column index of the pixel to reconstruct.
        k (int): Kernel size for reconstruction.
        mode (str): Reconstruction mode ("kernel" or "linear").
        print_flag (bool): If True, prints debug information.

    Returns:
        float: The reconstructed pixel value.
    """
    i_center, j_center = i_id, j_id
    angle = [i_center, i_center]

    # Determine the vertical bounds for reconstruction
    for i in range(1, i_center + 1):
        if i_center - i < 0:
            break
        if flag[i_center - i][j_center] == 0:
            angle[0] = i_center - i
            break

    for i in range(i_center + 1, len(data)):
        if flag[i][j_center] == 0:
            angle[1] = i
            break

    # Determine diagonal bounds for reconstruction (top-left to bottom-right)
    angle_ij = [[i_center, j_center], [i_center, j_center]]
    for idx in range(1, k + 1):
        if i_center - idx < 0 or j_center - idx < 0:
            break
        if flag[i_center - idx][j_center - idx] == 0:
            angle_ij[0] = [i_center - idx, j_center - idx]
            break

    for idx in range(1, k + 1):
        if i_center + idx >= len(data) or j_center + idx >= len(data[0]):
            break
        if flag[i_center + idx][j_center + idx] == 0:
            angle_ij[1] = [i_center + idx, j_center + idx]
            break

    # Determine diagonal bounds for reconstruction (top-right to bottom-left)
    angle_ji = [[i_center, j_center], [i_center, j_center]]
    for idx in range(1, k + 1):
        if i_center - idx < 0 or j_center + idx >= len(data[0]):
            break
        if flag[i_center - idx][j_center + idx] == 0:
            angle_ji[0] = [i_center - idx, j_center + idx]
            break

    for idx in range(1, k + 1):
        if i_center + idx >= len(data) or j_center - idx < 0:
            break
        if flag[i_center + idx][j_center - idx] == 0:
            angle_ji[1] = [i_center + idx, j_center - idx]
            break

    # Reconstruct the pixel value based on the mode
    if mode == "kernel":
        pixel_value = (
            (data[angle[1]][j_center] - data[angle[0]][j_center]) / (angle[1] - angle[0])
        ) * (i_center - angle[0]) + data[angle[0]][j_center]

        if angle_ij[1][0] - angle_ij[0][0] != 0:
            pixel_value_ij = (
                (data[angle_ij[1][0]][angle_ij[1][1]] - data[angle_ij[0][0]][angle_ij[0][1]])
                / (angle_ij[1][0] - angle_ij[0][0])
            ) * (i_center - angle_ij[0][0]) + data[angle_ij[0][0]][angle_ij[0][1]]
        else:
            pixel_value_ij = 0.0

        if angle_ji[1][0] - angle_ji[0][0] != 0:
            pixel_value_ji = (
                (data[angle_ji[1][0]][angle_ji[1][1]] - data[angle_ji[0][0]][angle_ji[0][1]])
                / (angle_ji[1][0] - angle_ji[0][0])
            ) * (i_center - angle_ji[0][0]) + data[angle_ji[0][0]][angle_ji[0][1]]
        else:
            pixel_value_ji = 0.0

        # Average the values from different directions
        pixel = (pixel_value + pixel_value_ij + pixel_value_ji) / 3

        if print_flag:
            print(i_center, j_center, angle, angle_ij, angle_ji)
            print(flag)

    elif mode == "linear":
        pixel_value = (
            (data[angle[1]][j_center] - data[angle[0]][j_center]) / (angle[1] - angle[0])
        ) * (i_center - angle[0]) + data[angle[0]][j_center]
        pixel = pixel_value

    return pixel


def metric(gt, re_im, info, start):
    """
    Computes the average absolute error between the ground truth and the reconstructed image.

    Args:
        gt (ndarray): Ground truth image.
        re_im (ndarray): Reconstructed image.
        info (list): List of row indices to compare.
        start (int): Starting index for comparison.

    Returns:
        float: The average absolute error.
    """
    count = 0
    Error = 0.0

    # Calculate total error and count of compared elements
    for i in range(len(info)):
        for j in range(len(re_im[info[i] - start])):
            Error += np.abs(re_im[info[i] - start][j] - gt[info[i] - start][j])
            count += 1

    # Compute average error
    Error /= count
    return Error


def dir_create(dir):
    """
    Creates a directory if it does not exist.

    Args:
        dir (str): Path of the directory to create.

    Returns:
        None
    """
    if not os.path.exists(dir):
        os.mkdir(dir)


def skip_scan_recover(filename, label_path, skip, start, step, save_gt, save_data, save_recover, mode="kernel", print_flag=False):
    """
    Performs image reconstruction with skip-scan recovery and computes MSE.

    Args:
        filename (str): Path to the input data file.
        label_path (str): Path to the ground truth label file.
        skip (int): Number of angles to skip during scanning.
        start (int): Starting index for reconstruction.
        step (int): Step size for reconstruction.
        save_gt (str): Path to save the ground truth images.
        save_data (str): Path to save the input data images.
        save_recover (str): Path to save the reconstructed images.
        mode (str): Reconstruction mode ("kernel" or "linear").
        print_flag (bool): If True, enables debug printing.

    Returns:
        list: List of MSE values for each reconstructed image.
    """
    # Read data, ground truth, info, and scanning scheme
    data, gt, info, scheme = read_data(filename, label_path, skip, start, step)
    mse = []

    # Process each data frame
    for i in range(len(data)):
        # Perform image reconstruction
        data_recover = image_reconstrution(data[i], info[i], scheme[i], mode, print_flag)

        # Save the ground truth, input data, and reconstructed images
        save_gt_obj = f"{save_gt}_{i}.png"
        save_data_obj = f"{save_data}_{i}.png"
        save_recover_obj = f"{save_recover}_{i}.png"
        cv2.imwrite(save_gt_obj, gt[i])
        cv2.imwrite(save_data_obj, data[i])
        cv2.imwrite(save_recover_obj, data_recover)

        # Compute MSE for the reconstructed image
        mse.append(metric(gt[i], data_recover, info[i], scheme[i][1]))

    return mse


def obtain_reconstruction_mse(filename, label_path, mode, skip, start, step, dis, state):
    """
    Computes the Mean Squared Error (MSE) for image reconstruction across scenarios and saves the results.

    Args:
        filename (str): Path to the raw data.
        label_path (str): Path to the ground truth labels.
        mode (list): List of reconstruction modes (e.g., "kernel", "linear").
        skip (int): Number of angles to skip during scanning.
        start (int): Starting index for reconstruction.
        step (int): Step size for reconstruction.
        dis (str): Distance parameter for saving directories.
        state (str): State identifier for saving directories.

    Returns:
        tuple: Arrays of MSE values for kernel and linear modes.
    """
    mse_linear = []
    mse_kernel = []
    scenarios = os.listdir(label_path)

    # Create directory structure for saving results
    dir = f"./image/{state}"
    dir_create(dir)
    dir = f"{dir}/{dis}"
    dir_create(dir)
    dir = f"{dir}/{skip}_{start}_{step}"
    dir_create(dir)

    # Process each mode and scenario
    for idx in range(len(mode)):
        for scenario in tqdm(scenarios):
            scenario = str(scenario)
            if scenario == ".DS_Store":
                continue

            # Define paths for the current scenario
            data_scenario_path = os.path.join(filename, scenario)
            label_scenario_path = os.path.join(label_path, scenario)

            if not os.path.exists(data_scenario_path):
                print(f"Raw data path {data_scenario_path} does not exist.")
                continue

            dir_s = f"{dir}/{scenario}"
            dir_create(dir_s)

            sonars = os.listdir(label_scenario_path)
            for sonar in sonars:
                if sonar == ".DS_Store":
                    continue

                # Define paths for the current sonar
                data_dir_path = os.path.join(data_scenario_path, sonar)
                label_dir_path = os.path.join(label_scenario_path, sonar)
                assert os.path.exists(data_dir_path), f"Sonar path {data_dir_path} does not exist."

                files = os.listdir(label_dir_path)
                if len(files) == 0:
                    continue

                # Sort files based on naming convention
                if '_' in files[0]:
                    files.sort(key=lambda x: int(x.split('_')[1]))
                else:
                    files.sort(key=lambda x: int(x.split('.')[0]))

                # Create directories for saving results
                gt_dir = f"{dir_s}/GT"
                data_dir = f"{dir_s}/data"
                recover_dir = f"{dir_s}/recover"
                dir_create(gt_dir)
                dir_create(data_dir)
                dir_create(recover_dir)

                for file in files:
                    if file == ".DS_Store":
                        continue

                    # Define paths for the current file
                    data_file_path = os.path.join(data_dir_path, file)
                    label_file_path = os.path.join(label_dir_path, file)
                    file_name = f"{sonar}_{file[:-4]}_{mode[idx]}_{skip}_{start}_{step}"
                    save_gt = f"{gt_dir}/{file_name}"
                    save_data = f"{data_dir}/{file_name}"
                    save_recover = f"{recover_dir}/{file_name}"

                    # Perform reconstruction and compute MSE
                    print_flag = False
                    mse = skip_scan_recover(
                        data_file_path, label_file_path, skip, start, step,
                        save_gt, save_data, save_recover, mode[idx], print_flag
                    )

                    # Append MSE values to the corresponding list
                    for i in range(len(mse)):
                        if idx == 0:
                            mse_kernel.append(mse[i])
                        else:
                            mse_linear.append(mse[i])

    mse_kernel = np.array(mse_kernel)
    mse_linear = np.array(mse_linear)
    return mse_kernel, mse_linear


def plot_error_bar(mse, labels, name="", save=False):
    """
    Plots an error bar chart for MSE values.

    Args:
        mse (list): List of MSE values for each label.
        labels (list): List of labels for the x-axis.
        name (str): Filename to save the plot.
        save (bool): If True, saves the plot to a file; otherwise, displays it.

    Returns:
        None
    """
    Mean_list = []
    Std_list = []
    color_list = []

    # Compute mean and standard deviation for each MSE list
    for i in range(len(mse)):
        if i % 4 == 0 or i % 4 == 1:
            color_list.append('blue')
        else:
            color_list.append('red')
        Mean_list.append(np.mean(mse[i]))
        Std_list.append(np.std(mse[i]))

    # Plot the error bar chart
    x_pos = np.arange(len(labels))
    fig, ax = plt.subplots()
    ax.bar(
        x_pos, Mean_list, yerr=Std_list, color=color_list, align='center',
        alpha=0.5, ecolor='black', capsize=10
    )
    ax.set_ylabel('MSE')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_title("Sonar Image Recovery Error")
    ax.yaxis.grid(True)

    # Save or display the plot
    plt.tight_layout()
    if save:
        plt.savefig(name)
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="difference")
    parser.add_argument("--raw", default='11111', type=str, help="path of raw picture")
    parser.add_argument("--save_txt", default='txt/', type=str, help="name of saving txt")
    parser.add_argument("--save_img", default='img/', type=str, help="name of saving picture")
    parser.add_argument("--skip", default= 2, type = int, help="skip angle")
    parser.add_argument("--offset", default = 1, type = int, help = "offset")
    parser.add_argument("--scan", default = 1, type = int, help = "scan angle")
    args = parser.parse_args()

    raw_data_root = "raw_data/"
    raw_data_path = raw_data_root + args.raw + '/'

    recovered_data_root = "recover/"
    # Recovered data (text files)
    recover_txt_path = recovered_data_root + args.save_txt
    # Recovered data (images)
    recover_img_path = recovered_data_root + args.save_img

    # List all sonar directories
    sonars = os.listdir(raw_data_path)
    for sonar in sonars:
        if sonar != '.DS_Store':
            # Construct paths for each sonar
            sonarpath = raw_data_path + sonar + '/'
            files = os.listdir(sonarpath)
            files.sort()

            # Paths to save recovered text and images
            save_txt_path = recover_txt_path + args.raw + '/' +sonar + '/'
            save_img_path = recover_img_path + args.raw + '/' + sonar + '/'
            # print(save_txt_path)
            # print(save_img_path)
            # Create directories if they do not already exist
            if not os.path.exists(save_txt_path):
                os.makedirs(save_txt_path)
            if not os.path.exists(save_img_path):
                os.makedirs(save_img_path)

            count = 0

            # Process each file in the sonar directory
            for file in tqdm(files):
                if file != '.DS_Store':
                    filename = sonarpath + file

                    # Modify the following settings to match your sonar configuration
                    
                    data, ground_truth, info, scheme, start_angle, end_angle = read_data(filename, args.skip, args.offset, args.scan)

                    # Iterate over all data slices
                    for i in range(len(data)):
                        mode = 'kernel'

                        # Reconstruction
                        data_recover = image_reconstrution(data[i], info[i], scheme[i], mode, print_flag=False)

                        # Paths for saving recovered data
                        save_txt_recover = save_txt_path + file
                        save_img_recover = save_img_path + file[:-4] + '.png'

                        # Add padding to match the desired format
                        padding_1 = np.zeros((int(start_angle), 500))
                        padding_2 = np.zeros((400 - int(end_angle), 500))
                        entire_data = np.vstack((padding_1, data_recover))
                        entire_data = np.vstack((entire_data, padding_2))
                        angles = np.arange(400)
                        entire_data = np.insert(entire_data, 0, values=angles, axis=1)

                        # Save the reconstructed data
                        np.savetxt(save_txt_recover, entire_data, fmt='%.1f', delimiter=' ')
                        cv2.imwrite(save_img_recover, entire_data)

                count += 1


if __name__=='__main__':
    main()
    
