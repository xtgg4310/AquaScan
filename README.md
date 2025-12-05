# Mobicom2025: AquaScan: A Sonar-based Underwater Sensing System for Human Activity Monitoring

## This repo contains the code, deployment instructions, and detailed usage of each function. If you find our repo is useful, please cite our paper.

## Contact information
If you have any questions, please contact the author through the email: 1155161507@link.cuhk.edu.hk (CUHK E-mail). 626201515@qq.com (Personal E-mail).

## Installation and Preparation

### Hardware dependencies

Computing platform: A server with a CPU and GPU. Our code is tested on a computer with a 7950x3D CPU, 4090 24GB GPU, and 64GB RAM.

Sensor Node:
* Control Unit: Raspberry Pi 4B model 8GB RAM
* Ping360 Scanning Sonar: https://bluerobotics.com/store/sonars/imaging-sonars/ping360-sonar-r1-rp/

Sonar scanning principle:
Sonar detects human motion by receiving the reflections from the human body. Considering that a human stands in front of the sonar, the human's limbs and body show the main reflections. Due to the inevitable motion, the real sonar echoes will generate a larger cluster than humans. A single sonar image is a 2D BEV image


### Software dependencies
For common Python packages such as numpy, matplotlib, you can download them through pip or conda install these packages

Some important package versions:
* numpy version: 1.24.3
* scikit-image: 0.20.0
* scikit-learn: 1.3.0
* matplotlib: 3.7.1
* PyTorch: 2.1.0
* CUDA Version: 12.1
* Opencv-python: 4.8.1.78

To control scanning sonar (Ping 360) to collect data by yourself, you should install the brping packages by running the command below:
```bash
pip install --user bluerobotics-ping --upgrade
```
### Deployment Instruction
To deploy Ping360 Sonar in the pool, we use a stand shown in the figure below to fix the position of the sonars.

![image](https://github.com/xtgg4310/AquaScan_Artifact/blob/main/figure/setup-2-2.jpg)

## Detail of each function
This section is to introduce the function. If you want to run the data, please make sure that all parameters are configured. The function also leaves the areas to be manually configured for the parameters. You can check the python files.

### Sonar Control
Please see the README.md in Sonar_control folder.

### Image reconstruction
Please see the README.md in image_reconstruction folder.

### Object detection
We have three codes:
* pre_sonar_bias.py: remove the bias caused by mechanical rotation of sonar (seldom happens)
* pre_sonar.py: denoise sonar images and detect objects on the sonar image.
* pre_sonar_opt: optimize the dynamic object detection with binary search. This function is suitable for shallow pools and noisy environments.

Denoising and object detection contain both static noise removal (removing noise caused by poolsides and wall reflection, which may be localized outside the wall).

Before using the function for your collected data, you should configure the data_remove and remove_line in the pre_sonar(_opt). Data_remove means removing the areas of the background signals. Remove_line means removing the swimming lanes.

#### Usage Guide
#### Parameters of pre_sonar_opt.py
| Argument Name | Argument Type | Required | Help Information |
| --- | --- | --- | --- |
| `--pre` | int | Yes | preprocess type |
| `--data` | str | Yes | data_path |
| `--label` | str | Yes | label_path |
| `--label_type` | int | Yes | label type |
| `--parad` | int (multiple values) | Yes | para for noise remove at different distance |
| `--parap` | int (multiple values) | Yes | para for resizing sonar images |
| `--paral` | int (multiple values) | Yes | para for resizing labels |
| `--obj_detect` | str | Yes | save suffix of object detection metric |
| `--obj_type` | str | Yes | folders for object detection metric |
| `--save_dir_all` | str | Yes | the folder that results(non-numerical) saved |

#### Parameters of per_sonar.py
| Argument Name | Argument Type | Required | Help Information |
| --- | --- | --- | --- |
| `--pre` | int | Yes | preprocess type |
| `--data` | str | Yes | data_path |
| `--label` | str | Yes | label_path |
| `--label_type` | int | Yes | label type |
| `--parad` | int (multiple values) | Yes | para for noise remove at different distance |
| `--parap` | int (multiple values) | Yes | para for resizing sonar images |
| `--paral` | int (multiple values) | Yes | para for resizing labels |
| `--blur_size` | int (multiple values) | Yes | blur size for dynamic processing |
| `--human_size` | int | Yes | human size as the threshold for dynamic processing (calculated as the bbox size with a specific distance) |
| `--remove` | int | Yes | remove semic/static noise under different settings |
| `--bg_path` | str | Yes | background data path |
| `--bg_sc` | str | Yes | backgroud folder name |
| `--max_blur` | int | Yes | max blur size for image denosing |
| `--process` | int | Yes | whether the sonar image is processed by pre_sonar_bias.py |
| `--obj_detect` | str | Yes | save suffix of object detection metric |
| `--obj_type` | str | Yes | folders for object detection metric |
| `--save_dir_all` | str | Yes | the folder that results(non-numerical) saved |

#### Parameters of pre_sonar_bias.py
| Argument Name | Argument Type | Required | Help Information |
| --- | --- | --- | --- |
| `--data` | str | Yes | data_path |
| `--label` | str | Yes | label_path |

Before using pre_sonar_bias.py, please configure the path in the file.

### tracking
We have two codes for tracking the objects:
* label2dis.py --generate location of subject in the sonar images
* track.py --generate tracjectory of subjects

Before using label2dis.py and track.py, please set the configuration in each method. The state record in label2dis.py and track.py is used for reference, not the recognized activities; you can even remove it or set it to another mark.

#### Usage Guide
#### Parameters of label2dis.py
| Argument Name | Argument Type | Required | Help Information |
| --- | --- | --- | --- |
| `--data` | str | Yes | data_path |
| `--detect` | str | Yes | label_path |
| `--gt` | str | Yes | gt_path |
| `--type` | int | Yes | data_type |
| `--remove` | int | Yes | data_type |
| `--dis` | int | Yes | dis |
| `--parad` | int (multiple values) | Yes | parad , the same with pre_sonar.py |
| `--save_dir_all` | str | Yes | label_path |

#### Parameters of track.py
| Argument Name | Argument Type | Required | Help Information |
| --- | --- | --- | --- |
| `--data` | str | Yes | data_path |
| `--label` | str | Yes | label_path |
| `--track` | str | Yes | tracking |
| `--track_re` | str | Yes | track_results |
| `--cfg` | int (multiple values) | Yes | track threshold setting |
| `--save_dir_all` | str | Yes | save_dir |

### Moving detection
Moving_detect.py is used for movement detection. 

Before using moving_detect.py, please make sure that the threshold is set.

#### Usage Guide
#### Parameters of Moving_detect.py
| Argument Name | Argument Type | Required | Help Information |
| --- | --- | --- | --- |
| `--data` | str | Yes | data_path |
| `--save` | str | Yes | save_path |
| `--save_dir_all` | str | Yes | save_dir |
| `--pre_cfg` | float (multiple values) | Yes | pre_cfg, the time window use for moving trace calcualtion. |
| `--smooth_cfg` | float (multiple values) | Yes | smooth_cfg, the time window use for smooth results. |

### Generate inference data for motion detection
generate_data_all.py is used for generating inference data

#### Usage Guide
#### Parameters of generate_data_all.py
| Argument Name | Argument Type | Required | Help Information |
| --- | --- | --- | --- |
| `--data` | str | Yes | data_path |
| `--label` | str | Yes | label_path |
| `--save` | str | Yes | save_path |
| `--file` | str | Yes | file |
| `--save_dir_all` | str | Yes | save_dir |

### Recognizing activities through a state-transfer-machine
* infe_state.py: motion detection.
* split_results.py: record the motion detection results in separate files.
* state.py: recognize activities.

Before using state.py, please make sure the threshold is set.

#### Usage Guide
#### Parameters of infe_state.py & train.py
| Argument Name | Argument Type | Required | Choices | Default Value | Help Information |
| --- | --- | --- | --- | --- | --- |
| `--file_type` | str | No | `yaml`, `json` | `yaml` | None |
| `--option_path` | str | Yes | None | None | path of the option json file |

#### Parameters of split_results.py
| Argument Name | Argument Type | Required | Help Information |
| --- | --- | --- | --- |
| `--dir` | str | Yes | motion |
| `--save_dir` | str | Yes | split_results |
| `--save_dir_all` | str | Yes | moving |

#### Parameters of state.py
| Argument Name | Argument Type | Required | Help Information |
| --- | --- | --- | --- |
| `--motion` | str | Yes | motion |
| `--moving` | str | Yes | moving |
| `--har` | str | Yes | har |
| `--detect` | str | Yes | detect |
| `--label` | str | Yes | detect |
| `--har_cfg` | str | Yes | har_cfg, use for constraint of re-smoothing the first 2 states |
| `--smooth_cfg` | str | Yes | smooth_cfg, use for motion state smoothing. |
| `--start_cfg` | str | Yes | start_cfg, use for record the timestamp. |
| `--gt_cfg` | str | Yes | gt_cfg |
| `--gt_mode` | int | Yes | gt_mode |
| `--gt_trans` | int | Yes | gt_trans |
| `--gt_sc` | str (multiple values) | Yes | gt_sc |
| `--dis` | int | Yes | distance |
| `--label_type` | int | Yes | label_type |
| `--sample` | int | Yes | sample |
| `--save` | str | Yes | save |
| `--save_dir_all` | str | Yes | save_dir_all |

### Show the numerical results and plot the confusion matrix
run the code:
```bash
python cal_results.py
```
cal_results.py provide two metric: one is generated from denoise_metric.py, which is for object detection. It reflects the false and miss detection and IoU. Another is from state.py, which is the recognition of activities detected in sonar system.


