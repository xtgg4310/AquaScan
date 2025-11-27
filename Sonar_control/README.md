# Ping360 Sonar Control Script

## Overview
This program controls the Ping360 sonar device, supporting multiple scanning modes, data acquisition, and visualization capabilities. The code offers flexible parameter configurations to adapt to various scanning strategies and data storage requirements.

---

## Key Features
- **Multi-mode Scanning**  
  - `Mode 0`: Time-controlled scanning (fixed duration)  
  - `Mode 1`: Count-controlled scanning (fixed cycles)  
  - `Mode 2`: Tracking scan with skipping pattern  
  - `Mode 3`: Back-and-forth scanning  
  - `Mode 4`: Dynamic directional tracking scan  
- **Data Storage**: Raw data saved as `.txt` files organized by batch and device ID.  
- **Visualization**: Generates polar and Cartesian (XY) sonar images.  
- **Background Noise Cancellation**: Enable via `--b` parameter.  
- **Flexible Configuration**: Customizable scan angles, sound speed, sample count, frequency, etc.

---

## Packages Used

### Core Packages
- `brping` - Interface for Ping360 sonar communication
  - `Ping360` - Main sonar control class
  - `definitions` - Protocol definitions
- `numpy` (as `np`) - Numerical operations and array handling
- `time` - Time-related functions
- `argparse` - Command-line argument parsing
- `json` - JSON data handling
- `sys` - System-specific parameters and functions
- `os` - Operating system interface

### Data Processing & Visualization
- `matplotlib` - Plotting and visualization
  - `pyplot` (as `plt`) - MATLAB-like plotting interface
- `sonar_display` - Custom sonar visualization module
- `Track_scan` - Custom scanning schemes module
- `data` - Custom data handling module

---

## Usage Guide

### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--mode` | int | 3 | Scanning mode (0-4) |
| `--udp` | str | `192.168.1.13:12345` | UDP connection (host:port) |
| `--speed` | int | 1500 | Sound speed in water (m/s) |
| `--dis` | int | 15 | Scanning distance (m) |
| `--trans_freq` | int | 750 | Sonar transmit frequency (kHz) |
| `--number_samples` | int | 500 | Samples per angle |
| `--time` | int | 120 | Scan duration in seconds (Mode 0 only) |
| `--count` | int | 3 | Total scan cycles (Modes 1-4) |
| `--step` | int | 3 | Angular step size |
| `--start` | int | 100 | Start angle (360° mapped to 0 -> 400) |
| `--end` | int | 200 | End angle (360° mapped to 0 -> 400) |
| `--D` | int | 0 | Execution time-day |
| `--H` | int | 0 | Execution time-hour |
| `--M` | int | 0 | Execution time-minute |
| `--save` | str | `mode_` | Root directory for data storage |
| `--polar_pic_path` | str | `/path/polar/` | Path for polar coordinate images |
| `--xy_pic_path` | str | `/path/xy/` | Path for Cartesian coordinate images |
| `--sonar_num` | str | `13` | Device ID (for data organization) |
| `--batch_id` | str | `12030001` | Batch ID (for data organization) |

### Example

**Mode 3 (Back-and-forth scanning)**  
   ```bash
   python ping360_control_v4.py --mode 3 --count 5 --sonar_num 11 --batch_id 20231001 --D --H 10 --start 0 --end 399 --step 3 --dis 25 --udp 192.168.1.11:12345 --M 17
   ```

---

## Data Storage Structure
```
mode_<mode_number>/
└── <batch_id>/
    └── sonar<device_number>/
        ├── 2023-10-01-12-30-00_0.txt  # Raw data files
        ├── ...
        └── figures/                   # Image files (if enabled)
```

---

## Notes
1. **Hardware Connection**  
   - Ensure proper connection with sonar.  
   - Modify parameters to use correct ports.  

2. **Path Permissions**  
   - Ensure write permissions for output paths (e.g., `--polar_pic_path`).   

3. **Background Noise File**  
   - Background file must be pre-recorded `.txt` with matching sample count.  

---
