import numpy as np
import os

def naive_scanning_scheme(start_angle, end_angle, skipping):
    if start_angle == -1:
        start=0
    if end_angle == -1:
        end=399
    scanning_range=np.arange(start,end,skipping+1)
    scanning_range.astype(int)
    return scanning_range

def back_forth_scanning_scheme(start_angle, end_angle, step,count):
    if start_angle == -1:
        start_angle = 0
    if end_angle == -1:
        start_angle= 399
    if count%2 ==0:
        scanning_range=np.arange(start_angle, end_angle+1, step)
    else:
        scanning_range = np.arange(end_angle, start_angle-1, -step)
    scanning_range.astype(int)
    return scanning_range


def ignore_scanning_scheme(start_angle, end_angle, scan_continue, skipping):
    scanning_range = []
    current_value = start_angle
    for i in np.arange(start_angle, end_angle + 1, 1):
        current_value = i
        if (i - start_angle) % (scan_continue + skipping) in range(scan_continue):
            scanning_range.append(i)
        else:
            continue
    return scanning_range
