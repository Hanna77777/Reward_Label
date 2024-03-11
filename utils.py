import pyrealsense2 as rs
import numpy as np
import cv2
def calculate_Reward1(raw_rewards):
    gamma = 0.7
    if len(raw_rewards)==1:
        return raw_rewards[-1]
    if raw_rewards[-1] == 0:
        reward = gamma*raw_rewards[-2]
    else:
        reward = raw_rewards[-1]
    return round(reward,2)
def calculate_Reward2(raw_rewards):
    length = len(raw_rewards)
    horizon = 7
    positive = [i>0 for i in raw_rewards[-horizon:]]
    negative = [i<0 for i in raw_rewards[-horizon:]]    
    if length<horizon:
        reward = 0
    else:
        if np.sum(positive)>0:
            reward = 1
        if np.sum(negative)>0:
            reward = -1
        if np.sum(positive)==0 and np.sum(negative) == 0:
            reward = 0
    return reward
def configure_camera(serial_number):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial_number)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    return pipeline