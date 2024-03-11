import pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import configure_camera,calculate_Reward1,calculate_Reward2
from parameters import WulongTea_HSV,Rice_HSV
import os
def nothing(x):
    pass

def main():
    Mid_pipeline = configure_camera('317622074564')

    cv2.namedWindow('track_bar')
    save_video = True
    tune = False
    # create trackbars for color change
    if tune:
        cv2.createTrackbar('H_low','track_bar',0,255,nothing)
        cv2.createTrackbar('H_high','track_bar',0,255,nothing)
        cv2.createTrackbar('S_low','track_bar',0,255,nothing)
        cv2.createTrackbar('S_high','track_bar',0,255,nothing)
        cv2.createTrackbar('V_low','track_bar',0,255,nothing)
        cv2.createTrackbar('V_high','track_bar',0,255,nothing)
    if save_video:
        fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
        output  = cv2.VideoWriter(
            "videos/test.mp4",
            fourcc,
            30,
            (640,480),
            True)
    last_state_value = -1
    last_h = 0
    rewards = []
    raw_rewards = []
    while(1):
        frames = Mid_pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_data = np.asanyarray(color_frame.get_data())
        start_x = 240
        start_y = 20
        end_x = 460
        end_y = 380
        Image_HSV = color_data[start_y:end_y,start_x:end_x,:]
        Image_HSV = cv2.cvtColor(Image_HSV,cv2.COLOR_BGR2HSV)
        if tune:
            H_low = cv2.getTrackbarPos('H_low','track_bar')
            H_high = cv2.getTrackbarPos('H_high','track_bar')
            S_low = cv2.getTrackbarPos('S_low','track_bar')
            S_high = cv2.getTrackbarPos('S_high','track_bar')
            V_low = cv2.getTrackbarPos('V_low','track_bar')
            V_high = cv2.getTrackbarPos('V_high','track_bar')
        else:
            H_low,H_high,S_low,S_high,V_low,V_high = Rice_HSV()

        lower = np.array([H_low,S_low,V_low])
        upper = np.array([H_high,S_high,V_high])

        mask = cv2.inRange(Image_HSV,lower,upper)
        
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)

        # kernel_2 = np.ones((3, 3), dtype=np.uint8) # 卷积核变为4*4
        # mask = cv2.dilate(mask, kernel_2,iterations = 1)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours)>0:
            area = [cv2.contourArea(contours[i]) for i in range(len(contours)) ]
            index = np.argmax(area)
            x,y,w,h = cv2.boundingRect(contours[index])
            cv2.rectangle(color_data,(x+start_x,y+start_y),(x+start_x+w,y+start_y+h),(0,0,255),2)
            h = np.max([h,last_h])
            print(h)
            last_h = h
            state_value = -np.abs((h-160)/160)
            raw_reward = state_value-last_state_value
            raw_rewards.append(raw_reward)
            reward = calculate_Reward2 (raw_rewards)
            rewards.append(reward)
            
            color_data = cv2.putText(color_data, f"Reward:{reward},state{round(state_value,2)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            last_state_value = state_value
        if color_frame is None:
            continue
        if (save_video):
            output.write(color_data)
        cv2.imshow("camera",color_data)
        cv2.imshow("mask",mask)
        if cv2.waitKey(1) == ord("q"):
            if save_video:
                plt.plot(rewards)
                plt.title(f'Reward Plot(Return={np.sum(rewards)})')
                plt.xlabel('Time Steps')
                plt.ylabel('Reward')
                plt.savefig(f'videos/reward_curve(Return={np.sum(rewards)}).png')
                plt.show()
                os.rename("videos/test.mp4",f'Return={np.sum(rewards)}.mp4')
            Mid_pipeline.stop() 
            break




    # 停止摄像头并关闭窗口
    

if __name__ == '__main__':

    main()


