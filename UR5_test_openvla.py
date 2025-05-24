#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import cv2
import urx
import time
from colorama import Fore, Back, Style, init
import math3d as m3d
from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper
import requests
import json_numpy

# Initialize json_numpy for handling numpy arrays in requests
json_numpy.patch()

IP_ROBOT = "192.168.0.6"
TRANS_TCP = (0, 0, 0.19, 1.2092, -1.2092, 1.2092)
#TRANS_BASE = [0, 0, 0, -0.61394313, 1.48218982, 0.61394313]
TRANS_BASE = [0, 0, 0, 0, 0, 3.1415926]

# OpenVLA Server settings
OPENVLA_SERVER_URL = "http://localhost:8000/act"

class InferenceNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('inference_node', anonymous=True)
        print(Fore.YELLOW + "Initialization success!" + Style.RESET_ALL)
        # Initialize UR5 Robot
        self.robot = urx.Robot(IP_ROBOT, True)  # Replace with UR5's IP address
        self.robot.set_tcp(TRANS_TCP)  # Replace with TCP
        self.robot.set_payload(0.5, (0, 0, 0.1))  # Replace with payload
        self.robot.set_csys(m3d.Transform(TRANS_BASE))
        
        # confirm the coordinates
        # current_pose = self.robot.get_pose()
        # new_pose = current_pose.copy()
        # new_pose.pos.x -= 0.05
        # self.robot.set_pose(new_pose, wait=True)
        # finish the confirmation
        
        # Prompt for instruction input
        self.instruction = self.get_instruction()

        # Initialize the CvBridge
        self.bridge = CvBridge()

        # Subscribe to the image topic
        rospy.Subscriber('/usb_cam/image_raw', Image, self.image_callback)

        # Sleep for robot stabilization
        time.sleep(2)
        self.gripper = Robotiq_Two_Finger_Gripper(self.robot)

    def get_instruction(self):
        # Prompt the user to enter an instruction
        instruction = input(Fore.YELLOW + "Please enter the instruction: "+ Style.RESET_ALL)
        rospy.loginfo(Fore.GREEN + f'Instruction received: {instruction}'+ Style.RESET_ALL)
        return instruction

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Resize image to expected dimensions (256x320)
            resized_image = cv2.resize(cv_image, (256, 256))

            # Ensure image is uint8 format
            if resized_image.dtype != np.uint8:
                resized_image = resized_image.astype(np.uint8)
            
            # Construct the payload for OpenVLA server
            payload = {
                "image": resized_image,
                "instruction": self.instruction
            }
            
            # Send request to OpenVLA server
            print(Fore.BLUE + 'Sending request to OpenVLA server' + Style.RESET_ALL)
            response = requests.post(OPENVLA_SERVER_URL, json=payload)
            
            # Parse the response
            if response.status_code == 200:
                action = response.json()
                print(Fore.GREEN + f'Received action: {action}' + Style.RESET_ALL)
                
                # Execute the action on the robot
                self.control_ur5(action)
            else:
                rospy.logerr(f'Error from OpenVLA server: {response.text}')

        except CvBridgeError as e:
            rospy.logerr(f'Error converting image: {e}')

    def control_ur5(self, action):
        """
        Control the UR5 robot based on the action received from OpenVLA.
        OpenVLA returns absolute position and rotation values.
        """
        try:
            # Check if action is a dictionary or array
            if isinstance(action, dict):
                # Handle dictionary format action
                position = action.get('position', None)
                rotation = action.get('rotation', None)
                gripper_state = action.get('gripper', None)
                
                # Create a new pose
                current_pose = self.robot.get_pose()
                new_pose = current_pose.copy()
                
                # Update position if provided
                if position is not None and len(position) >= 3:
                    new_pose.pos.x = float(position[0])
                    new_pose.pos.y = float(position[1])
                    new_pose.pos.z = float(position[2])
                
                # Update rotation if provided
                if rotation is not None and len(rotation) >= 3:
                    # Convert rotation to m3d rotation
                    # This assumes rotation is provided as [rx, ry, rz] in radians
                    rx, ry, rz = float(rotation[0]), float(rotation[1]), float(rotation[2])
                    new_pose.orient = m3d.Orientation.new_euler([rx, ry, rz], encoding='xyz')
                
                # Move robot to new pose
                self.robot.set_pose(new_pose, wait=True)
                rospy.loginfo(Fore.BLUE + f'Moved robot to new pose: {new_pose}' + Style.RESET_ALL)
                
                # Control gripper if gripper state is provided
                if gripper_state is not None:
                    self.set_gripper(bool(gripper_state))
                    
            elif isinstance(action, np.ndarray):
                # Handle array format action
                # Assuming action format: [x, y, z, rx, ry, rz, gripper]
                if len(action) >= 6:
                    current_pose = self.robot.get_pose()
                    new_pose = current_pose.copy()
                    
                    # Update position
                    new_pose.pos.x = float(action[0])
                    new_pose.pos.y = float(action[1])
                    new_pose.pos.z = float(action[2])
                    
                    # Update rotation
                    rx, ry, rz = float(action[3]), float(action[4]), float(action[5])
                    new_pose.orient = m3d.Orientation.new_euler([rx, ry, rz], encoding='xyz')
                    
                    # Move robot to new pose
                    self.robot.set_pose(new_pose, wait=True)
                    rospy.loginfo(Fore.BLUE + f'Moved robot to new pose: {new_pose}' + Style.RESET_ALL)
                    
                    # Control gripper if provided
                    if len(action) >= 7:
                        self.set_gripper(bool(action[6] > 0.5))
                        
                else:
                    rospy.logwarn(f'Received action array of unexpected length: {len(action)}')
            else:
                rospy.logwarn(f'Received action of unexpected type: {type(action)}')
                
        except Exception as e:
            rospy.logerr(f'Error executing robot action: {e}')
            import traceback
            rospy.logerr(traceback.format_exc())
            
    def set_gripper(self, val):
        """
        gripper position control
        :param val: boolean (False:open, True:close)
        :return: None
        """
        if val:
            self.gripper.close_gripper()
        else:
            self.gripper.open_gripper()


    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = InferenceNode()
        node.run()
    except rospy.ROSInterruptException:
        pass

