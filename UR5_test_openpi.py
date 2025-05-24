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
from openpi_client import image_tools
from openpi_client import websocket_client_policy


IP_ROBOT = "192.168.0.6"
TRANS_TCP = (0, 0, 0.19, 1.2092, -1.2092, 1.2092)
#TRANS_BASE = [0, 0, 0, -0.61394313, 1.48218982, 0.61394313]
TRANS_BASE = [0, 0, 0, 0, 0, 3.1415926]
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

        
        # Initialize the OpenPI client
        self.policy_client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)

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
            
            # Resize and convert image using OpenPI utilities
            processed_image = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(cv_image, 224, 224)
            )
            
            # Construct the observation dictionary
            observation = {
                "observation/image": processed_image,
                "prompt": self.instruction,
            }
            
            # Perform inference using the OpenPI client
            print(Fore.BLUE + 'Performing inference with OpenPI client' + Style.RESET_ALL)
            action_chunk = self.policy_client.infer(observation)["action"]
            action = np.asarray(action_chunk[:, :7])
            
            print(action)
            # Convert action to UR5 command
            self.control_ur5(action)
            
        except CvBridgeError as e:
            rospy.logerr(f'Error converting image: {e}')
        except Exception as e:
            rospy.logerr(f'Error during inference: {e}')

    def control_ur5(self, action):
        # Example RT-1 action
        # action_dict = {
        #    'base_displacement_vector': np.array([0.1, 0.0], dtype='float32'),
        #    'base_displacement_vertical_rotation': np.array([0.0], dtype='float32'),
        #    'gripper_closedness_action': np.array([1.0], dtype='float32'),
        #    'rotation_delta': np.array([0.1, 0.0, 0.0], dtype='float32'),
        #    'terminate_episode': np.array([0, 0, 0], dtype='int32'),
        #    'world_vector': np.array([0.2, 0.0, 0.0], dtype='float32')
        #}

        # if isinstance(action, dict):
        #     # Assuming action = [delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw]
        #     base_vec = action['base_displacement_vector']
        #     base_rot = action['base_displacement_vertical_rotation']
        #     gripper_pos = action['gripper_closedness_action']
        #     arm_del = action['world_vector']
        #     arm_rot = action['rotation_delta']
        #     terminate_episode = action['terminate_episode']

        #     # Update pose with deltas
        #     current_pose = self.robot.get_pose()
        #     new_pose = current_pose.copy()
        #     new_pose.pos.x += arm_del[0]
        #     new_pose.pos.y += arm_del[1]
        #     new_pose.pos.z += arm_del[2]
        #     new_pose.orient.rotate_x(arm_rot[0])
        #     new_pose.orient.rotate_y(arm_rot[1])
        #     new_pose.orient.rotate_z(arm_rot[2])

        #     # Move to the new pose
        #     self.robot.set_pose(new_pose, wait=True)
            
        #     # Gripper action
        #     if gripper_pos > 0:
        #         self.set_gripper(True)  # Close gripper
        #     else:
        #         self.set_gripper(False)  # Open gripper
        #     rospy.loginfo(Fore.BLUE + f'Moved UR5 to new pose with deltas: {action}' + Style.RESET_ALL)
        # else:
        #     rospy.logwarn(f'Invalid action received: {action}')

        # Assuming action = [x, y, z, r, p, y]
        gripper_pos = action[-1]
        arm_del = action[0:3]
        rotation_vector = action[3:]
        
        # Update pose with deltas
        current_pose = self.robot.get_pose()
        new_pose = current_pose.copy()
        new_pose.pos.x = arm_del[0]
        new_pose.pos.y = arm_del[1]
        new_pose.pos.z = arm_del[2]
        
        # Update orientation using rotation vector
        # If the rotation vector represents axis-angle rotation:
        if np.linalg.norm(rotation_vector) > 0:
            # Convert axis-angle to rotation matrix
            angle = np.linalg.norm(rotation_vector)
            axis = rotation_vector / angle
            new_pose.orient = m3d.Orientation.new_axangle(axis, angle)
        # If the rotation vector represents Euler angles (rx, ry, rz):
        else:
            # Specify the rotation convention (e.g., 'xyz', 'zyx', etc.)
            new_pose.orient = m3d.Orientation.new_euler(rotation_vector, encoding='xyz')
        
        # Move robot to new pose
        rospy.loginfo(Fore.BLUE + f"Moving to: {arm_del}, rotation: {rotation_vector}" + Style.RESET_ALL)
        # Move to the new pose
        self.robot.set_pose(new_pose, wait=True)
        
        # Gripper action
        if gripper_pos > 0:
            self.set_gripper(True)  # Close gripper
        else:
            self.set_gripper(False)  # Open gripper
        rospy.loginfo(Fore.BLUE + f'Moved UR5 to new pose with deltas: {action}' + Style.RESET_ALL)
    

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

