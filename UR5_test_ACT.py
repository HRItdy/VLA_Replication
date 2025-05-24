from config.config import POLICY_CONFIG, TASK_CONFIG, TRAIN_CONFIG  # must import first

import os
import cv2
import torch
import pickle
import argparse
import numpy as np
import h5py
import rospy
from time import time
import urx
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState, Image
from cv_bridge import CvBridge

from training.utils import *

# parse the task name via command line
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='task1')
parser.add_argument('--robot_ip', type=str, default='192.168.1.100', help='IP address of the UR5 robot')
parser.add_argument('--camera_topic', type=str, default='/camera/rgb/image_raw', help='ROS topic for camera feed')
args = parser.parse_args()
task = args.task

# config
cfg = TASK_CONFIG
policy_config = POLICY_CONFIG
train_cfg = TRAIN_CONFIG
device = os.environ.get('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')


class UR5Robot:
    def __init__(self, ip):
        """
        Initialize connection to UR5 robot using URX
        """
        self.robot = urx.Robot(ip)
        
        # ROS publishers and subscribers
        self.joint_state_sub = rospy.Subscriber('/joint_states', JointState, self.joint_state_callback)
        self.joint_cmd_pub = rospy.Publisher('/ur5/joint_commands', Float64MultiArray, queue_size=10)
        
        # Robot state
        self.joint_positions = None
        self.joint_velocities = None
        self.rate = rospy.Rate(10)  # 10Hz
        
        print("Waiting for initial joint states...")
        while self.joint_positions is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        print("Robot connection established!")

    def joint_state_callback(self, msg):
        """Callback for joint state updates"""
        self.joint_positions = list(msg.position)
        self.joint_velocities = list(msg.velocity)

    def read_position(self):
        """Read current joint positions"""
        if self.joint_positions is None:
            return [0] * cfg['state_dim']  # Safe default
        return self.joint_positions

    def read_velocity(self):
        """Read current joint velocities"""
        if self.joint_velocities is None:
            return [0] * cfg['state_dim']  # Safe default
        return self.joint_velocities

    def set_goal_pos(self, position):
        """Send position command to robot"""
        # Send via ROS
        cmd_msg = Float64MultiArray()
        cmd_msg.data = position
        self.joint_cmd_pub.publish(cmd_msg)
        
        # Also send to URX for redundancy
        try:
            self.robot.movej(position, acc=0.1, vel=0.1, wait=False)
        except Exception as e:
            rospy.logwarn(f"URX command failed: {e}")
        
        # Wait a bit to allow movement
        self.rate.sleep()


class ROSCamera:
    def __init__(self, topic):
        """
        Initialize camera using ROS topic
        """
        self.bridge = CvBridge()
        self.image = None
        self.image_sub = rospy.Subscriber(topic, Image, self.image_callback)
        
        print("Waiting for camera image...")
        while self.image is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        print("Camera connection established!")
    
    def image_callback(self, msg):
        """Callback for image updates"""
        try:
            self.image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        except Exception as e:
            rospy.logerr(f"Failed to convert image: {e}")
    
    def capture_image(self):
        """Get the latest camera image and process it"""
        if self.image is None:
            return np.zeros((cfg['cam_height'], cfg['cam_width'], 3), dtype=np.uint8)
        
        # Process the image (crop and resize)
        image = self.image.copy()
        # Define your crop coordinates if needed
        x1, y1 = 400, 0  # Top left of crop rectangle
        x2, y2 = 1600, 900  # Bottom right of crop rectangle
        
        # Check if cropping is needed based on image dimensions
        if image.shape[0] > y2 and image.shape[1] > x2:
            image = image[y1:y2, x1:x2]
            
        # Resize to config dimensions
        image = cv2.resize(image, (cfg['cam_width'], cfg['cam_height']), interpolation=cv2.INTER_AREA)
        
        return image

if __name__ == "__main__":
    try:
        # Initialize ROS node
        rospy.init_node('act_ur5_controller', anonymous=True)
        
        # Initialize camera from ROS topic
        ros_camera = ROSCamera(args.camera_topic)
        
        # Initialize UR5 robot
        robot = UR5Robot(args.robot_ip)

        # Load the policy
        ckpt_path = os.path.join(train_cfg['checkpoint_dir'], train_cfg['eval_ckpt_name'])
        policy = make_policy(policy_config['policy_class'], policy_config)
        loading_status = policy.load_state_dict(torch.load(ckpt_path, map_location=torch.device(device)))
        print(loading_status)
        policy.to(device)
        policy.eval()

        print(f'Loaded: {ckpt_path}')
        stats_path = os.path.join(train_cfg['checkpoint_dir'], f'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)

        pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
        post_process = lambda a: a * stats['action_std'] + stats['action_mean']

        query_frequency = policy_config['num_queries']
        if policy_config['temporal_agg']:
            query_frequency = 1
            num_queries = policy_config['num_queries']

        # Initialize observation
        obs = {
            'qpos': robot.read_position(),
            'qvel': robot.read_velocity(),
            'images': {cn: ros_camera.capture_image() for cn in cfg['camera_names']}
        }
        
        # Notify start
        print("Starting execution")

        n_rollouts = 1
        for i in range(n_rollouts):
            ### evaluation loop
            if policy_config['temporal_agg']:
                all_time_actions = torch.zeros([cfg['episode_len'], cfg['episode_len']+num_queries, cfg['state_dim']]).to(device)
            qpos_history = torch.zeros((1, cfg['episode_len'], cfg['state_dim'])).to(device)
            
            with torch.inference_mode():
                # init buffers
                obs_replay = []
                action_replay = []
                for t in range(cfg['episode_len']):
                    qpos_numpy = np.array(obs['qpos'])
                    qpos = pre_process(qpos_numpy)
                    qpos = torch.from_numpy(qpos).float().to(device).unsqueeze(0)
                    qpos_history[:, t] = qpos
                    curr_image = get_image(obs['images'], cfg['camera_names'], device)

                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image)
                    if policy_config['temporal_agg']:
                        all_time_actions[[t], t:t+num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights.astype(np.float32)).to(device).unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]

                    ### post-process actions
                    raw_action = raw_action.squeeze(0).cpu().numpy()
                    action = post_process(raw_action)
                    
                    ### take action
                    robot.set_goal_pos(action)

                    ### update obs
                    obs = {
                        'qpos': robot.read_position(),
                        'qvel': robot.read_velocity(),
                        'images': {cn: ros_camera.capture_image() for cn in cfg['camera_names']}
                    }
                    
                    ### store data
                    obs_replay.append(obs)
                    action_replay.append(action)

            # Notify stop
            print("STOP")

            # create a dictionary to store the data
            data_dict = {
                '/observations/qpos': [],
                '/observations/qvel': [],
                '/action': [],
            }
            # there may be more than one camera
            for cam_name in cfg['camera_names']:
                data_dict[f'/observations/images/{cam_name}'] = []

            # store the observations and actions
            for o, a in zip(obs_replay, action_replay):
                data_dict['/observations/qpos'].append(o['qpos'])
                data_dict['/observations/qvel'].append(o['qvel'])
                data_dict['/action'].append(a)
                # store the images
                for cam_name in cfg['camera_names']:
                    data_dict[f'/observations/images/{cam_name}'].append(o['images'][cam_name])

            t0 = time()
            max_timesteps = len(data_dict['/observations/qpos'])
            # create data dir if it doesn't exist
            data_dir = cfg['dataset_dir']  
            if not os.path.exists(data_dir): os.makedirs(data_dir)
            # count number of files in the directory
            idx = len([name for name in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, name))])
            dataset_path = os.path.join(data_dir, f'episode_{idx}')
            # save the data
            with h5py.File("data/demo/trained.hdf5", 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
                root.attrs['sim'] = False  # Changed to False since we're using real robot
                obs = root.create_group('observations')
                image = obs.create_group('images')
                for cam_name in cfg['camera_names']:
                    _ = image.create_dataset(cam_name, (max_timesteps, cfg['cam_height'], cfg['cam_width'], 3), dtype='uint8',
                                            chunks=(1, cfg['cam_height'], cfg['cam_width'], 3), )
                qpos = obs.create_dataset('qpos', (max_timesteps, cfg['state_dim']))
                qvel = obs.create_dataset('qvel', (max_timesteps, cfg['state_dim']))
                action = root.create_dataset('action', (max_timesteps, cfg['action_dim']))
                
                for name, array in data_dict.items():
                    root[name][...] = array
            
    except Exception as e:
        print(f"Error: {e}")