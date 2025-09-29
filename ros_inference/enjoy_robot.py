import sys

from ros_inference.enjoy_custom_net_static_camera_ros import NN_Inference_ROS, parse_aerialgym_cfg, quat_rotate, quat_conjugate, quat_rotate_inverse, quat_mul

import torch
# import torchvision
import numpy as np

import rospy
import tf
from std_msgs.msg import UInt8MultiArray, Header, Float64
from visualization_msgs.msg import Marker
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped, PoseStamped, Vector3
from sensor_msgs.msg import Image, JointState
from mavros_msgs.msg import PositionTarget
import ros_numpy
import os
import numpy as np
from ros_inference.vae_image_encoder import VAEImageEncoder
from scipy.spatial.transform import Rotation

class vae_config:
    use_vae = True
    latent_dims = 64
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_file = (dir_path + "/utils/ICRA_vae.pth")
    model_folder = dir_path + "/utils/"
    image_res = (270, 480)
    interpolation_mode = "nearest"
    return_sampled_latent = True


class EMA:
    def __init__(self, beta):
        self.beta = beta
        self.value = None

    def reset(self):
        self.value = None

    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.beta * self.value + (1 - self.beta) * new_value
        return self.value

class RlActiveCamera:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.enable_rl = True
        self.camera_joint_pitch = 0.0
        self.camera_joint_yaw = 0.0
        
        # observation 1d
        self.odom = Odometry()
        self.action = np.zeros(6)
        self.prev_action = np.zeros(6)
        self.target = np.zeros(3)
        self.position = np.zeros(3)
        self.rpy = np.zeros(3)
        self.dist_divider = 1.0
        self.agent_state = torch.zeros((85), device=self.device, requires_grad=False)

        cfg = parse_aerialgym_cfg(evaluation=True)
        self.nn_model = NN_Inference_ROS(cfg)
        self.VAE_net_interface = VAEImageEncoder(config=vae_config, device=self.device)

        image_res = (270, 480)
        self.image_height = image_res[0]
        self.image_width = image_res[1]

        self.local_occupancy_map = torch.zeros((21, 21, 21), device=self.device, requires_grad=False)
        self.current_height = 0.0
        self.max_height = 1.5
        self.depth_image = -1.0 * torch.ones((self.image_height, self.image_width), device=self.device, requires_grad=False)
        self.max_depth = 10.0
        self.min_depth = 0.2
        
        self.local_occupancy_map_subscriber = rospy.Subscriber(
            "/occupancy_node/local_map", UInt8MultiArray, self.localOccupancyMapCallback)
        self.odometry_subscriber = rospy.Subscriber(
            "/rig_node/graph/odom", Odometry, self.odometryCallback)
        self.joint_state_subscriber = rospy.Subscriber(
            "/camera_joint_states", JointState, self.cameraJointStateCallback)
        self.target_sub = rospy.Subscriber(
            "/target", PoseStamped, self.target_callback, queue_size=1
        )

        self.action_publisher = rospy.Publisher(
            "/mavros/setpoint_raw/local", PositionTarget, queue_size=1)
        self.action_publisher_viz = rospy.Publisher(                        
            "/mavros/setpoint_raw/local_viz", TwistStamped, queue_size=1)
        self.camera_joint_pitch_controller = rospy.Publisher(
            "/servo_pitch_target", Float64, queue_size=1)
        self.camera_joint_yaw_controller = rospy.Publisher(
            "/servo_yaw_target", Float64, queue_size=1)
        self.action_publisher_camera_viz = rospy.Publisher(                        
            "/camera_joint/local_viz", TwistStamped, queue_size=1)
        self.depth_image_publisher = rospy.Publisher(
            "/rl_policy/depth_image", Image, queue_size=1) 
        self.decoded_depth_image_publisher = rospy.Publisher(
            "/rl_policy/decoded_depth_image", Image, queue_size=1)
        self.goal_dir_publisher = rospy.Publisher(
            "/rl_policy/goal_direction", Marker, queue_size=1)

        # d455 camera
        self.image_height_camera = 480
        self.image_width_camera = 640
        self.fx_camera = 386.1221923828125
        self.fy_camera = 386.1221923828125
        self.cx_camera = 318.6071472167969
        self.cy_camera = 236.15013122558594
        
        self.fx = self.fx_camera * self.image_width / self.image_width_camera
        self.fy = self.fy_camera * self.image_height / self.image_height_camera
        self.cx = self.cx_camera * self.image_width / self.image_width_camera
        self.cy = self.cy_camera * self.image_height / self.image_height_camera
        self.depth_image_topic = "/camera/depth/image_rect_raw"
        
        self.depth_image_camera = -1.0 * torch.ones((self.image_height_camera, self.image_width_camera), device=self.device, requires_grad=False)
        self.depth_image_camera_subscriber = rospy.Subscriber(
            self.depth_image_topic, Image, self.depthImageCameraCallback)
    
    def localOccupancyMapCallback(self, msg):
        local_occupancy_map = torch.from_numpy(np.ndarray((21, 21, 21), np.uint8, msg.data, 0)).to(self.device).float()
        local_occupancy_map = torch.where(local_occupancy_map == 0, 0.0, local_occupancy_map)
        local_occupancy_map = torch.where(local_occupancy_map == 1, -2.0, local_occupancy_map)
        local_occupancy_map = torch.where(local_occupancy_map == 2, -1.0, local_occupancy_map)
        self.local_occupancy_map = -1.0 * local_occupancy_map
    
    def odometryCallback(self, msg):
        agent_state = torch.zeros((85), device=self.device, requires_grad=False)
        
        self.position[0] = msg.pose.pose.position.x
        self.position[1] = msg.pose.pose.position.y
        self.position[2] = msg.pose.pose.position.z
        self.current_height = msg.pose.pose.position.z

        quat_msg = msg.pose.pose.orientation
        quat = Rotation.from_quat([quat_msg.x, quat_msg.y, quat_msg.z, quat_msg.w])
        ### TODO: Check if this is the correct order for the euler angles
        rpy = quat.as_euler("xyz", degrees=False)

        self.rpy[0] = rpy[0]
        self.rpy[1] = rpy[1]
        self.rpy[2] = rpy[2]

        self.vehicle_rpy = self.rpy.copy()
        self.vehicle_rpy[2] = 0.0
        vehicle_frame_matrix = Rotation.from_euler("xyz", self.rpy, degrees=False)
        
        goal_dir = self.target - self.position
        goal_dir = vehicle_frame_matrix.inv().apply(goal_dir)
        goal_direction = goal_dir / np.linalg.norm(goal_dir)
        goal_magnitude = np.linalg.norm(goal_dir) / self.dist_divider
        goal_magnitude = np.clip(goal_magnitude, 0, 10.0)
        if goal_magnitude < 1.5:
            self.enable_rl = rospy.set_param("/rl_policy/enable_planner", False)
        

        # agent state 1D 
        agent_state[0] = goal_direction[0]
        agent_state[1] = goal_direction[1]
        agent_state[2] = goal_direction[2]
        agent_state[3] = goal_magnitude
        agent_state[4] = self.rpy[0]
        agent_state[5] = self.rpy[1]
        agent_state[6] = 0.0
        # lin vel 
        agent_state[7] = msg.twist.twist.linear.x
        agent_state[8] = msg.twist.twist.linear.y
        agent_state[9] = msg.twist.twist.linear.z
        # ang vel
        agent_state[10] = msg.twist.twist.angular.x
        agent_state[11] = msg.twist.twist.angular.y
        agent_state[12] = msg.twist.twist.angular.z
        # action 
        agent_state[13:19] = torch.from_numpy(self.prev_action)
        # camera joint state
        agent_state[19] = self.camera_joint_pitch
        agent_state[20] = self.camera_joint_yaw
        
        self.agent_state[:21] = agent_state[:21].clone()

        goal_dir_marker = Marker()
        goal_dir_marker.header.stamp = rospy.Time.now()
        goal_dir_marker.header.frame_id = "robot"
        # marker type arrow
        goal_dir_marker.type = 0
        # start point is 0,0,0
        goal_dir_marker.points.append(Vector3(0.0, 0.0, 0.0))
        # end point is the goal direction
        goal_dir_marker.points.append(Vector3(goal_dir[0], goal_dir[1], goal_dir[2]))
        goal_dir_marker.scale.x = 0.1
        goal_dir_marker.scale.y = 0.3
        goal_dir_marker.scale.z = 0.2
        goal_dir_marker.color.a = 1.0
        goal_dir_marker.color.r = 0.0
        goal_dir_marker.color.g = 0.0
        goal_dir_marker.color.b = 1.0

        self.goal_dir_publisher.publish(goal_dir_marker)

    def cameraJointStateCallback(self, msg):
        self.camera_joint_pitch = msg.position[0]
        self.camera_joint_yaw = msg.position[1]

    def depthImageCameraCallback(self, msg):
        depth_image_camera = torch.from_numpy(ros_numpy.numpify(msg).astype('float32')).to(self.device) / 1000.0
        self.depth_image_camera[:] = torch.where((depth_image_camera < self.min_depth), -self.max_depth, depth_image_camera)/self.max_depth
        self.depth_image_camera[:] = torch.clamp(self.depth_image_camera, -1.0, 1.0)
        self.depth_image = torch.nn.functional.interpolate(self.depth_image_camera.unsqueeze(0).unsqueeze(0), size=(self.image_height, self.image_width), mode='nearest-exact').squeeze()
        self.agent_state[21:] = self.VAE_net_interface.encode(self.depth_image.unsqueeze(0).unsqueeze(0))

    def target_callback(self, target):
        self.target[0] = target.pose.position.x
        self.target[1] = target.pose.position.y
        self.target[2] = target.pose.position.z
        # Also reset Network state when a new target is received
        self.nn_model.reset()
    
def main():
    ema_x = 0.0
    alpha_x = 0.5
    
    ema_y = 0.0
    alpha_y = 0.5
    
    ema_z = 0.0
    alpha_z = 0.5
    
    ema_yaw = 0.0
    alpha_yaw = 0.5

    ema_camera_pitch = 0.0
    alpha_camera_pitch = 0.5
    
    ema_camera_yaw = 0.0
    alpha_camera_yaw = 0.5

    # Initialize ROS node
    rospy.init_node('rl_active_camera_py')
    rl_active_camera = RlActiveCamera()
    init = True
    
    action_mavros = PositionTarget()
    action_mavros.coordinate_frame = PositionTarget.FRAME_BODY_NED
    action_mavros.type_mask = PositionTarget.IGNORE_PX + PositionTarget.IGNORE_PY + PositionTarget.IGNORE_PZ + \
                    PositionTarget.IGNORE_AFX + PositionTarget.IGNORE_AFY + \
                    PositionTarget.IGNORE_AFZ + PositionTarget.IGNORE_YAW
    action_ros_camera_pitch = Float64()
    action_ros_camera_yaw = Float64()
    
    action_mavros_viz = TwistStamped()
    action_mavros_viz.header = Header(frame_id="robot")
    action_camera_viz = TwistStamped()
    action_camera_viz.header = Header(frame_id="camera_rgb_camera_optical_frame")
    
    rate = rospy.Rate(10) # 10hz
    
    observations = torch.zeros((1, 85), dtype=torch.float32, requires_grad=False, device=rl_active_camera.device)
    observations_map = torch.zeros((1, 1, 21, 21, 21), dtype=torch.float32, requires_grad=False, device=rl_active_camera.device)

    rl_active_camera.nn_model.reset()
    obs_dict = {"observations": observations, "observations_map": observations_map}
    counter = 1
    
    while not rospy.is_shutdown():

        if rl_active_camera.enable_rl:
            
            # observation space -- robot state
            obs_dict["observations"] = rl_active_camera.agent_state.unsqueeze(0)
            # observation space -- map
            obs_dict["observations_map"]   = rl_active_camera.local_occupancy_map.unsqueeze(0)
            
            actions = rl_active_camera.nn_model.get_action(obs_dict)
            actions = np.clip(actions, -1.0, 1.0)
            velocity_magnitude = np.linalg.norm(actions[0:3])
            scaling_factor = np.clip(1.5 / (velocity_magnitude + 1e-6), a_min=0.0, a_max=1.0)
            actions[0:3] *= scaling_factor
            actions[3] *= 1.0
            actions[4] *= np.pi / 3.0
            actions[5] *= np.pi / 4.0
            rl_active_camera.prev_action = actions.copy()
            actions[0] *= 0.3
            actions[1] *= 0.3
            actions[2] *= 0.3
            actions[3] *= 0.3
            
            ema_x = (actions[0] * alpha_x) + (ema_x * (1 - alpha_x))
            ema_y = (actions[1] * alpha_y) + (ema_y * (1 - alpha_y))
            ema_z = (actions[2] * alpha_z) + (ema_z * (1 - alpha_z))
            ema_yaw = (actions[3] * alpha_yaw) + (ema_yaw * (1 - alpha_yaw))
            ema_camera_pitch = (actions[4] * alpha_camera_pitch) + (ema_camera_pitch * (1 - alpha_camera_pitch))
            ema_camera_yaw = (actions[5] * alpha_camera_yaw) + (ema_camera_yaw * (1 - alpha_camera_yaw))
            
            action_mavros.velocity.x = ema_x
            action_mavros.velocity.y = ema_y
            action_mavros.velocity.z = ema_z
            action_mavros.yaw_rate = ema_yaw
            
            action_ros_camera_pitch.data = np.rad2deg(ema_camera_pitch)
            action_ros_camera_yaw.data = np.rad2deg(ema_camera_yaw)
            
            
            action_mavros_viz.header.stamp = rospy.Time.now()
            action_mavros_viz.twist.linear.x = ema_x
            action_mavros_viz.twist.linear.y = ema_y
            action_mavros_viz.twist.linear.z = ema_z
            action_mavros_viz.twist.angular.z = ema_yaw
            
            action_camera_viz.header.stamp = rospy.Time.now()
            action_camera_viz.twist.angular.y = actions[4]
            action_camera_viz.twist.angular.z = actions[5]
            
            counter += 1
            if init == True:
                init = False
                rl_active_camera.enable_rl = rospy.set_param("/rl_policy/enable_planner", False)
        else:
            # reset RL planner if it is not in use
            rl_active_camera.nn_model.reset()
            # send zero velocity commands if not in use
            action_mavros.velocity.x = 0.0
            action_mavros.velocity.y = 0.0
            action_mavros.velocity.z = 0.0
            action_mavros.yaw_rate = 0.0

            action_ros_camera_pitch.data = 0.0
            action_ros_camera_yaw.data = 0.0
            
            action_mavros_viz.header.stamp = rospy.Time.now()
            action_mavros_viz.twist.linear.x = 0.0
            action_mavros_viz.twist.linear.y = 0.0
            action_mavros_viz.twist.linear.z = 0.0
            action_mavros_viz.twist.angular.z = 0.0
            
            action_camera_viz.header.stamp = rospy.Time.now()
            action_camera_viz.twist.angular.y = 0.0
            action_camera_viz.twist.angular.z = 0.0

            ema_x = 0.0
            ema_y = 0.0
            ema_z = 0.0
            ema_yaw = 0.0
            
            ema_camera_pitch = 0.0
            ema_camera_yaw = 0.0
        
        rl_active_camera.action_publisher.publish(action_mavros)
        rl_active_camera.action_publisher_viz.publish(action_mavros_viz)
        rl_active_camera.camera_joint_pitch_controller.publish(action_ros_camera_pitch)
        rl_active_camera.camera_joint_yaw_controller.publish(action_ros_camera_yaw)
        rl_active_camera.action_publisher_camera_viz.publish(action_camera_viz)
        rl_active_camera.depth_image_publisher.publish(ros_numpy.msgify(Image, rl_active_camera.depth_image.cpu().numpy(), encoding="32FC1"))
        image = rl_active_camera.VAE_net_interface.decode(rl_active_camera.agent_state[21:].unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        rl_active_camera.decoded_depth_image_publisher.publish(ros_numpy.msgify(Image, image.cpu().numpy(), encoding="32FC1"))
        rl_active_camera.enable_rl = rospy.get_param("/rl_policy/enable_planner")

        rospy.loginfo_throttle_identical(2, "RL-Planner is active %s", rl_active_camera.enable_rl)
        rate.sleep()

if __name__ == "__main__":
    sys.exit(main())
