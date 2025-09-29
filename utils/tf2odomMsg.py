#!/usr/bin/env python

import rospy
import tf
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, PoseWithCovariance, TwistWithCovariance
from std_msgs.msg import Empty

PARENT_FRAME = "map"  # Change if needed
CHILD_FRAME = "camera_depth_optical_frame"
CHILD_FRAME_BODY = "body"
ODOM_TOPIC = "/camera_depth_optical_frame/odom"
ODOM_TOPIC_BODY = "/body/odom"

class TF2OdomPublisher:
    def __init__(self):
        rospy.init_node('tf2odom_publisher')
        self.odom_pub = rospy.Publisher(ODOM_TOPIC, Odometry, queue_size=1)
        self.odom_pub_body = rospy.Publisher(ODOM_TOPIC_BODY, Odometry, queue_size=1)
        self.tf_listener = tf.TransformListener()
        self.triggered = False
        rospy.Subscriber('/publish_odom_trigger', Empty, self.trigger_callback)
        # self.rate = rospy.Rate(20)  # Not needed for trigger mode

    def trigger_callback(self, msg):
        self.triggered = True

    def spin(self):
        while not rospy.is_shutdown():
            if self.triggered:
                try:
                    (trans, rot) = self.tf_listener.lookupTransform(PARENT_FRAME, CHILD_FRAME, rospy.Time(0))
                    # Publish original odometry
                    odom_msg = Odometry()
                    odom_msg.header.stamp = rospy.Time.now()
                    odom_msg.header.frame_id = PARENT_FRAME
                    odom_msg.child_frame_id = CHILD_FRAME
                    odom_msg.pose.pose.position.x = trans[0]
                    odom_msg.pose.pose.position.y = trans[1]
                    odom_msg.pose.pose.position.z = trans[2]
                    odom_msg.pose.pose.orientation.x = rot[0]
                    odom_msg.pose.pose.orientation.y = rot[1]
                    odom_msg.pose.pose.orientation.z = rot[2]
                    odom_msg.pose.pose.orientation.w = rot[3]
                    self.odom_pub.publish(odom_msg)

                    # Publish odometry with +90 deg about y axis of camera_depth_optical_frame
                    plus90_y_quat = tf.transformations.quaternion_from_euler(0, np.pi/2, 0)
                    rot_plus90 = tf.transformations.quaternion_multiply(rot, plus90_y_quat)
                    odom_msg_plus90 = Odometry()
                    odom_msg_plus90.header.stamp = rospy.Time.now()
                    odom_msg_plus90.header.frame_id = PARENT_FRAME
                    odom_msg_plus90.child_frame_id = CHILD_FRAME + "_plus90y"
                    odom_msg_plus90.pose.pose.position.x = trans[0]
                    odom_msg_plus90.pose.pose.position.y = trans[1]
                    odom_msg_plus90.pose.pose.position.z = trans[2]
                    odom_msg_plus90.pose.pose.orientation.x = rot_plus90[0]
                    odom_msg_plus90.pose.pose.orientation.y = rot_plus90[1]
                    odom_msg_plus90.pose.pose.orientation.z = rot_plus90[2]
                    odom_msg_plus90.pose.pose.orientation.w = rot_plus90[3]
                    self.odom_pub.publish(odom_msg_plus90)

                    # Publish odometry with -90 deg about y axis of camera_depth_optical_frame
                    minus90_y_quat = tf.transformations.quaternion_from_euler(0, -np.pi/2, 0)
                    rot_minus90 = tf.transformations.quaternion_multiply(rot, minus90_y_quat)
                    odom_msg_minus90 = Odometry()
                    odom_msg_minus90.header.stamp = rospy.Time.now()
                    odom_msg_minus90.header.frame_id = PARENT_FRAME
                    odom_msg_minus90.child_frame_id = CHILD_FRAME + "_minus90y"
                    odom_msg_minus90.pose.pose.position.x = trans[0]
                    odom_msg_minus90.pose.pose.position.y = trans[1]
                    odom_msg_minus90.pose.pose.position.z = trans[2]
                    odom_msg_minus90.pose.pose.orientation.x = rot_minus90[0]
                    odom_msg_minus90.pose.pose.orientation.y = rot_minus90[1]
                    odom_msg_minus90.pose.pose.orientation.z = rot_minus90[2]
                    odom_msg_minus90.pose.pose.orientation.w = rot_minus90[3]
                    self.odom_pub.publish(odom_msg_minus90)

                    (trans, rot) = self.tf_listener.lookupTransform(PARENT_FRAME, CHILD_FRAME_BODY, rospy.Time(0))
                    odom_msg = Odometry()
                    odom_msg.header.stamp = rospy.Time.now()
                    odom_msg.header.frame_id = PARENT_FRAME
                    odom_msg.child_frame_id = CHILD_FRAME
                    odom_msg.pose.pose.position.x = trans[0]
                    odom_msg.pose.pose.position.y = trans[1]
                    odom_msg.pose.pose.position.z = trans[2]
                    odom_msg.pose.pose.orientation.x = rot[0]
                    odom_msg.pose.pose.orientation.y = rot[1]
                    odom_msg.pose.pose.orientation.z = rot[2]
                    odom_msg.pose.pose.orientation.w = rot[3]
                    self.odom_pub_body.publish(odom_msg)
                    
                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    pass
                self.triggered = False
            rospy.sleep(0.05)

if __name__ == '__main__':
    TF2OdomPublisher().spin()