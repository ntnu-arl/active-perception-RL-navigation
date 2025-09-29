#!/usr/bin/env python

import rospy
from std_msgs.msg import UInt8MultiArray
from visualization_msgs.msg import Marker
import geometry_msgs.msg
import numpy as np

# Set your grid shape here (e.g., 20x20x20)
GRID_SHAPE = (21, 21, 21)

class LocalGridVisualizer:
    def __init__(self):
        rospy.init_node('local_grid_visualizer')
        self.marker_pub = rospy.Publisher('/local_grid/markers', Marker, queue_size=1)
        rospy.Subscriber('/occupancy_node/local_map', UInt8MultiArray, self.map_callback)

    def map_callback(self, msg):
        # Handle both bytes and list cases for msg.data
        if isinstance(msg.data, bytes):
            data = np.frombuffer(msg.data, dtype=np.uint8)
        else:
            data = np.array(msg.data, dtype=np.uint8)
        if data.size != np.prod(GRID_SHAPE):
            rospy.logwarn("Received data does not match expected grid shape!")
            return
        grid = data.reshape(GRID_SHAPE)
        self.publish_markers(grid)

    def publish_markers(self, grid):
        marker = Marker()
        marker.header.frame_id = "robot"
        # marker.header.stamp = rospy.Time.now()
        marker.ns = "local_grid"
        marker.id = 0
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.05  # Set voxel size
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        marker.points = []
        for x in range(GRID_SHAPE[0]):
            for y in range(GRID_SHAPE[1]):
                for z in range(GRID_SHAPE[2]):
                    if grid[x, y, z] == 1:
                        pt = geometry_msgs.msg.Point()
                        pt.x = (x-11) * 0.1
                        pt.y = (y-11) * 0.1
                        pt.z = (z-11) * 0.1
                        marker.points.append(pt)
        self.marker_pub.publish(marker)

if __name__ == '__main__':
    LocalGridVisualizer()
    rospy.spin()
