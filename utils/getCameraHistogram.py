#!/usr/bin/env python

import rosbag
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sensor_msgs.msg import JointState
import os

# Bag location and topic
# Mission L-corridor
start_time_offset = 44.0
end_time_offset = 84.0
bag_dir = "..."

bag_files = [os.path.join(bag_dir, f) for f in os.listdir(bag_dir) if f.endswith('.bag')]
topic = "/camera_joint_states"

# Histogram settings
pitch_min, pitch_max = -np.pi/3, np.pi/3
yaw_min, yaw_max = -np.pi/4, np.pi/4
bin_size_deg = 15
pitch_bins = np.arange(pitch_min, pitch_max + np.deg2rad(bin_size_deg), np.deg2rad(bin_size_deg))
yaw_bins = np.arange(yaw_min, yaw_max + np.deg2rad(bin_size_deg), np.deg2rad(bin_size_deg))

pitch_data = []
yaw_data = []

rcParams['font.family'] = 'DejaVu Sans'  # Use a widely available font
rcParams['font.size'] = 36  # Increase default font size

plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

current_time_sec = [None]  # Use a mutable object to allow update inside update_hist

def update_hist():
    ax1.clear()
    ax2.clear()
    # Pitch histogram normalization
    if len(pitch_data) > 0:
        counts, bins, patches = ax1.hist(pitch_data, bins=pitch_bins, color='blue', alpha=0.7)
        total = counts.sum()
        if total > 0:
            for rect in patches:
                rect.set_height(rect.get_height() / total)
    # ax1.set_title('Camera Pitch Histogram')
    ax1.set_xlabel('Pitch')
    ax1.set_ylabel('')
    ax1.set_xlim(pitch_min, pitch_max)
    ax1.set_ylim(0, 0.4)
    ax1.set_yticks(np.arange(0, 0.41, 0.1))
    # Set x-ticks every 15 degrees for both plots
    xticks_deg = np.arange(-60, 61, 15)  # -60 to 60 degrees
    xticks_rad = np.deg2rad(xticks_deg)
    ax1.set_xticks(xticks_rad)
    ax1.set_xticklabels([f"{deg}°" for deg in xticks_deg])
    # Yaw histogram normalization
    if len(yaw_data) > 0:
        counts, bins, patches = ax2.hist(yaw_data, bins=yaw_bins, color='green', alpha=0.7)
        total = counts.sum()
        if total > 0:
            for rect in patches:
                rect.set_height(rect.get_height() / total)
    # ax2.set_title('Camera Yaw Histogram')
    ax2.set_xlabel('Yaw')
    ax2.set_ylabel('')
    # Set yaw x-axis from -45° to 45°
    yaw_min_deg, yaw_max_deg = -45, 45
    yaw_min_rad, yaw_max_rad = np.deg2rad(yaw_min_deg), np.deg2rad(yaw_max_deg)
    ax2.set_xlim(yaw_min_rad, yaw_max_rad)
    ax2.set_ylim(0, 0.4)
    ax2.set_yticks(np.arange(0, 0.41, 0.1))
    # Set x-ticks for yaw from -45° to 45° every 15°
    xticks_yaw_deg = np.arange(-45, 46, 15)
    xticks_yaw_rad = np.deg2rad(xticks_yaw_deg)
    ax2.set_xticks(xticks_yaw_rad)
    ax2.set_xticklabels([f"{deg}°" for deg in xticks_yaw_deg])
    if current_time_sec[0] is not None:
        fig.suptitle(f"Current time: {current_time_sec[0]:.2f} s", fontsize=14)
    fig.text(0.0, 0.5, 'Fraction of mission time', va='center', rotation='vertical', fontsize=rcParams['font.size'])
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.draw()
    plt.pause(0.001)

# Find the global start time across all bags
all_start_times = []
for bag_file in bag_files:
    with rosbag.Bag(bag_file) as bag:
        all_start_times.append(bag.get_start_time())
global_start_time = min(all_start_times)

skip_until = global_start_time + start_time_offset
end_at = global_start_time + end_time_offset

update_every = 20  # Update plot every 100 messages for speed
msg_count = 0

for bag_file in bag_files:
    print(f"Processing {bag_file}")
    with rosbag.Bag(bag_file) as bag:
        for topic_name, msg, t in bag.read_messages(topics=[topic]):
            t_sec = t.to_sec()
            if t_sec < skip_until:
                continue
            if t_sec > end_at:
                break
            msg_count += 1
            if msg_count % update_every == 0:
                pitch = msg.position[0]
                yaw = msg.position[1]
                pitch_data.append(pitch)
                yaw_data.append(yaw)
                current_time_sec[0] = t_sec - global_start_time
                update_hist()

# Final update at the end
update_hist()



# --- High-resolution animation and video export ---
import matplotlib.animation as animation

def animate(i):
    ax1.clear()
    ax2.clear()
    # Pitch histogram normalization
    if i > 0:
        counts, bins, patches = ax1.hist(pitch_data[:i], bins=pitch_bins, color='blue', alpha=0.7)
        total = counts.sum()
        if total > 0:
            for rect in patches:
                rect.set_height(rect.get_height() / total)
    ax1.set_xlabel('Pitch')
    ax1.set_ylabel('')
    ax1.set_xlim(pitch_min, pitch_max)
    ax1.set_ylim(0, 0.4)
    ax1.set_yticks(np.arange(0, 0.41, 0.1))
    xticks_deg = np.arange(-60, 61, 15)
    xticks_rad = np.deg2rad(xticks_deg)
    ax1.set_xticks(xticks_rad)
    ax1.set_xticklabels([f"{deg}°" for deg in xticks_deg])
    # Yaw histogram normalization
    if i > 0:
        counts, bins, patches = ax2.hist(yaw_data[:i], bins=yaw_bins, color='green', alpha=0.7)
        total = counts.sum()
        if total > 0:
            for rect in patches:
                rect.set_height(rect.get_height() / total)
    ax2.set_xlabel('Yaw')
    ax2.set_ylabel('')
    yaw_min_deg, yaw_max_deg = -45, 45
    yaw_min_rad, yaw_max_rad = np.deg2rad(yaw_min_deg), np.deg2rad(yaw_max_deg)
    ax2.set_xlim(yaw_min_rad, yaw_max_rad)
    ax2.set_ylim(0, 0.4)
    ax2.set_yticks(np.arange(0, 0.41, 0.1))
    xticks_yaw_deg = np.arange(-45, 46, 15)
    xticks_yaw_rad = np.deg2rad(xticks_yaw_deg)
    ax2.set_xticks(xticks_yaw_rad)
    ax2.set_xticklabels([f"{deg}°" for deg in xticks_yaw_deg])
    if i < len(pitch_data):
        fig.suptitle(f"Current time: {i * update_every:.2f} steps", fontsize=14)
    fig.text(0.0, 0.5, 'Fraction of mission time', va='center', rotation='vertical', fontsize=rcParams['font.size'])
    plt.tight_layout(rect=[0, 0, 1, 0.96])


# Set video filename based on bag directory name
bag_dir_name = os.path.basename(os.path.normpath(bag_dir))
video_filename = f"camera_histogram_animation_{bag_dir_name}.mp4"
print(f"Exporting animation to high-resolution MP4 as {video_filename} (this may take a while)...")
ani = animation.FuncAnimation(fig, animate, frames=len(pitch_data), interval=20, repeat=False)
ani.save(video_filename, writer='ffmpeg', dpi=400, bitrate=4000)
print(f'Video saved as {video_filename}')