import time
import numpy as np
import RPi.GPIO as GPIO
import rospy
from std_msgs.msg import Float64
# JointState publisher for yaw and pitch
from sensor_msgs.msg import JointState
import tf
import serial

# Use BOARD mode and correct pins
output_pin_pitch = 15
output_pin_yaw = 33
PWM_FREQ = 200

class ServoSysID:
    def setup_pid(self, kp=1.0, ki=0.0, kd=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.pitch_integral = 0.0
        self.pitch_prev_error = 0.0
        self.yaw_integral = 0.0
        self.yaw_prev_error = 0.0

    def step_pitch_pid(self, dt):
        # Use true feedback (degrees) for PID
        # Convert true pitch (deg) to radians
        feedback_angle = np.deg2rad(self.pitch_true)
        # Compute error in radians
        error = self.ref_pitch - feedback_angle
        self.pitch_integral += error * dt
        derivative = (error - self.pitch_prev_error) / dt if dt > 0 else 0.0
        output = self.kp * error + self.ki * self.pitch_integral + self.kd * derivative
        # Convert output (rad) to duty cycle increment
        self.val_pitch = 32.0 + output
        self.val_pitch = min(max(self.val_pitch, 0.0), 50.0)
        self.p_pitch.ChangeDutyCycle(self.val_pitch)
        self.pitch_prev_error = error

    def step_yaw_pid(self, dt):
        # Use true feedback (degrees) for PID
        feedback_angle = np.deg2rad(self.yaw_true)
        error = self.ref_yaw - feedback_angle
        self.yaw_integral += error * dt
        derivative = (error - self.yaw_prev_error) / dt if dt > 0 else 0.0
        output = self.kp * error - self.ki * self.yaw_integral + self.kd * derivative
        self.val_yaw = 28.5 + output
        self.val_yaw = min(max(self.val_yaw, 0.0), 50.0)
        self.p_yaw.ChangeDutyCycle(self.val_yaw)
        self.yaw_prev_error = error

    def __init__(self, pin_pitch=output_pin_pitch, pin_yaw=output_pin_yaw, freq=PWM_FREQ):
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(pin_pitch, GPIO.OUT, initial=GPIO.HIGH)
        GPIO.setup(pin_yaw, GPIO.OUT, initial=GPIO.HIGH)
        self.p_pitch = GPIO.PWM(pin_pitch, freq)
        self.p_yaw = GPIO.PWM(pin_yaw, freq)
        self.p_pitch.start(32.0)
        self.p_yaw.start(28.5)
        self.val_pitch = 32.0
        self.val_yaw = 28.5
        self.ref_pitch = 0.0
        self.ref_yaw = 0.0
        self.yaw_true = 0.0
        self.pitch_true = 0.0
        self.last_update_time = time.time()
        # ROS target angle integration
        self.pitch_target_angle_msg = None
        self.yaw_target_angle_msg = None
        rospy.Subscriber("/servo_pitch_target", Float64, self.pitch_target_callback)
        rospy.Subscriber("/servo_yaw_target", Float64, self.yaw_target_callback)
        self.joint_state_pub = rospy.Publisher("/camera_joint_states", JointState, queue_size=10)

    def pitch_target_callback(self, msg):
        self.pitch_target_angle_msg = msg.data
        pitch_rad = np.deg2rad(self.pitch_target_angle_msg)
        pitch_rad = np.clip(pitch_rad, -np.pi/3, np.pi/2)
        self.set_reference_pitch(pitch_rad)

    def yaw_target_callback(self, msg):
        self.yaw_target_angle_msg = msg.data
        yaw_rad = np.deg2rad(self.yaw_target_angle_msg)
        yaw_rad = np.clip(yaw_rad, -np.pi/4, np.pi/4)
        self.set_reference_yaw(yaw_rad)

    def set_reference_pitch(self, ref_pitch):
        self.ref_pitch = ref_pitch

    def set_reference_yaw(self, ref_yaw):
        self.ref_yaw = ref_yaw

    def cleanup(self):
        self.p_pitch.stop()
        self.p_yaw.stop()
        GPIO.cleanup()

if __name__ == "__main__":
    servo = ServoSysID()
    ser = serial.Serial('/dev/ttyACM0', 480000000, timeout=0.001)
    import rospy
    rospy.init_node('servo_pid_control_node')
    print("PID Position Control mode (ROS): Enter PID gains (kp ki kd), or press Enter for slower defaults (0.05 0.0 0.0)")
    kp, ki, kd = -1.0, -20.0, 0.0
    servo.setup_pid(kp, ki, kd)
    print("Send target pitch and yaw angles in degrees to /servo_pitch_target and /servo_yaw_target topics.")
    br = tf.TransformBroadcaster()
    rate = rospy.Rate(100)
    try:
        while not rospy.is_shutdown():
            # Read feedback from serial (pitch_true, yaw_true in degrees)
            start_time = time.time()
            try:
                # while ser.in_waiting:
                #     
                line = ser.readline().decode('utf-8').strip()
                pitch_str, yaw_str = line.split("\t")
                servo.pitch_true = float(pitch_str)
                servo.yaw_true = float(yaw_str)
                received_time = rospy.Time.now()
                br.sendTransform(
                (0.0, 0.0, 0.0),
                tf.transformations.quaternion_from_euler(
                    -np.deg2rad(servo.yaw_true),
                    -np.deg2rad(servo.pitch_true),
                    0.0,
                    'ryxz'
                ),
                received_time,
                "/camera_depth_optical_frame",
                "/camera_depth_optical_frame_static"
            )
            except ValueError:
                print("Value error is happening. FInd out why.")
                continue

            now = time.time()
            dt = now - servo.last_update_time
            servo.step_pitch_pid(dt)
            servo.step_yaw_pid(dt)
            servo.last_update_time = time.time()

            # Publish JointState message for yaw and pitch (in radians)
            joint_state_msg = JointState()
            joint_state_msg.header.stamp = received_time
            joint_state_msg.name = ["camera_pitch_joint", "camera_yaw_joint"]
            joint_state_msg.position = [np.deg2rad(servo.pitch_true), np.deg2rad(servo.yaw_true)]
            servo.joint_state_pub.publish(joint_state_msg)

            # Publish TF using actual pitch and yaw in radians
            rate.sleep()
    finally:
        servo.cleanup()