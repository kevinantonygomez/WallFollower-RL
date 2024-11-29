#!/usr/bin/env python3
import rospy
import numpy
import time
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
import math
import pickle
import tf.transformations

'''
Object that represents the Q-table and policy 
'''

class Q:
    def __init__(self) -> None:
        print("Loading Q table")
        with open("sarsa_q_table.pkl", "rb") as f:
            self.Q_table = pickle.load(f)

    def policy(self, state_key):
        q_vals = self.Q_table[state_key] # get all values for current state
        print(q_vals)
        action_num = numpy.argmax(q_vals) # choose action with highest value
        return action_num

    def run(self, triton, gazebo):
        print("Entered Sarsa running mode!")
        gazebo.set_model_state()

        while True:
            if triton.dists == None:
                time.sleep(1)
            else:
                break

        while True:
            curr_state = triton.get_current_state()
            action = self.policy(curr_state)
            if action == 0:
                triton.move_forward()
            elif action == 1: 
                triton.turn_left()
            elif action == 2: 
                triton.turn_right()
            else:
                print("SOMETHING WENT WRONG. Stopping triton")
                triton.stop()

            triton.rate.sleep()

 
'''
Object that represents the Gazebo env. Used to set the Triton robots inital pose
'''
class Gazebo:
    def __init__(self) -> None:
        self.model_state = ModelState()
        self.model_state.model_name = 'triton'
        
    def set_model_state(self):
        rospy.wait_for_service('gazebo/set_model_state')
        self.model_state.pose.position.x = -3.6
        self.model_state.pose.position.y = -3.6
        self.model_state.pose.position.z = 0

        set_model_state_service = rospy.ServiceProxy('gazebo/set_model_state', SetModelState)
        resp = set_model_state_service(self.model_state)
        if not resp.success:
            print(f'Set model state failed!: {resp}')


'''
Object that represents the Triton robot. Handles LiDAR detection and movement
'''
class Triton:
    def __init__(self, q_agent: Q, gazebo: Gazebo) -> None:
        rospy.init_node("sarsa_run")
        self.dists = None
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.laser_sub = rospy.Subscriber("/scan", LaserScan, self.laserscan_callback)
        self.rate = rospy.Rate(10)
        q_agent.run(self, gazebo)
        while not rospy.is_shutdown():
            self.rate.sleep() 

    def dist_group(self, val, direction):
        # categorize distance
        if direction == 'frontright':
            if math.isinf(val) or val < 0.45:
                return 'close'
            if val >= 0.45 and val < 0.75:
                return 'medium'
            else:
                return 'far'
            
        elif direction == 'front':
            if math.isinf(val) or val < 0.4:
                return 'too_close'
            elif val >= 0.4 and val < 0.5:
                return 'close'
            if val >= 0.5 and val < 1:
                return 'medium'
            else:
                return 'far'
            
        elif direction == 'left':
            if math.isinf(val) or val < 0.5:
                return 'close'
            if val >= 0.5 and val < 1:
                return 'medium'
            else:
                return 'far'
            
        elif direction == 'right':
            if math.isinf(val) or val < 0.3:
                return 'too_close'
            elif val >= 0.3 and val < 0.4:
                return 'close'
            if val >= 0.4 and val < 0.5:
                return 'medium'
            else:
                return 'far'
            
    def get_current_state(self) -> str:
        # create current state key
        state_key = ""
        dists = self.dists
        for i, direction in enumerate(['front', 'left', 'right', 'frontright']):
            state_key += direction + "_" + self.dist_group(dists[i], direction)
            if i != 3:
                state_key += ","
        print(f"{state_key} for {dists}",end=" ")
        return state_key

    def laserscan_callback(self, data):
        # LiDAR readings
        front = min(min(data.ranges[0:30]), min(data.ranges[330:]))
        frontright = min(data.ranges[300:330])
        right = min(data.ranges[260:330])
        left = min(data.ranges[60:120])
        self.dists = [front, left, right, frontright]

    def move_forward(self):
        print("Moving forward")
        cmd = Twist()
        cmd.angular.z = 0
        cmd.linear.x = 0.25
        cmd.linear.y = 0
        self.vel_pub.publish(cmd)

    def turn_left(self):
        print("Turning left")
        cmd = Twist()
        cmd.angular.z = 0.4
        cmd.linear.x = 0.25
        cmd.linear.y = 0
        self.vel_pub.publish(cmd)

    def turn_right(self):
        print("Turning right")
        cmd = Twist()
        cmd.angular.z = -0.4
        cmd.linear.x = 0.25
        cmd.linear.y = 0
        self.vel_pub.publish(cmd)
        

    def stop(self):
        print("Stopping Triton")
        cmd = Twist()
        cmd.angular.z = 0
        cmd.linear.x = 0
        cmd.linear.y = 0
        self.vel_pub.publish(cmd)


if __name__ == '__main__':
    try:
        time.sleep(5) # wait a bit to ensure the Gazebo GUI is visible before running
        gazebo = Gazebo()
        q_agent = Q()
        Triton(q_agent, gazebo) 
    except Exception as e:
        print(e)
