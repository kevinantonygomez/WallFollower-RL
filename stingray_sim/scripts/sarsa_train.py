#!/usr/bin/env python3
import rospy
import numpy
import time
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, GetModelState
import random
import math
import pickle
import tf.transformations

'''
Object that represents the Q-table and policy 
'''
class Q:
    def __init__(self) -> None:
        DIRECT_DIST_COMB_1 = ['front_too_close', 'front_close', 'front_medium', 'front_far']
        DIRECT_DIST_COMB_2 = ['left_close', 'left_medium', 'left_far'] 
        DIRECT_DIST_COMB_3 = ['right_too_close', 'right_close', 'right_medium', 'right_far', 'right_too_far']
        DIRECT_DIST_COMB_4 = ['frontright_close', 'frontright_medium', 'frontright_far']
        self.Q_table = {}
        self.EPISODES = 350
        self.TIME_STEPS = 6000
        self.epsilon = 0.9
        self.d = 0.985
        self.GOAL_STATES = ['front_medium,left_medium,right_medium,frontright_medium', 'front_medium,left_far,right_medium,frontright_medium',
                            'front_medium,left_medium,right_medium,frontright_far', 'front_medium,left_far,right_medium,frontright_far',
                            'front_far,left_medium,right_medium,frontright_medium', 'front_far,left_far,right_medium,frontright_medium',
                            'front_far,left_medium,right_medium,frontright_far', 'front_far,left_far,right_medium,frontright_far',
                            'front_medium,left_medium,right_close,frontright_medium', 'front_medium,left_far,right_close,frontright_medium',
                            'front_medium,left_medium,right_close,frontright_far', 'front_medium,left_far,right_close,frontright_far',
                            'front_far,left_medium,right_close,frontright_medium', 'front_far,left_far,right_close,frontright_medium',
                            'front_far,left_medium,right_close,frontright_far', 'front_far,left_far,right_close,frontright_far']
        
        '''Create all possible states as keys to dict 'Q_table' and store the values corresponding to
        the actions: move_forward, turn_left, turn_right.'''
        for a in DIRECT_DIST_COMB_1:
            for b in DIRECT_DIST_COMB_2:
                for c in DIRECT_DIST_COMB_3:
                    for d in DIRECT_DIST_COMB_4:
                        state_key = f"{a},{b},{c},{d}"
                        self.Q_table[state_key] = [0, 0, 0] # values for [move_forward, turn_left, turn_right]
                        if (a == 'front_close' and d == 'frontright_close') or \
                            (a == 'front_too_close' and d == 'frontright_close'):
                            self.Q_table[state_key] = [0, 0.1, 0] 
                                
        print(f"q_init: {self.Q_table}")

    def policy(self, n, state_key):
        q_vals = self.Q_table[state_key] # get all values for current state
        rand_num = random.uniform(0,1)
        if rand_num > self.epsilon:
            action_num = numpy.argmax(q_vals) # choose action with highest value
            print(f"best action: {action_num} for {state_key}")
        else:
            action_num = random.choice([0,1,2]) # randomly choose an index num and therefore an action
            print(f"rand action: {action_num} for {state_key}")

        self.epsilon = self.epsilon*(self.d)**(n)
        return action_num
    
    def get_reward(self, curr_state):
        if any(state in curr_state for state in ['front_too_close', 'right_too_far', 'right_too_close', 'left_close']):
            print("reward = -1")
            return -1
        elif curr_state in self.GOAL_STATES:
            print("reward = 0.5")
            return 0.5
        else:
            print("reward = 0")
            return 0
    
    def update_q_table(self, prev_state, curr_state, action, new_action, alpha=0.2, gamma=0.8):
        old_val = self.Q_table[prev_state][action]
        reward = self.get_reward(prev_state)
        future_val = self.Q_table[curr_state][new_action]
        self.Q_table[prev_state][action] = old_val + (alpha * (reward + (gamma * future_val) - old_val))


    def train(self, triton, gazebo):
        STUCK_TOL = 0.003
        STUCK_LIMIT = 3
        GOOD_POLICY_LIMIT = 1000
        print("entered training mode!")
        while True:
            if triton.dists == None:
                time.sleep(1)
            else:
                break

        for n in range(1, self.EPISODES+1):
            stuck_check = dict()
            print(f"\n ----- {n}/{self.EPISODES+1} ----- \n")
            stuck_count = 0
            good_policy_count = 0
            self.epsilon = 0.9
            gazebo.set_rand_model_state()
            curr_model_state = gazebo.get_model_state()
            curr_state = triton.get_current_state()

            action = self.policy(n, curr_state)
            
            for i in range(self.TIME_STEPS):
                if action == 0:
                    triton.move_forward()
                elif action == 1: 
                    triton.turn_left()
                elif action == 2: 
                    triton.turn_right()
                else:
                    print("SOMETHING WENT WRONG. Stopping triton")
                    triton.stop()

                # observe new state and select new action
                actual_curr_model_state = gazebo.get_model_state()
                actual_curr_state =  triton.get_current_state()
                new_action = self.policy(n, actual_curr_state)

                self.update_q_table(curr_state, actual_curr_state, action, new_action) # curr_state here is now the previous state

                # curr_model_state is now the prev model state
                if math.isclose(curr_model_state.pose.position.x, actual_curr_model_state.pose.position.x, rel_tol=STUCK_TOL) and\
                math.isclose(curr_model_state.pose.position.y, actual_curr_model_state.pose.position.y, rel_tol=STUCK_TOL):
                    stuck_count += 1
                    stuck_check[i] = 1
                    if i > 10:
                        temp_stuck_check = 0
                        for j in range(1, 10):
                            if stuck_check[i-j] == 1: 
                                temp_stuck_check +=1
                        
                        if temp_stuck_check >= 3:
                            print("Stuck. Breaking...")
                            break
                else:
                    stuck_check[i] = 0
                    stuck_count = 0

                if stuck_count >= STUCK_LIMIT:
                    print("Stuck. Breaking...")
                    break

                roll, pitch, yaw = tf.transformations.euler_from_quaternion([actual_curr_model_state.pose.orientation.x, actual_curr_model_state.pose.orientation.y,
                                                                            actual_curr_model_state.pose.orientation.z, actual_curr_model_state.pose.orientation.w])
                
                if abs(roll) > 0.4 or abs(pitch) > 0.4:
                    print("Triton tripped. Breaking...")
                    break

                if actual_curr_model_state.pose.position.z > 0.5: # handle instances where robot glitches and moves up
                    print("Triton going up. Breaking...")
                    break

                if curr_state in self.GOAL_STATES:
                    good_policy_count += 1
                    print(f"GOOD POLICY {good_policy_count}")
                
                if good_policy_count >= GOOD_POLICY_LIMIT:
                    print(f"GOOD POLICY found. Breaking...")
                    break

                # transit states
                curr_model_state = actual_curr_model_state
                curr_state = actual_curr_state
                action = new_action
                print("\n")

                with open("sarsa_trained_q_table.pkl", "wb+") as f:
                    pickle.dump(self.Q_table, f)

                triton.rate.sleep()

 
'''
Object that represents the Gazebo env. Used to set the Triton robots inital pose
'''
class Gazebo:
    def __init__(self) -> None:
        self.model_state = ModelState()
        self.model_state.model_name = 'triton'

    def get_model_state(self):
        rospy.wait_for_service('gazebo/get_model_state')
        get_model_state_service = rospy.ServiceProxy('gazebo/get_model_state', GetModelState)
        return get_model_state_service(self.model_state.model_name, 'world')

        
    def set_model_state(self):
        rospy.wait_for_service('gazebo/set_model_state')
        # set robot at bottom left position of the map initally to follow the wall
        self.model_state.pose.position.x = 3.5
        self.model_state.pose.position.y = -3.5
        self.model_state.pose.position.z = 0
        set_model_state_service = rospy.ServiceProxy('gazebo/set_model_state', SetModelState)
        resp = set_model_state_service(self.model_state)
        if not resp.success:
            print(f'Set model state failed!: {resp}')
    
    def set_rand_model_state(self):
        rospy.wait_for_service('gazebo/set_model_state')
        x = random.uniform(-3.5, 3.5) # avoid spawing on/in walls
        y = random.uniform(-3.5, 3.5)
        self.model_state.pose.position.x = x 
        self.model_state.pose.position.y = y
        self.model_state.pose.position.z = 0

        q = tf.transformations.quaternion_from_euler(0,0,1.57)
        self.model_state.pose.orientation.x = q[0]
        self.model_state.pose.orientation.y = q[1]
        self.model_state.pose.orientation.z = q[2]
        self.model_state.pose.orientation.w = q[3]

        set_model_state_service = rospy.ServiceProxy('gazebo/set_model_state', SetModelState)
        resp = set_model_state_service(self.model_state)
        if not resp.success:
            print(f'Set model state failed!: {resp}')


'''
Object that represents the Triton robot. Handles LiDAR detection and movement
'''
class Triton:
    def __init__(self, q_agent: Q, gazebo: Gazebo) -> None:
        rospy.init_node("sarsa_train")
        self.dists = None
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.laser_sub = rospy.Subscriber("/scan", LaserScan, self.laserscan_callback)
        self.rate = rospy.Rate(10)
        q_agent.train(self, gazebo)
        while not rospy.is_shutdown():
            self.rate.sleep() 

    def dist_group(self, val, direction):
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
            elif val >= 0.4 and val < 0.6:
                return 'close'
            if val >= 0.6 and val < 1.3:
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
        print(f"created {state_key} for {dists}")
        return state_key

    def laserscan_callback(self, data):
        # LiDAR readings
        front = min(min(data.ranges[0:20]), min(data.ranges[340:]))
        frontright = min(data.ranges[300:330])
        right = min(data.ranges[250:330])
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