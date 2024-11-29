# WallFollower-RL
Robot wall following using Reinforcement Learning - Gazebo simulation in ROS

## About
This project simulates a triton robot using the Gazebo simulator in ROS to teach it to follow the walls of an enivronment. Two classic reinforcement learning algorithms are implemented - Q-learning and SARSA.

![](https://github.com/kevinantonygomez/WallFollower-RL/blob/main/demo/demo.gif)

## Getting Started
### Install ROS Noetic
Follow the instructions to install ROS Noetic for Ubuntu 20.04 at: http://wiki.ros.org/noetic/Installation/Ubuntu. Install the ros-noetic-desktop-full version.

### Install Other ROS Dependencies
```bash
sudo apt install ros-noetic-gazebo-ros-pkgs ros-noetic-depthimage-to-laserscan ros-noetic-gmapping python3-catkin-tools python3-pip
pip3 install pynput
```

### Create a Catkin Workspace (if none exists)
Follow the instructions at: https://wiki.ros.org/catkin/Tutorials/create_a_workspace

```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/
catkin_make
```

### Clone this repo
```bash
cd ~/catkin_ws/src
git clone https://github.com/kevinantonygomez/WallFollower-RL.git
```
The wall following code is implemented as so:\
'scripts/q_td_train.py' trains the robot using Q-learning\
'scripts/q_td_run.py' tests the robot that was trained using Q-learning\
'scripts/sarsa_train.py' trains the robot using SARSA\
'scripts/sarsa_run.py' tests the robot that was trained using SARSA\

## Running the Simulation

```bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash
roslaunch stingray_sim wall_following.launch choice_and_mode:=x
```
where x can be 1,2,3 or 4.\
1 = q_td_train.py\
2 = q_td_run.py\
3 = sarsa_train.py\
4 = sarsa_run.py\
The generated log files should be available in your '.ros' folder (which should be in your Home directory).

## Generation and storage of q-tables
The scripts folder contains two pickled files for the best pre-learned q-tables for both
algroithms: q_td_q_table.pkl and sarsa_q_table.pkl.\
To use them with the run files, place these files in your '.ros' folder before running q_td_run.py or sarsa_run.py\
After each step of each episode during training, the currently learned q-tables are dumped to the '.ros' folder directly 
as either 'q_td_trained_q_table.pkl' or 'sarsa_trained_q_table.pkl'.\
To test using these files, rename them as 'q_td_q_table.pkl' or 'sarsa_q_table.pkl' before running q_td_run.py or sarsa_run.py