#!/usr/bin/python
# -*- coding: utf-8 -*-
#import sys
#sys.path.append("/opt/ros/indigo/%lib/python2.7/dist-packages")
#sys.path.append("/home/zyb/test_ws/src/stage_ros/srv")
from stage_ros.srv import *

import numpy as np
from numpy import random
from numpy.random import randint
from gym import Env
from gym.envs.classic_control import rendering
from object import *
import math
#ROS denpendence
import rospy
import tf
import tf2_ros
import actionlib
from actionlib_msgs.msg import *
from geometry_msgs.msg import Pose,Point,PoseWithCovarianceStamped,Quaternion, Twist,PoseStamped
from move_base_msgs.msg import MoveBaseActionGoal, MoveBaseGoal
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry,Path
from geometry_msgs.msg import Twist
from visualization_msgs.msg import Marker


SAME_POLICY = False # True if the other agent adopts the same policy learned

coordinate=[[-5,-4],[6,4],[6,6],[-2,2],[0,0]]
pi=3.1415926
#为雷达所检测的十个距离值,雷达最大检测范围为30米

def calcDist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])  #求欧几里得范数	
class ObstacleEnv(Env):
    def __init__(self):
        self.Cr = 0.5     #初始值
        self.count_i = 0  #超参数函数的计数器
        self.linear = 0   #linear,angular为机器人的线速度角速度
        self.angular = 0
        self.success_time = 0
        self.marker_pub = rospy.Publisher('visualization_marker', Marker, queue_size=1)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=5) #发送速度命令
        self.xAnt = 0
        self.yAnt = 0 #坐标更新判断值
        self.obstacles_from_lidar = [30, 30, 30, 30, 30, 30, 30, 30, 30, 30]
        self.Yaw=0
        self.states = np.zeros([1, 14])
        self.robots = []
        self.robots.append(Robot())
        self.goals = np.asarray([[6, 6]])
        rospy.init_node('listener')
        listener = tf.TransformListener()
        rospy.Subscriber('scan', LaserScan, self.get_laser_data)
        rospy.Subscriber('/odom', Odometry, self.get_pose)

    def C_hype(self):
        decay_rate = 0.96  # 衰减率
        global_steps = 500# 总步数
        decay_steps = 1 # 衰减次数 #默认50最终衰减值为0.13,30最终衰减值为0.29,60最终衰减值为0.08,100最终衰减值为0.017,150最终衰减值为0.002,
        if self.count_i >= global_steps/decay_steps:
            self.Cr *= decay_rate
            self.count_i = 0
        return self.Cr

    def get_pose(self, odom):   #获取并更新小车的线速度和角速度,以及画线
        self.linear = odom.twist.twist.linear.x
        self.angular = odom.twist.twist.angular.z
        self.robots[0].pos[0] = odom.pose.pose.position.x
        self.robots[0].pos[1] = odom.pose.pose.position.y
        (r, p, self.Yaw) = tf.transformations.euler_from_quaternion([
            odom.pose.pose.orientation.x,
            odom.pose.pose.orientation.y,
            odom.pose.pose.orientation.z,
            odom.pose.pose.orientation.w
        ])


    def get_laser_data(self,scan=LaserScan()): #获取并更新雷达数据
        bias_correction = 0.5
        self.obstacles_from_lidar[0] = round(scan.ranges[0], 2)-bias_correction
        self.obstacles_from_lidar[1] = round(scan.ranges[1], 2)-bias_correction
        self.obstacles_from_lidar[2] = round(scan.ranges[2], 2)-bias_correction
        self.obstacles_from_lidar[3] = round(scan.ranges[3], 2)-bias_correction
        self.obstacles_from_lidar[4] = round(scan.ranges[4], 2)-bias_correction
        self.obstacles_from_lidar[5] = round(scan.ranges[5], 2)-bias_correction
        self.obstacles_from_lidar[6] = round(scan.ranges[6], 2)-bias_correction
        self.obstacles_from_lidar[7] = round(scan.ranges[7], 2)-bias_correction
        self.obstacles_from_lidar[8] = round(scan.ranges[8], 2)-bias_correction
        self.obstacles_from_lidar[9] = round(scan.ranges[9], 2)-bias_correction

    def calcMinDist(self):
        return min(self.obstacles_from_lidar) #计算机器人离障碍物最小的距离

    def updateState(self, action0):   #每走一步所进行的状态更新
        global pi
        listener = tf.TransformListener()
        self.count_i += 1
        move_cmd = Twist()
        move_cmd.linear.x = action0[0]
        move_cmd.angular.z = action0[1]
        self.cmd_vel_pub.publish(move_cmd) #给小车发送命令
        for id in range(1):
            delta = self.goals[id] - self.robots[id].pos
            dist = np.linalg.norm(delta) #小车距目的地的距离,极坐标的长度
            #print("delta:",delta)
            #ang_world = math.atan2(delta[1], delta[0]) #
        if delta[0] == 0 and delta[1] > 0:
            ang_target_current = pi/2
        elif delta[0] == 0 and delta[1] < 0:
            ang_target_current = -pi/2
        elif delta[0] > 0 and delta[1] >= 0:
            ang_target_current = math.atan(delta[1]/delta[0])
        elif delta[0] < 0 and delta[1] >= 0:
            ang_target_current = pi-math.atan(delta[1]/(-delta[0]))
        elif delta[0] < 0 and delta[1] <= 0:
            ang_target_current = math.atan(delta[1]/delta[0])-pi
        elif delta[0] > 0 and delta[1]<= 0:
            ang_target_current = -math.atan(-(delta[1])/delta[0])
        else:
            print("Error!")
        ang_robot = -(self.Yaw - ang_target_current)

        while ang_robot > pi:
            ang_robot = ang_robot-2 * pi
        while ang_robot < -pi:
            ang_robot = ang_robot+2 * pi
            ang_robot = ang_robot/pi

        self.states[id][0] = self.linear
        self.states[id][1] = self.angular
        self.states[id][2] = dist
        self.states[id][3] = ang_robot
        self.states[id][4] = self.obstacles_from_lidar[0]
        self.states[id][5] = self.obstacles_from_lidar[1]
        self.states[id][6] = self.obstacles_from_lidar[2]
        self.states[id][7] = self.obstacles_from_lidar[3]
        self.states[id][8] = self.obstacles_from_lidar[4]
        self.states[id][9] = self.obstacles_from_lidar[5]
        self.states[id][10] = self.obstacles_from_lidar[6]
        self.states[id][11] = self.obstacles_from_lidar[7]
        self.states[id][12] = self.obstacles_from_lidar[8]
        self.states[id][13] = self.obstacles_from_lidar[9]

    def _step(self, action):
        # 选择动作，更新状态
        self.updateState(action)
        # 计算奖励
        reward, done = self.calcReward()
        return self.states, reward, done, {}

    def _seed(self):
        return

    def calcReward(self):
        self.Cr = self.C_hype()
        #计算目标距离
        dis_robot_to_goal=calcDist(self.robots[0].pos, self.goals[0])
        #print("dis:",dis_robot_to_goal)
        min_dist = self.calcMinDist()
        if min_dist <= 0:
            # print('collision detected')
            reward = -0.5
            done = True
        elif dis_robot_to_goal <= 0.5: #在目标0.2米之内,视为到底目的地
            print('goal reached')
            reward = 1
            done = True
            self.success_time+=1
        else:
            done=False
            reward = self.Cr*(self.disPre_robot_to_goal-dis_robot_to_goal)# 越靠近目的地,奖励分数越高,此地方需设置一个超参数C_hype,目前先默认为0.96
            self.disPre_robot_to_goal=dis_robot_to_goal#如果目标仍在运动.则前一刻的距离更新.
            if 0<min_dist and min_dist<0.2: #如果小车离障碍的距离在0.2米之内,则需要扣分.目的:使小车与障碍物保持安全的距离.
                reward=reward -0.2 - min_dist / 2.0
                #print('closed to obstacle')
        return reward, done

    def _reset(self):
        random.shuffle(coordinate)#把坐标位置乱序
        initial_poses = PoseWithCovarianceStamped()
        initial_poses.pose.pose.position.x = 0
        initial_poses.pose.pose.position.y = 0
        initial_poses.pose.pose.position.z = 0
        initial_poses.pose.pose.orientation.w = 0
        initial_poses.pose.pose.orientation.x = 0
        initial_poses.pose.pose.orientation.y = 0
        initial_poses.pose.pose.orientation.z = 1
        rospy.wait_for_service('reset_positions')
        try:
                val = rospy.ServiceProxy('reset_positions', pose)
                resp1 = val(initial_poses)
        except rospy.ServiceException, e:
            print e
            print("doesn't update initialpose")

        self.goals = np.asarray([[-3.5, 1.5]])
        if self.success_time % 5 == 0 and self.success_time > 0:
            print("time of suceess trial:%d" %(self.success_time))
        self.updateState([0, 0])

        goal_marker = Marker()
        init_marker = Marker()
        goal_marker.id = 1
        init_marker.id = 0
        goal_marker.header.frame_id = "map"
        goal_marker.header.stamp = rospy.Time.now()
        goal_marker.type = Marker.CUBE
        goal_marker.pose.position.x = self.goals[0][0]  ######
        goal_marker.pose.position.y = self.goals[0][1]
        goal_marker.pose.orientation.w = 1.0
        goal_marker.scale.x = 0.1
        goal_marker.scale.y = 0.1
        goal_marker.color.g = 1.0
        goal_marker.color.a = 1.0
        self.marker_pub.publish(goal_marker)

        init_marker.header.frame_id = "map"
        init_marker.header.stamp = rospy.Time.now()
        init_marker.type = Marker.CYLINDER
        init_marker.pose.position.x = initial_poses.pose.pose.position.x
        init_marker.pose.position.y = initial_poses.pose.pose.position.y
        init_marker.pose.orientation.w = 1.0
        init_marker.scale.x = 0.1
        init_marker.scale.y = 0.1
        init_marker.color.r = 1.0
        init_marker.color.a = 1.0
        self.marker_pub.publish(init_marker)

        self.disPre_robot_to_goal = calcDist(self.robots[0].pos, self.goals[0])#如果目标到达目的地或者终止,那么下一次运动时,前一刻的距离应重置为起始距离
        return self.states
