#!/usr/bin/env python
# -*- coding: utf-8 -*-
#导入库
import rospy
import tf
import csv
import math
from tf2_msgs.msg import TFMessage
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal,MoveBaseActionGoal
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist,PoseWithCovarianceStamped,Quaternion
from visualization_msgs.msg import Marker
from nav_msgs.msg import Odometry
import actionlib
from stage_ros.srv import *
import random


#随机小车初始位置和目标位置,不会出现在障碍物里面,有效区域有五个
def poker_shuffle():
    poker=random.shuffle([round(random.uniform(-5.5, -3.5),3),round(random.uniform(1.5, 4.5),3)], \
        [round(random.uniform(-2, 2),3),round(random.uniform(3.5, 4.5),3)], \
        [round(random.uniform(2.5, 5.5),3),round(random.uniform(-0.5, 4.5),3)], \
        [round(random.uniform(-2.5, 1.5),3),round(random.uniform(-0.5, 1.5),3)], \
        [round(random.uniform(-5.5, 5.5),3),round(random.uniform(-4.5,-3),3)] )
    #return poker[0] #如果想让初始位置和目标位置可以出现在同一个有效区域内,使用这句
    #return poker[0],poker[1]#如果想让初始位置和目标位置不能出现在同一个有效区域内,使用这句

def get_data(scan):#获取雷达信息
    global lidar_obstacle
    lidar_obstacle = scan.ranges
def get_pose(odom):
    global line_vel,ang_vel,position,Yaw#获取并更新小车的线速度和角速度,以及位姿
    line_vel=odom.twist.twist.linear.x
    ang_vel=odom.twist.twist.angular.z
    position[0]=odom.pose.pose.position.x
    position[1]=odom.pose.pose.position.y
    (r, p, Yaw) = tf.transformations.euler_from_quaternion([odom.pose.pose.orientation.x,odom.pose.pose.orientation.y,odom.pose.pose.orientation.z,odom.pose.pose.orientation.w])
    #print(odom)

if __name__=='__main__':
    position = [0, 0]  # 小车当前位置
    pre_position = [0, 0]  # 小车前一时时位置
    rospy.init_node('print_14_dim')
    pi=3.1415926
    rate = rospy.Rate(10.0)
    lidar_obstacle=[0,0,0,0,0,0,0,0,0,0]
    line_vel=0
    ang_vel=0
    Yaw=0
    #定义初始位置
    initial_poses = PoseWithCovarianceStamped()
    initial_poses.header.frame_id= 'map'
    initial_poses.pose.pose.position.x = 0
    initial_poses.pose.pose.position.y = 0
    initial_poses.pose.pose.orientation.z = 1
    initial_poses.pose.pose.orientation.w = 0
    try:
        val = rospy.ServiceProxy('reset_positions', pose)#初始化起始点位姿
        resp1 = val(initial_poses)
        # print resp1.success
    except rospy.ServiceException, e:
        print e
        print("doesn't update initialpose")
    # 定义目标点位姿
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = 'map'
    goal.target_pose.pose.position.x = round(random.uniform(-1.5, -3),3)#随机目标点
    goal.target_pose.pose.position.y = round(random.uniform(-3, -4.5),3)
    print("x,y:",goal.target_pose.pose.position.x,goal.target_pose.pose.position.y)
    goal.target_pose.pose.orientation.w = 1
    move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)
    move_base.wait_for_server(rospy.Duration(60))
    move_base.send_goal(goal)#发送目标位置,小车开始运动

    #显示目标位置
    goal_marker = Marker()
    goal_marker.header.frame_id = 'map'
    goal_marker.type = Marker.CUBE
    goal_marker.pose.position.x = goal.target_pose.pose.position.x
    goal_marker.pose.position.y = goal.target_pose.pose.position.y
    goal_marker.scale.x = 0.1
    goal_marker.scale.y = 0.1
    goal_marker.color.g = 1.0
    goal_marker.color.a = 1.0
    marker_pub = rospy.Publisher('visualization_marker', Marker, queue_size=2)
    marker_pub.publish(goal_marker)


    #订阅和发布的节点

    listener = tf.TransformListener()
    rospy.Subscriber('odom', Odometry, get_pose)
    rospy.Subscriber('scan',LaserScan,get_data)
    #rospy.Subscriber('move_base/goal',MoveBaseActionGoal,get_goal)
    #rospy.Subscriber('initialpose', PoseWithCovarianceStamped,initial_poses)



    for i in range(100000):
        #如果小车开始移动(step:0.03m),则运行以下程序.(现在只考虑小车的位置,以后会考虑小车的姿态)
        if math.sqrt((position[0] - pre_position[0])**2+ (position[1] - pre_position[1])**2)>=0.0:
            #print("pre_position", pre_position)
            #print("postion", position)
            pre_position[0] = position[0]  # 更新前一时刻的值
            pre_position[1] = position[1]
            scan_ranges = list(lidar_obstacle)  # 10-dim from laser
            #得到小车到目标点的极坐标
            diff_x=goal.target_pose.pose.position.x-position[0]
            diff_y=goal.target_pose.pose.position.y-position[1]
            distance_to_target=math.sqrt(diff_x**2+diff_y**2)
            if diff_x==0 and diff_y>0:
                ang_target_current=pi/2
            elif diff_x==0 and diff_y<0:
                ang_target_current=-pi/2
            elif diff_x>0 and diff_y>=0:
                ang_target_current=math.atan(diff_y/diff_x)
            elif diff_x<0 and diff_y>=0:
                ang_target_current=pi-math.atan(diff_y/(-diff_x))
            elif diff_x<0 and diff_y<=0:
                ang_target_current=math.atan(diff_y/diff_x)-pi
            elif diff_x>0 and diff_y<=0:
                ang_target_current=-math.atan(-(diff_y)/diff_x)
            else:
                print("Error!")
                #print("YAW:",y)
                #print("theta:",ang_target_current)
                #print("YAW:",y)
            ab_ang=-(Yaw-ang_target_current)
            if ab_ang>pi:
                ab_ang=ab_ang-2*pi
            if ab_ang<-pi:
                ab_ang=ab_ang+2*pi
            ab_ang=ab_ang/pi
            #print("angular:",ab_ang)

            #保存小车数据
            # print(line_vel,ang_vel)
            text=list()
            text.append(round(line_vel, 3))
            text.append(round(ang_vel, 3))
            text.append(round(distance_to_target, 3))
            text.append(round(ab_ang, 3))
            for i in range(10):
                text.append(round(scan_ranges[i], 3))
            #print(text)
            with open("/home/ld/data/data_train0.03_lidar5.csv", "a") as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(text)
                #print("writing succeed")

           #如果小车到达目标点(离目标距离0.5米),更新并显示随机目标点,最后重置起始位置
        if math.hypot(position[0] - goal.target_pose.pose.position.x, position[1] - goal.target_pose.pose.position.y)<=0.5:
            pre_position=[0,0]
            goal.target_pose.pose.position.x = round(random.uniform(-1.5, -3), 3)
            goal.target_pose.pose.position.y = round(random.uniform(-3, -4.5), 3)
            print("x,y:", goal.target_pose.pose.position.x, goal.target_pose.pose.position.y)
            goal_marker.pose.position.x = goal.target_pose.pose.position.x
            goal_marker.pose.position.y = goal.target_pose.pose.position.y
            move_base.send_goal(goal)
            marker_pub.publish(goal_marker)
            rospy.loginfo("goal achives...")
            #rospy.wait_for_service('reset_positions')
            try:
                val = rospy.ServiceProxy('reset_positions', pose)
                resp1 = val(initial_poses)
                # print resp1.success
            except rospy.ServiceException, e:
                print e
                print("doesn't update initialpose")
        rate.sleep()#程序运行一次的频率

