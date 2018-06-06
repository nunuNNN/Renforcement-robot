#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from gym.envs.classic_control import rendering
from math import *

class GeomContainer(rendering.Geom):
    def __init__(self, geom, collider_func=None, pos_x=0, pos_y=0, angle=0):
        rendering.Geom.__init__(self)
        self.geom = geom
        self.collider_func = collider_func
        self.pos = np.asarray([pos_x, pos_y], dtype=np.float64)
        assert self.pos.shape == (2,), 'Invalid pos-array shape'
        self.angle = angle
        self.abs_pos = np.copy(self.pos)
        self.abs_angle = self.angle
        self.trans = rendering.Transform()
        self.segments_cache = None
        self.add_attr(self.trans)
        self.radius = 0
    
    def setRadius(self, radius):
        self.radius = radius
    
    def getRadius(self):
        return self.radius
    
    def getAbsPose(self):
        return self.abs_pos

    def render(self):
        self.geom._color = self._color
        self.geom.attrs = self.attrs
        self.geom.render()
    
    def setPos(self, pos_x, pos_y):
        self.pos[:] = pos_x, pos_y
        self.update()

    def moveXY(self, diffX, diffY):
        self.setPos(self.pos[0] + diffX, self.pos[1] + diffY)

    def move(self, v):
        self.moveXY(v * np.cos(self.angle), v * np.sin(self.angle))
    
    def setAngle(self, angle, deg=False):
        self.angle = angle if not deg else np.deg2rad(angle)
        self.update()

    def rotate(self, diff_angle, deg=False):
        self.setAngle(self.angle + diff_angle if not deg else np.deg2rad(diff_angle))
    
    def update(self):
        self.trans.set_translation(*self.pos)
        self.trans.set_rotation(self.angle)
        self.abs_pos[:] = 0.0
        self.abs_angle = 0.0
        prev_angle = 0.0
        for attr in reversed(self.attrs):
            if isinstance(attr, rendering.Transform):
                self.abs_pos += rotate(attr.translation, prev_angle)
                self.abs_angle += attr.rotation
                prev_angle = attr.rotation
        self.segments_cache = None

    def getSegments(self):
        if self.segments_cache is None:
            self.segments_cache = self.collider_func(self.abs_pos, self.abs_angle)
        return self.segments_cache

    def getIntersections(self, segment_list):
        if self.collider_func is None:
            return []
        intersections = []
        for collider_segment in self.getSegments():
            for segment in segment_list:
                intersection = collider_segment.getIntersection(segment)
                if intersection is not None:
                    intersections.append(intersection)
        return intersections

    def getGeomList(self):
        return [self]

class Segment():
    def __init__(self, start=(0, 0), end=(0, 0)):
        self.start = np.asarray(start)
        self.end = np.asarray(end)

    def diffX(self):
        return self.end[0] - self.start[0]

    def diffY(self):
        return self.end[1] - self.start[1]

    def updateStartEnd(self, start, end):
        self.start[:] = start
        self.end[:] = end
        
    def getIntersection(self, segment):
        def checkIntersectionLs(line, segment):
            l = line.end - line.start
            p1 = segment.start - line.start
            p2 = segment.end - line.start
            return (p1[0]*l[1] - p1[1]*l[0] > 0) ^ (p2[0]*l[1] - p2[1]*l[0] > 0) # TODO: sign==0
        def checkIntersectionSs(seg1, seg2):
            return checkIntersectionLs(line=seg1, segment=seg2) and checkIntersectionLs(line=seg2, segment=seg1)
        s1, s2 = self, segment
        if checkIntersectionSs(s1, s2):
            r = (s2.diffY() * (s2.start[0] - s1.start[0]) - s2.diffX() * \
                (s2.start[1] - s1.start[1])) / (s1.diffX() * s2.diffY() - s1.diffY() * s2.diffX())
            return (1 -r) * s1.start + r * s1.end
        else:
            return None

class Wall(GeomContainer):
    def __init__(self, start, end, color, **kwargs):
        vertice = [start, end, end + [1, 1], start + [1, 1]]
        GeomContainer.__init__(self, rendering.make_polyline(vertice), 
                collider_func=self.collider_func, **kwargs)
        self.set_color(*color)
        self.wall_segment = Segment(start, end)
        
    def setPos(self, pos_x, pos_y):
        pass

    def setAngle(self, angle, deg=False):
        pass

    def collider_func(self, *args):
        return [self.wall_segment]

def rotate(pos_array, angle, deg=False):
    pos_array = np.asarray(pos_array)
    if deg:
        angle = np.deg2rad(angle)
    c, s = np.cos(angle), np.sin(angle)
    rotation_matrix = np.asarray([[c, -s], [s, c]])
    return np.dot(rotation_matrix, pos_array.T).T

_DISTANCE_SENSOR_MAX_DISTANCE = 200

class Sensor(GeomContainer):
    def __init__(self, geom, **kwargs):
        GeomContainer.__init__(self, geom, **kwargs)
        self.value = None

    def getNearestPoint(self, pos_list):
        sorted_pos_list = sorted(pos_list, key=lambda pos: \
                            np.linalg.norm(pos - self.abs_pos, ord=2))
        return sorted_pos_list[0]

    def detect(self, obstacles):
        raise NotImplementedError()

    def setSensorVal(self, value):
        self.value = value

class DistanceSensor(Sensor):
    def __init__(self, geom, **kwargs):
        Sensor.__init__(self, geom, **kwargs)
        self.ray_geom = rendering.Line()
        self.ray_geom.set_color(1, 0.5, 0.5)
        self.effect_geom = GeomContainer(rendering.make_circle(radius=5, filled=False))
        self.effect_geom.set_color(1, 0.5, 0.5)
        self.intersection_pos = [0, 0]
        self.max_distance = _DISTANCE_SENSOR_MAX_DISTANCE
        self._ray_segment = Segment()
        self.updateRaySegment()

    def render(self):
        Sensor.render(self)
        self.ray_geom.start = self.abs_pos
        self.ray_geom.end = self.intersection_pos
        self.ray_geom.render()
        self.effect_geom.setPos(*self.intersection_pos)
        self.effect_geom.render()

    def getGeomList(self):
        return Sensor.getGeomList(self) + [self.ray_geom]

    def updateRaySegment(self):
        self._ray_segment.updateStartEnd(self.abs_pos, self.abs_pos + \
                rotate([self.max_distance, 0], self.abs_angle))
    
    def detect(self, obstacles):
        self.updateRaySegment()
        intersections = []
        for obs in obstacles:
            intersections += obs.getIntersections([self._ray_segment])
        if len(intersections) > 0:
            intersection_pos = self.getNearestPoint(intersections)
            distance = np.linalg.norm(self.intersection_pos - self.abs_pos, ord=2)
        else:
            intersection_pos = self._ray_segment.end
            distance = self.max_distance
        self.intersection_pos = intersection_pos
        self.setSensorVal(distance)

_NUM_DISTANCE_SENSOR = 8

class Robot(GeomContainer):
    def __init__(self, radius=10, maxVel=1.5, sensor_enabled=False, **kwargs):
        geom = rendering.make_circle(radius)
        collider_func = None
        GeomContainer.__init__(self, geom, collider_func=collider_func)
        self.radius = radius
        self.vel = np.zeros(2)
        self.max_vel = maxVel
        self.set_color(0, 0, 1)
        self.sensors = []
        if sensor_enabled:
            for i in range(_NUM_DISTANCE_SENSOR):
                dist_sensor = DistanceSensor(rendering.make_circle(5))
                dist_sensor.set_color(1, 0, 0)
                dist_sensor.setPos(*(rotate([self.radius, 0], 360 / _NUM_DISTANCE_SENSOR * i, deg=True)))
                dist_sensor.setAngle(360 / _NUM_DISTANCE_SENSOR * i, True)
                dist_sensor.add_attr(self.trans)
                self.sensors.append(dist_sensor)

    def render(self):
        GeomContainer.render(self)
        for sensor in self.sensors:
            sensor.render()

    def getGeomList(self):
        return GeomContainer.getGeomList(self) + self.sensors

    def update(self):
        GeomContainer.update(self)
        for sensor in self.sensors:
            sensor.update()

    def updateSensors(self, visible_objects):
        for sensor in self.sensors:
            sensor.detect(visible_objects)
    
    def getSensorVals(self):
        return [sensor.value for sensor in self.sensors]
