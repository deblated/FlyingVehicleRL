import pymunk
import pymunk.pygame_util
from pymunk import Vec2d
import numpy as np
import pygame


class FlyingVehicle:

    def __init__(self, x, y, angle, height, width, mass_f, mass_l, mass_r, space):
        distance_between_joints = height / 2 - 3
        self.vehicle_radius = width / 2 - height / 2

        self.frame_shape = pymunk.Poly.create_box(None, size=(width, height / 2))
        frame_moment_of_inertia = pymunk.moment_for_poly(mass_f, self.frame_shape.get_vertices())

        frame_body = pymunk.Body(mass_f, frame_moment_of_inertia, body_type=pymunk.Body.DYNAMIC)
        frame_body.position = x, y
        frame_body.angle = angle

        self.frame_shape.body = frame_body
        self.frame_shape.sensor = True
        self.frame_shape.color = pygame.Color((0, 76, 153))

        space.add(frame_body, self.frame_shape)

        self.left_motor_shape = pymunk.Poly.create_box(None, size=(height, height))
        left_motor_moment_of_inertia = pymunk.moment_for_poly(mass_l, self.left_motor_shape.get_vertices())

        left_motor_body = pymunk.Body(mass_l, left_motor_moment_of_inertia, body_type=pymunk.Body.DYNAMIC)
        left_motor_body.position = np.cos(angle + np.pi) * self.vehicle_radius + x, np.sin(
            angle + np.pi) * self.vehicle_radius + y
        left_motor_body.angle = angle

        self.left_motor_shape.body = left_motor_body
        self.left_motor_shape.sensor = True
        self.left_motor_shape.color = pygame.Color((25, 0, 51))

        space.add(left_motor_body, self.left_motor_shape)

        self.right_motor_shape = pymunk.Poly.create_box(None, size=(height, height))
        right_motor_moment_of_inertia = pymunk.moment_for_poly(mass_r, self.right_motor_shape.get_vertices())

        right_motor_body = pymunk.Body(mass_r, right_motor_moment_of_inertia, body_type=pymunk.Body.DYNAMIC)
        right_motor_body.position = np.cos(angle) * self.vehicle_radius + x, np.sin(angle) * self.vehicle_radius + y
        right_motor_body.angle = angle

        self.right_motor_shape.body = right_motor_body
        self.right_motor_shape.sensor = True
        self.right_motor_shape.color = pygame.Color((25, 0, 51))

        space.add(right_motor_body, self.right_motor_shape)

        self.left_1 = pymunk.PivotJoint(
            self.left_motor_shape.body,
            self.frame_shape.body,
            (-distance_between_joints, 0),
            (-self.vehicle_radius - distance_between_joints, 0))
        self.left_1.error_bias = 0

        self.left_2 = pymunk.PivotJoint(
            self.left_motor_shape.body,
            self.frame_shape.body,
            (0, 0),
            (-self.vehicle_radius, 0))
        self.left_2.error_bias = 0

        self.left_3 = pymunk.PivotJoint(
            self.left_motor_shape.body,
            self.frame_shape.body,
            (distance_between_joints, 0),
            (-self.vehicle_radius + distance_between_joints, 0))
        self.left_3.error_bias = 0

        space.add(self.left_1)
        space.add(self.left_2)
        space.add(self.left_3)

        self.right_1 = pymunk.PivotJoint(
            self.right_motor_shape.body,
            self.frame_shape.body,
            (-distance_between_joints, 0),
            (self.vehicle_radius - distance_between_joints, 0))
        self.right_1.error_bias = 0

        self.right_2 = pymunk.PivotJoint(
            self.right_motor_shape.body,
            self.frame_shape.body,
            (0, 0),
            (self.vehicle_radius, 0))
        self.right_2.error_bias = 0

        self.right_3 = pymunk.PivotJoint(
            self.right_motor_shape.body,
            self.frame_shape.body,
            (distance_between_joints, 0),
            (self.vehicle_radius + distance_between_joints, 0))
        self.right_3.error_bias = 0

        space.add(self.right_1)
        space.add(self.right_2)
        space.add(self.right_3)
