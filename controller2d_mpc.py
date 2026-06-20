#!/usr/bin/env python3

"""
2D Controller Class to be used for the CARLA waypoint follower demo.
"""

import cutils
import numpy as np
import math

class Controller2D(object):
    def __init__(self, waypoints):
        self.vars                = cutils.CUtils()
        self._current_x          = 0
        self._current_y          = 0
        self._current_yaw        = 0
        self._current_speed      = 0
        self._desired_speed      = 0
        self._current_frame      = 0
        self._current_timestamp  = 0
        self._start_control_loop = False
        self._set_throttle       = 0
        self._set_brake          = 0
        self._set_steer          = 0
        self._waypoints          = waypoints
        self._conv_rad_to_steer  = 180.0 / 70.0 / np.pi
        self._pi                 = np.pi
        self._2pi                = 2.0 * np.pi
        self._wheelbase          = 2.875   # meters (CARLA Model 3)
        self.vars.create_var('v_error_int', 0.0)
        self.vars.create_var('v_error_prev', 0.0)
        self.vars.create_var('t_prev', 0.0)
        self.vars.create_var('v_previous', 0.0)
        self.vars.create_var('closest_idx_prev', 0)
        self.vars.create_var('delta_prev', 0.0)
        self.vars.create_var('a_prev', 0.0)
        self.vars.create_var('throttle_prev', 0.0)
        self.vars.create_var('brake_prev', 0.0)

    def update_values(self, x, y, yaw, speed, timestamp, frame):
        self._current_x         = x
        self._current_y         = y
        self._current_yaw       = yaw
        self._current_speed     = speed
        self._current_timestamp = timestamp
        self._current_frame     = frame
        if self._current_frame:
            self._start_control_loop = True

    def update_desired_speed(self):
        min_idx       = 0
        min_dist      = float("inf")
        desired_speed = 0
        for i in range(len(self._waypoints)):
            dist = np.linalg.norm(np.array([
                    self._waypoints[i][0] - self._current_x,
                    self._waypoints[i][1] - self._current_y]))
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        if min_idx < len(self._waypoints)-1:
            desired_speed = self._waypoints[min_idx][2]
        else:
            desired_speed = self._waypoints[-1][2]
        self._desired_speed = desired_speed

    def update_waypoints(self, new_waypoints):
        self._waypoints = new_waypoints

    def get_commands(self):
        return self._set_throttle, self._set_steer, self._set_brake

    def set_throttle(self, input_throttle):
        # Clamp the throttle command to valid bounds
        throttle           = np.fmax(np.fmin(input_throttle, 1.0), 0.0)
        self._set_throttle = throttle

    def set_steer(self, input_steer_in_rad):
        # Convert radians to [-1, 1]
        input_steer = self._conv_rad_to_steer * input_steer_in_rad

        # Clamp the steering command to valid bounds
        steer           = np.fmax(np.fmin(input_steer, 1.0), -1.0)
        self._set_steer = steer

    def set_brake(self, input_brake):
        # Clamp the steering command to valid bounds
        brake           = np.fmax(np.fmin(input_brake, 1.0), 0.0)
        self._set_brake = brake


    def cross_track_error(self, x, y, x1, y1, x2, y2):
        # Path vector
        vx = x2 - x1
        vy = y2 - y1

        # Vector from waypoint 1 to vehicle
        wx = x - x1
        wy = y - y1

        # 2D cross product (scalar)
        cross = vx * wy - vy * wx

        # Norm of path vector
        norm = math.sqrt(vx*vx + vy*vy)

        # Signed CTE
        return cross / norm

    def compute_yv(self, x, y, yaw, x_l, y_l):
        dx = x_l - x
        dy = y_l - y

        # Transform into vehicle frame
        x_v =  dx * math.cos(yaw) + dy * math.sin(yaw)
        y_v = -dx * math.sin(yaw) + dy * math.cos(yaw)

        return y_v

    def angle_normalize(self, ang):
        while ang > np.pi:
            ang -= 2.0 * np.pi
        while ang < -np.pi:
            ang += 2.0 * np.pi
        return ang

    def mpc_longitudinal(self, v, v_desired, dt):
        # MPC-like velocity horizon with a longer lookahead and rate-limited outputs.
        N = 30
        a_min, a_max = -3.0, 1.0
        horizon_scale = max(N * dt + 1.0, 2.0)

        a_cmd = np.clip((v_desired - v) / horizon_scale, a_min, a_max)
        a_cmd = 0.75 * self.vars.a_prev + 0.25 * a_cmd
        self.vars.a_prev = a_cmd

        v_error = v_desired - v
        if abs(v_error) < 0.12:
            throttle = 0.0
            brake = 0.0
        elif a_cmd > 0.0:
            throttle = np.clip(a_cmd / 0.9, 0.0, 1.0)
            if v_error < 0.5:
                throttle = min(throttle, 0.65)
            brake = 0.0
        else:
            throttle = 0.0
            brake = np.clip(-a_cmd / 3.5, 0.0, 1.0)

        # Softly rate-limit transitions for smoother actuation.
        throttle = self.vars.throttle_prev + np.clip(throttle - self.vars.throttle_prev, -0.05, 0.05)
        brake = self.vars.brake_prev + np.clip(brake - self.vars.brake_prev, -0.05, 0.05)
        throttle = np.clip(throttle, 0.0, 1.0)
        brake = np.clip(brake, 0.0, 1.0)
        self.vars.throttle_prev = throttle
        self.vars.brake_prev = brake

        return throttle, brake

    def mpc_lateral(self, x, y, yaw, v, waypoints):
        # Receding-horizon steering using a constant-control candidate search.
        N = 10
        dt = 0.1
        L = self._wheelbase
        total_wp = len(waypoints)

        closest_idx = self.vars.closest_idx_prev
        if closest_idx < 0 or closest_idx >= total_wp:
            closest_idx = 0

        closest_dist = float('inf')
        for i in range(closest_idx, total_wp):
            dx = waypoints[i][0] - x
            dy = waypoints[i][1] - y
            dist = math.hypot(dx, dy)
            if dist < closest_dist:
                closest_dist = dist
                closest_idx = i

        if closest_idx >= total_wp - 2:
            closest_idx = max(total_wp - 2, 0)
        self.vars.closest_idx_prev = closest_idx

        lookahead_dist = max(10.0, 1.5 * v)
        lookahead_idx = closest_idx
        while lookahead_idx < total_wp - 1:
            dx = waypoints[lookahead_idx][0] - x
            dy = waypoints[lookahead_idx][1] - y
            if math.hypot(dx, dy) >= lookahead_dist:
                break
            lookahead_idx += 1

        if lookahead_idx >= total_wp:
            lookahead_idx = total_wp - 1

        x1, y1 = waypoints[closest_idx][0], waypoints[closest_idx][1]
        x2, y2 = waypoints[lookahead_idx][0], waypoints[lookahead_idx][1]
        if closest_idx == lookahead_idx:
            x2, y2 = waypoints[min(closest_idx + 1, total_wp - 1)][0], waypoints[min(closest_idx + 1, total_wp - 1)][1]

        path_yaw = math.atan2(y2 - y1, x2 - x1)
        if v < 0.1:
            delta = 0.0
            self.vars.delta_prev = delta
            return delta

        q_cte = 4.0
        q_psi = 2.0
        r = 0.8

        best_cost = float('inf')
        best_delta = self.vars.delta_prev
        for delta_candidate in np.linspace(-0.45, 0.45, 31):
            x_pred, y_pred, yaw_pred = x, y, yaw
            cost = 0.0
            for _ in range(N):
                x_pred += v * math.cos(yaw_pred) * dt
                y_pred += v * math.sin(yaw_pred) * dt
                yaw_pred += v / L * delta_candidate * dt
                yaw_pred = self.angle_normalize(yaw_pred)

                cte_pred = self.cross_track_error(x_pred, y_pred, x1, y1, x2, y2)
                e_psi_pred = self.angle_normalize(path_yaw - yaw_pred)
                cost += q_cte * (cte_pred ** 2) + q_psi * (e_psi_pred ** 2) + r * (delta_candidate ** 2)

            if cost < best_cost:
                best_cost = cost
                best_delta = delta_candidate

        delta = 0.82 * self.vars.delta_prev + 0.18 * best_delta
        delta = np.clip(delta, -0.55, 0.55)
        self.vars.delta_prev = delta
        return delta


    def update_controls(self):
        ######################################################
        # RETRIEVE SIMULATOR FEEDBACK
        ######################################################
        x         = self._current_x
        y         = self._current_y
        yaw       = self._current_yaw
        v         = self._current_speed
        t         = self._current_timestamp
        waypoints = self._waypoints

        throttle_output = 0.0
        brake_output    = 0.0
        steer_output    = 0.0

        # Update desired speed from waypoints
        self.update_desired_speed()
        v_desired = self._desired_speed

        # Skip the first frame to store previous values properly
        if self._start_control_loop and len(waypoints) >= 2:

            dt = t - self.vars.t_prev
            if dt <= 0.0:
                dt = 1e-3

            throttle_output, brake_output = self.mpc_longitudinal(v, v_desired, dt)
            steer_output = self.mpc_lateral(x, y, yaw, v, waypoints)

            self.vars.t_prev = t

            self.set_throttle(throttle_output)  # 0 to 1
            self.set_steer(steer_output)        # rad
            self.set_brake(brake_output)        # 0 to 1

        self.vars.v_previous = v

