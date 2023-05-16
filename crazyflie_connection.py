# -*- coding: utf-8 -*-
#
#     ||          ____  _ __
#  +------+      / __ )(_) /_______________ _____  ___
#  | 0xBC |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
#  +------+    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
#   ||  ||    /_____/_/\__/\___/_/   \__,_/ /___/\___/
#
#  Copyright (C) 2014 Bitcraze AB
#
#  Crazyflie Nano Quadcopter Client
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
"""
Simple example that connects to the first Crazyflie found, logs the Stabilizer
and prints it to the console. After 10s the application disconnects and exits.
"""
import logging
import time
from threading import Timer
import copy
import numpy as np

import cflib.crtp  # noqa
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.utils import uri_helper

from controllers.main.parameters import Parameters

uri = uri_helper.uri_from_env(default='radio://0/100/2M/E7E7E7E710')

# Only output errors from the logging framework
logging.basicConfig(level=logging.ERROR)


class CrazyflieConnection():
    """
    Simple logging example class that logs the Stabilizer from a supplied
    link uri and disconnects after 5s.
    """

    def __init__(self, link_uri, params: Parameters):
        """ Initialize and run the example with the specified link_uri """
        self.params = params

        self._cf = Crazyflie(rw_cache='./cache')

        # Connect some callbacks from the Crazyflie API
        self._cf.connected.add_callback(self._connected)
        self._cf.disconnected.add_callback(self._disconnected)
        self._cf.connection_failed.add_callback(self._connection_failed)
        self._cf.connection_lost.add_callback(self._connection_lost)

        print('Connecting to %s' % link_uri)

        # Try to connect to the Crazyflie
        self._cf.open_link(link_uri)

        # Variable used to keep main loop occupied until disconnect
        self.is_connected = True

        # sensor data
        self._sensor_data = { 
            "x_global": 0.0,
            "y_global": 0.0,
            "yaw": 0.0,
            "range_front": 0.0,
            "range_left": 0.0,
            "range_back": 0.0,
            "range_right": 0.0,
            "range_down": 0.0,
        }

    def setParamValue(self, param, value):
        self._cf.param.set_value(param, value)

    def setCommand(self, command):
        self._cf.commander.send_hover_setpoint(command[0], command[1], command[2], command[3])

    def setStopCommand(self):
        self._cf.commander.send_stop_setpoint()

    def getSensorData(self):
        sensor_data = copy.deepcopy(self._sensor_data)
        return sensor_data

    def _connected(self, link_uri):
        """ This callback is called form the Crazyflie API when a Crazyflie
        has been connected and the TOCs have been downloaded."""
        print('Connected to %s' % link_uri)

        # The definition of the logconfig can be made before connecting
        self._lg_stab = LogConfig(name='Stabilizer', period_in_ms=20) # 50
        self._lg_stab.add_variable('stateEstimate.x', 'float')
        self._lg_stab.add_variable('stateEstimate.y', 'float')
        self._lg_stab.add_variable('stateEstimate.z', 'float')
        self._lg_stab.add_variable('stabilizer.yaw', 'float')
        self._lg_stab.add_variable('range.front')
        self._lg_stab.add_variable('range.back')
        self._lg_stab.add_variable('range.left')
        self._lg_stab.add_variable('range.right')
        self._lg_stab.add_variable('range.zrange')
        # The fetch-as argument can be set to FP16 to save space in the log packet
        # self._lg_stab.add_variable('pm.vbat', 'FP16')

        # Adding the configuration cannot be done until a Crazyflie is
        # connected, since we need to check that the variables we
        # would like to log are in the TOC.
        try:
            self._cf.log.add_config(self._lg_stab)
            # This callback will receive the data
            self._lg_stab.data_received_cb.add_callback(self._stab_log_data)
            # This callback will be called on errors
            self._lg_stab.error_cb.add_callback(self._stab_log_error)
            # Start the logging
            self._lg_stab.start()
        except KeyError as e:
            print('Could not start log configuration,'
                  '{} not found in TOC'.format(str(e)))
        except AttributeError:
            print('Could not add Stabilizer log config, bad configuration.')

        # Start a timer to disconnect in 10s
        t = Timer(500, self._cf.close_link)
        t.start()

    def _stab_log_error(self, logconf, msg):
        """Callback from the log API when an error occurs"""
        print('Error when logging %s: %s' % (logconf.name, msg))

    def _stab_log_data(self, timestamp, data, logconf):
        """Callback from a the log API when data arrives"""
        # print(f'[{timestamp}][{logconf.name}]: ', end='')
        # for name, value in data.items():
        #     print(f'{name}: {value:3.3f} ', end='')
        # print()

        x_global = data["stateEstimate.x"] + self.params.path_init_pos[0]
        y_global = data["stateEstimate.y"] + self.params.path_init_pos[1]
        yaw = data["stabilizer.yaw"] * np.pi / 180.0
        front = data["range.front"] / 1000.0
        back = data["range.back"] / 1000.0
        left = data["range.left"] / 1000.0
        right = data["range.right"] / 1000.0
        down = data["range.zrange"] / 1000.0

        # normalize yaw
        if yaw - self._sensor_data["yaw"] > np.pi:
            yaw -= 2 * np.pi
        elif yaw - self._sensor_data["yaw"] <= -np.pi:
            yaw += 2 * np.pi

        alpha = 1.0
        self._sensor_data["x_global"] = alpha * x_global + (1 - alpha) * self._sensor_data["x_global"]
        self._sensor_data["y_global"] = alpha * y_global + (1 - alpha) * self._sensor_data["y_global"]
        self._sensor_data["yaw"] = alpha * yaw + (1 - alpha) * self._sensor_data["yaw"]
        self._sensor_data["range_down"] =  alpha * down + (1 - alpha) * self._sensor_data["range_down"]

        alpha = 0.5
        self._sensor_data["range_front"] = alpha * front + (1 - alpha) * self._sensor_data["range_front"]
        self._sensor_data["range_back"] = alpha * back + (1 - alpha) * self._sensor_data["range_back"]
        self._sensor_data["range_left"] = alpha * left + (1 - alpha) * self._sensor_data["range_left"]
        self._sensor_data["range_right"] = alpha * right + (1 - alpha) * self._sensor_data["range_right"]
        

        # print(f"yaw: meas={round(np.rad2deg(yaw), 2)}, filtered={round(np.rad2deg(self._sensor_data['yaw']), 2)}")

        # print(f"measurement: \t{round(front, 2)}, {round(back, 2)}, {round(left, 2)}, {round(right, 2)}, {round(down, 2)}")
        # print(f"filtered: \t{round(self._sensor_data['range_front'], 2)}, {round(self._sensor_data['range_back'], 2)}, \
        #       {round(self._sensor_data['range_left'], 2)}, {round(self._sensor_data['range_right'], 2)}, {round(self._sensor_data['range_down'], 2)}")

    def _connection_failed(self, link_uri, msg):
        """Callback when connection initial connection fails (i.e no Crazyflie
        at the specified address)"""
        print('Connection to %s failed: %s' % (link_uri, msg))
        self.is_connected = False

    def _connection_lost(self, link_uri, msg):
        """Callback when disconnected after a connection has been made (i.e
        Crazyflie moves out of range)"""
        print('Connection to %s lost: %s' % (link_uri, msg))

    def _disconnected(self, link_uri):
        """Callback when the Crazyflie is disconnected (called in all cases)"""
        print('Disconnected from %s' % link_uri)
        self.is_connected = False

