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
and prints it to the consodrone. After 10s the application disconnects and exits.
"""
import logging
import time
from threading import Timer

import cflib.crtp  # noqa
from cflib.crazyflie import Crazyflie
from cflib.drone.log import LogConfig
from cflib.utils import uri_helper

from drone.crazyflie_connection import CrazyflieConnection

uri = uri_helper.uri_from_env(default='radio://0/100/2M/E7E7E7E710')


# Only output errors from the logging framework
logging.basicConfig(level=logging.ERROR)


def main():
    # Initialize the low-level drivers
    cflib.crtp.init_drivers()

    drone = CrazyflieConnection(link_uri=uri)

    drone.setParamValue(param='kalman.resetEstimation', value='1')
    time.sleep(0.1)
    drone.setParamValue(param='kalman.resetEstimation', value='0')
    time.sleep(2)

    # The Crazyflie lib doesn't contain anything to keep the application alive,
    # so this is where your application should do something. In our case we
    # are just waiting until we are disconnected.
    while drone.is_connected:
        time.sleep(0.01)
        for y in range(10):
            command = (0, 0, 0, y / 25)
            drone.setCommand(command=command)
            time.sleep(0.1)

        for _ in range(20):
            command = (0, 0, 0, 0.4)
            drone.setCommand(command=command)
            time.sleep(0.1)

        for _ in range(50):
            command = (0.5, 0, 36 * 2, 0.4)
            drone.setCommand(command=command)
            time.sleep(0.1)

        for _ in range(50):
            command = (0.5, 0, -36 * 2, 0.4)
            drone.setCommand(command=command)
            time.sleep(0.1)

        for _ in range(20):
            command = (0, 0, 0, 0.4)
            drone.setCommand(command=command)
            time.sleep(0.1)

        for y in range(10):
            command = (0, 0, 0, (10 - y) / 25)
            drone.setCommand(command=command)
            time.sleep(0.1)

        drone.setStopCommand()
        break


if __name__ == '__main__':
    main()