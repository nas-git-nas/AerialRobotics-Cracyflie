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
from collections.abc import Callable, Iterable, Mapping
import logging
import time
from threading import Timer
from typing import Any
import numpy as np
import threading
import keyboard

import cflib.crtp  # noqa
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.utils import uri_helper

from crazyflie_connection import CrazyflieConnection
from controllers.main.parameters import Parameters
from controllers.main.my_control import MyController

uri = uri_helper.uri_from_env(default='radio://0/100/2M/E7E7E7E710')


# Only output errors from the logging framework
logging.basicConfig(level=logging.ERROR)

class KeyEventThread(threading.Thread):
    def __init__(
            self, 
            group: None = None, 
            target: Callable[..., object] | None = None, 
            name: str | None = None, 
            args: Iterable[Any] = ..., kwargs: Mapping[str, Any] | None = None, 
            *, 
            daemon: bool | None = None
        ) -> None:
        super().__init__(group, target, name, args, kwargs, daemon=daemon)

        # keyboard commands
        self.key_kill = False
        self.key_land = False

    def run(self):
        while True:
            if keyboard.read_key() == "k":
                self.key_kill = True
                break
            if keyboard.read_key() == "l":
                self.key_land = True
            time.sleep(0.2) # press the key at leat for 0.2s

def main():
    # initialize parameters
    params = Parameters()

    # thread for keyboard commands
    keythread = KeyEventThread()
    keythread.start()

    # initialize controller
    controller = MyController(params=params)

    # Initialize the low-level drivers
    cflib.crtp.init_drivers()

    drone = CrazyflieConnection(link_uri=uri, params=params)

    drone.setParamValue(param='kalman.resetEstimation', value='1')
    time.sleep(0.1)
    drone.setParamValue(param='kalman.resetEstimation', value='0')
    time.sleep(1)

    # The Crazyflie lib doesn't contain anything to keep the application alive,
    # so this is where your application should do something. In our case we
    # are just waiting until we are disconnected.
    while drone.is_connected:
        # measure time of entire control loop
        start_time = time.time()

        # get sensor data
        sensor_data = drone.getSensorData()

        # determine control command
        command = controller.step_control(sensor_data=sensor_data)

        # check if stop command was sent
        if keythread.key_kill or np.allclose(command, (0.0, 0.0, 0.0, 0.0)):
            drone.setStopCommand()
            print("main: shutting down crazyflie")
            break

        # if keyboard command landing is True
        if keythread.key_land:
            controller.setLanding()

        # send control command to crazyflie
        drone.setCommand(command=command)

        # sleep to ensure control loop time
        step_time = time.time() - start_time
        # print(f"step time: {step_time}")
        if step_time < params.control_loop_period:
            time.sleep(params.control_loop_period - step_time)

    print("main: join keyboard thread")
    keythread.join()



if __name__ == '__main__':
    main()



"""
Questions:
- distance between obstacles: unknown but more than in simulation  
"""