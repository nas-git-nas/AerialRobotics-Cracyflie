import multiprocessing
from multiprocessing import shared_memory
import numpy as np
import matplotlib.pyplot as plt
import time
# import sysv_ipc


class ImagePlotter(multiprocessing.Process):
    def __init__(self, event, a):
        super().__init__()
        self.event = event
        self.a = a

    def run(self):
        existing_shm = shared_memory.SharedMemory(name='map')
        c = np.ndarray(self.a.shape, dtype=self.a.dtype, buffer=existing_shm.buf)


        while True:
            # Wait for the event to be set
            self.event.wait()
            # c[:,:] = np.random.rand(2,2)
            print(f"child: c: {c}")

            # Reset the event
            self.event.clear()

if __name__ == '__main__':
    # create map
    a = np.zeros((2, 2))

    try:
        shm = shared_memory.SharedMemory(create=True, size=a.nbytes, name='map')
        print("Create shared memory")
    except:
        shm = multiprocessing.shared_memory.SharedMemory(name='map')
        shm.close()
        shm.unlink()
        print("Shared memory already exists -> open it")
        shm = shared_memory.SharedMemory(create=True, size=a.nbytes, name='map')

        # # verify that memory has the same size than the array
        # if shm.size != a.nbytes:
        #     print("Error: shared memory has wrong size")
        #     print(f"    shm.size: {shm.size}, a.nbytes: {a.nbytes}")
        #     exit(1)


    # create shared memory
    # shm = shared_memory.SharedMemory(create=True, size=a.nbytes, name='map')

    b = np.ndarray(a.shape, dtype=a.dtype, buffer=shm.buf)
    b[:] = a[:]


    # Create the event and start the trigger thread
    event = multiprocessing.Event()

    # Create the plotter process and start it
    plotter = ImagePlotter(event, a)
    plotter.start()

    

    # # Get a list of all shared memory segments
    # segments = sysv_ipc.shm_list()

    # # Print information about each segment
    # for seg in segments:
    #     print("ID:", seg[0])
    #     print("Size:", seg[1])
    #     print("Key:", seg[2])
    #     print("Attached Processes:", seg[3])

    while True:
        # Wait for 1 second before setting the event
        time.sleep(1)
        b[:,:] = np.random.rand(2, 2)
        print(f"parent: b: {b}")
        event.set()