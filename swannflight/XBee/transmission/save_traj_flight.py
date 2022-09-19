import pickle
import obs_utils
import sys
import time
from watchdog.observers import Observer  #creating an instance of the watchdog.observers.Observer from watchdogs class.
from watchdog.events import FileSystemEventHandler


class MyHandler(FileSystemEventHandler):
    def on_modified(self, event):
        obs_utils.convert_traj_to_flight_log(pickle.load( open( "traj.p", "rb" ) ), "traj")


if __name__ == "__main__":
    observer = Observer()
    observer.schedule(MyHandler(), "traj.p", recursive=False)  #Scheduling monitoring of a path with the observer instance and event handler. There is 'recursive=True' because only with it enabled, watchdog.observers.Observer can monitor sub-directories
    observer.start()  #for starting the observer thread
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()