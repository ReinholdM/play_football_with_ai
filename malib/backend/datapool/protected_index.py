from signal import signal, SIGTERM
from threading import Lock
from readerwriterlock import rwlock


class StrictIndex:
    def __init__(self, capacity: int = int(1e5), segmentation_size: int = 25) -> None:
        self.lock = rwlock.RWLockFair()

    def write(self, batch):
        pass

    def read(self, bat_size):
        pass
