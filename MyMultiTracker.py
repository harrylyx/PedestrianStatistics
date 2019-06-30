from cv2 import *
class MyMultiTracker(MultiTracker):
    def __init__(self):
        MultiTracker.__init__(self)

    def delete(self):
        self.getObjects()
