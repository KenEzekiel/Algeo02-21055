import cv2
# source: https://solarianprogrammer.com/2018/04/21/python-opencv-show-video-tkinter-window/


class VideoCapture:
    def __init__(self, source=0) -> None:
        self.source = source

    def start(self):
        self.video = cv2.VideoCapture(self.source)
        if not self.video.isOpened():
            raise RuntimeError("Unable to open video source", self.source)
        self.width = self.video.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def stop(self) -> None:
        if self.video is not None and self.video.isOpened():
            self.video.release()
        self.video = None

    def get_image(self):
        if self.video is not None and self.video.isOpened():
            success, img = self.video.read()
            if success:
                return (success, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            return (success, None)
        return (False, None)
