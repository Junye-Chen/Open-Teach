import numpy as np
from openteach.components import Component
from openteach.utils.images import rotate_image, rescale_image
from openteach.utils.timer import FrequencyTimer
from openteach.utils.network import ZMQCameraPublisher, ZMQCompressedImageTransmitter
from openteach.constants import *
import cv2
from typing import Union, Any, Optional
import time

MIN_DEPTH = 20  # 20mm
MAX_DEPTH = 10000  # 10000mm


from pyorbbecsdk import (
    Config, Pipeline,
    OBSensorType, OBStreamType, FormatConvertFilter,
    VideoFrame, AlignFilter, OBConvertFormat,
    OBFormat, OBError, FrameSet, VideoStreamProfile
)

def i420_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    y = frame[0:height, :]
    u = frame[height:height + height // 4].reshape(height // 2, width // 2)
    v = frame[height + height // 4:].reshape(height // 2, width // 2)
    yuv_image = cv2.merge([y, u, v])
    bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_I420)
    return bgr_image

def nv21_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    y = frame[0:height, :]
    uv = frame[height:height + height // 2].reshape(height // 2, width)
    yuv_image = cv2.merge([y, uv])
    bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV21)
    return bgr_image


def nv12_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    y = frame[0:height, :]
    uv = frame[height:height + height // 2].reshape(height // 2, width)
    yuv_image = cv2.merge([y, uv])
    bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV12)
    return bgr_image

def frame_to_bgr_image(frame: VideoFrame) -> Union[Optional[np.array], Any]:
    width = frame.get_width()
    height = frame.get_height()
    color_format = frame.get_format()
    data = np.asanyarray(frame.get_data())
    image = np.zeros((height, width, 3), dtype=np.uint8)
    if color_format == OBFormat.RGB:
        image = np.resize(data, (height, width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif color_format == OBFormat.BGR:
        image = np.resize(data, (height, width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif color_format == OBFormat.YUYV:
        image = np.resize(data, (height, width, 2))
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
    elif color_format == OBFormat.MJPG:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    elif color_format == OBFormat.I420:
        image = i420_to_bgr(data, width, height)
        return image
    elif color_format == OBFormat.NV12:
        image = nv12_to_bgr(data, width, height)
        return image
    elif color_format == OBFormat.NV21:
        image = nv21_to_bgr(data, width, height)
        return image
    elif color_format == OBFormat.UYVY:
        image = np.resize(data, (height, width, 2))
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
    else:
        print("Unsupported color format: {}".format(color_format))
        return None
    return image

class TemporalFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.previous_frame = None

    def process(self, frame):
        if self.previous_frame is None:
            result = frame
        else:
            result = cv2.addWeighted(frame, self.alpha, self.previous_frame, 1 - self.alpha, 0)
        self.previous_frame = result
        return result


class GeminiCamera(Component):
    def __init__(self, stream_configs, cam_serial_num, cam_id, cam_configs, stream_oculus = False):
        # Disabling scientific notations
        np.set_printoptions(suppress=True)
        self.cam_id = cam_id
        self.cam_configs = cam_configs
        self._cam_serial_num = cam_serial_num
        self._stream_configs = stream_configs
        self._stream_oculus = stream_oculus
        self.temporal_filter = TemporalFilter(alpha=0.5)

        # Different publishers to avoid overload
        self.rgb_publisher = ZMQCameraPublisher(
            host = stream_configs['host'],
            port = stream_configs['port']
        )
        
        if self._stream_oculus:
            self.rgb_viz_publisher = ZMQCompressedImageTransmitter(
                host = stream_configs['host'],
                port = stream_configs['port'] + VIZ_PORT_OFFSET
            )

        self.depth_publisher = ZMQCameraPublisher(
            host = stream_configs['host'],
            port = stream_configs['port'] + DEPTH_PORT_OFFSET
        )

        self.timer = FrequencyTimer(CAM_FPS)

        # Starting the realsense pipeline
        self._start_gemini(self._cam_serial_num)

    def _start_gemini(self, cam_serial_num):
        config = Config()
        self.pipeline = Pipeline()
        try:
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            profile_list1 = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            try:
                color_profile = profile_list.get_video_stream_profile(640, 0, OBFormat.RGB, 30)
                depth_profile = profile_list1.get_default_video_stream_profile()
            except OBError as e:
                print(e)
                color_profile = profile_list.get_default_video_stream_profile()
                print("color profile: ", color_profile)

            config.enable_stream(color_profile)
            config.enable_stream(depth_profile)  # ! 有可能出问题

        except Exception as e:
            print(e)
            return
        self.pipeline.start(config)

        self.align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

        # self.pipeline.stop()

    def get_rgb_depth_images(self):     
        frames = None   
        color_frame = None
        depth_frame = None
        while color_frame is None or depth_frame is None:
            try:
                frames: FrameSet = self.pipeline.wait_for_frames(100)
                if frames is None:
                    continue
                # color_frame = frames.get_color_frame()
                # depth_frame = frames.get_depth_frame()

                frames = self.align_filter.process(frames)
                if not frames:
                    continue
                frames = frames.as_frame_set()
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if color_frame is None or depth_frame is None:
                    continue
                depth_format = depth_frame.get_format()
                if depth_format != OBFormat.Y16:
                    print("depth format is not Y16")
                    continue

                # covert to RGB format
                color_image = frame_to_bgr_image(color_frame)
                color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                if color_image is None:
                    print("failed to convert frame to image")
                    continue

                width = depth_frame.get_width()
                height = depth_frame.get_height()
                scale = depth_frame.get_depth_scale()
                depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                depth_data = depth_data.reshape((height, width))

                depth_data = depth_data.astype(np.float32) * scale
                depth_data = np.where((depth_data > MIN_DEPTH) & (depth_data < MAX_DEPTH), depth_data, 0)
                depth_data = depth_data.astype(np.uint16)
                # # Apply temporal filtering
                depth_data = self.temporal_filter.process(depth_data)
                depth_image = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            except KeyboardInterrupt:
                break        

        # return color_image, frames.get_timestamp()
        return color_image, depth_image, time.time()


    def stream(self):
        # Starting the realsense stream
        self.notify_component_start('gemini')
        print(f"Started the gemini pipeline for camera: {self._cam_serial_num}!")
        print("Starting stream on {}:{}...\n".format(self._stream_configs['host'], self._stream_configs['port']))
        
        if self._stream_oculus:
            print('Starting oculus stream on port: {}\n'.format(self._stream_configs['port'] + VIZ_PORT_OFFSET))

        while True:
            #try:
                self.timer.start_loop()
                color_image, depth_image, timestamp = self.get_rgb_depth_images()
                # color_image, timestamp = self.get_rgb_depth_images()
                # print('timestamp', timestamp)

                # 打开窗口显示color_image
                # cv2.imshow("color", color_image)
                # cv2.waitKey(1)
                
                # depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
                # cv2.imshow("Depth Viewer", depth_image)
                # cv2.waitKey(1)

                if color_image is None:
                    raise ValueError("Failed to read color image from Dabai camera.")  # 这里添加判断color_image是否读取成功
                color_image = rotate_image(color_image, self.cam_configs.rotation_angle)
                depth_image = rotate_image(depth_image, self.cam_configs.rotation_angle)

                # Publishing the rgb images
                self.rgb_publisher.pub_rgb_image(color_image, timestamp)
                # TODO - move the oculus publisher to a separate process - this cycle works at 40 FPS
                if self._stream_oculus:
                    self.rgb_viz_publisher.send_image(rescale_image(color_image, self.cam_configs.width, self.cam_configs.height)) # 640 * 360

                # Publishing the depth images
                self.depth_publisher.pub_depth_image(depth_image, timestamp)
                # self.depth_publisher.pub_intrinsics(self.intrinsics_matrix) # Publishing inrinsics along with the depth publisher

                self.timer.end_loop()
            # except KeyboardInterrupt:
            #     break
        
        print('Shutting down realsense pipeline for camera {}.'.format(self.cam_id))
        self.rgb_publisher.stop()
        if self._stream_oculus:
            self.rgb_viz_publisher.stop()
        self.depth_publisher.stop()
        self.pipeline.stop()
        
if __name__ == '__main__':

    stream_configs = {'host':'192.168.1.119','port':int(10005)}
    cam_serial_num = "145645318"
    cam_id = 0
    stream_oculus = False    

    class Cam_configs:
        def __init__(self):
            self.rotation_angle = 0
            self.width = 640
            self.height = 360
    cam_configs = Cam_configs()

    cam = GeminiCamera(stream_configs, cam_serial_num, cam_id, cam_configs, stream_oculus = True)

    cam.stream()

