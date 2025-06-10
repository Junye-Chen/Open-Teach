import numpy as np
from openteach.components import Component
from openteach.utils.images import rotate_image, rescale_image
from openteach.utils.timer import FrequencyTimer
from openteach.utils.network import ZMQCameraPublisher, ZMQCompressedImageTransmitter
from openteach.constants import *
import subprocess as sp
import cv2
import time
import multiprocessing as mp
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import threading
import time


"""
cd ~/ros_ws
source ./devel/setup.bash 
roslaunch astra_camera dabai_dc1.launch
"""


class DabaiCamera(Component):
    def __init__(self, stream_configs, cam_serial_num, cam_id, cam_configs, stream_oculus = False):
        # Disabling scientific notations
        np.set_printoptions(suppress=True)
        self.cam_id = cam_id
        self.cam_configs = cam_configs
        self._cam_serial_num = cam_serial_num
        self._stream_configs = stream_configs
        self._stream_oculus = stream_oculus
        self.image_type = "rgb"  # 添加image_type属性，默认为rgb

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

        # self.depth_publisher = ZMQCameraPublisher(
        #     host = stream_configs['host'],
        #     port = stream_configs['port'] + DEPTH_PORT_OFFSET
        # )

        self.timer = FrequencyTimer(CAM_FPS)

        # Starting the realsense pipeline
        self._start_dabai(self._cam_serial_num)

    def _start_dabai(self, cam_serial_num):
        self.bridge = CvBridge()
        self.latest_cv_image = None  # 用于存储最新的图像
        self.image_lock = threading.Lock() # 用于线程安全的锁
        
        # 根据类型设置话题前缀
        self.topic_prefix = rospy.get_param("~camera_name", "camera")
        self.topic_map = {
            "rgb": f"{self.topic_prefix}/color/image_raw",
            "depth": f"{self.topic_prefix}/depth/image_raw",
            "ir": f"{self.topic_prefix}/ir/image_raw"
        }
        
        # 订阅话题
        rospy.Subscriber(
            self.topic_map["rgb"],
            Image,
            self.image_callback,
            queue_size=10
        )

        # # 订阅深度图像话题
        # rospy.Subscriber(
        #     self.topic_map["depth"],
        #     Image,
        #     self.image_callback,
        #     queue_size=10
        # )

    def image_callback(self, msg):
        try:
            cv_image_temp = None # 初始化 cv_image
            if self.image_type == "rgb":
                cv_image_temp = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            elif self.image_type == "depth":
                try:
                    # 尝试获取16位深度图
                    cv_image_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough") # 或者 "16UC1"
                    if cv_image_depth.dtype == 'uint16':
                        # 示例：归一化到0-255以便显示 (需要根据实际深度范围调整)
                        image_min = np.min(cv_image_depth)
                        image_max = np.max(cv_image_depth)       
                        cv_image_display = ((cv_image_depth.astype(float) - image_min) / (image_max - image_min) * 255.0).astype('uint8')
                    
                    else: # 如果不是预期的深度格式，退回到 mono8
                        rospy.logwarn_once(f"Depth image format is {cv_image_depth.dtype}, falling back to mono8 conversion.")
                        cv_image_display = self.bridge.imgmsg_to_cv2(msg, "mono8")

                    cv_image_temp = cv_image_display # 用于显示的图像
                    rospy.loginfo(f"Image type: {cv_image_temp.dtype}, shape: {cv_image_temp.shape}")
                except CvBridgeError as e:
                    rospy.logerr(f"CV Bridge Error for depth: {e}")
                    return # 出现错误则不继续处理
            # 如果图像成功处理，则更新latest_cv_image
            if cv_image_temp is not None:
                with self.image_lock: # 获取锁以安全地更新共享数据
                    self.latest_cv_image = cv_image_temp.copy() # 使用copy()以避免后续修改影响存储的图像
                rospy.loginfo("Successfully processed and updated image")  # 添加成功日志
            
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge Error in image_callback: {e}")
        except Exception as e: # 捕获其他任何可能的异常
            rospy.logerr(f"Unexpected error in image_callback: {e}")

    def get_latest_image(self):
        """
        获取最新处理的图像帧。
        返回:
            numpy.ndarray: 最新的图像帧，如果还没有图像则返回None。
        """
        with self.image_lock: # 获取锁以安全地读取共享数据
            if self.latest_cv_image is not None:
                return self.latest_cv_image.copy() # 返回副本以防止外部修改
            return None

    def get_rgb_depth_images(self):
        frames = None
        start_time = time.time()
        timeout = 5.5  # 5秒超时
        time.sleep(0.5)

        while frames is None:                
            frames = self.get_latest_image()
            
            if time.time() - start_time > timeout:
                rospy.logerr("Timeout waiting for camera image")
                raise TimeoutError("Failed to get camera image within timeout period")
            time.sleep(0.01)  # 添加短暂休眠以减少CPU使用

        return np.asanyarray(frames), time.time()


    def stream(self):
        # Starting the realsense stream
        self.notify_component_start('dabai')
        # print(f"Started the Realsense pipeline for camera: {self._cam_serial_num}!")
        print("Starting stream on {}:{}...\n".format(self._stream_configs['host'], self._stream_configs['port']))
        
        if self._stream_oculus:
            print('Starting oculus stream on port: {}\n'.format(self._stream_configs['port'] + VIZ_PORT_OFFSET))

        while True:
            try:
                self.timer.start_loop()
                color_image, timestamp = self.get_rgb_depth_images()
                cv2.imwrite('color_image.jpg', color_image) # 保存图像
                if color_image is None:
                    raise ValueError("Failed to read color image from Dabai camera.")  # 这里添加判断color_image是否读取成功
                color_image = rotate_image(color_image, self.cam_configs.rotation_angle)

                # Publishing the rgb images
                self.rgb_publisher.pub_rgb_image(color_image, timestamp)
                # TODO - move the oculus publisher to a separate process - this cycle works at 40 FPS
                if self._stream_oculus:
                    self.rgb_viz_publisher.send_image(rescale_image(color_image, self.cam_configs.width, self.cam_configs.height)) # 640 * 360

                # Publishing the depth images
                # self.depth_publisher.pub_depth_image(depth_image, timestamp)
                # self.depth_publisher.pub_intrinsics(self.intrinsics_matrix) # Publishing inrinsics along with the depth publisher

                self.timer.end_loop()
            except KeyboardInterrupt:
                break

        print('Shutting down pipeline for camera {}.'.format(self.cam_id))
        self.rgb_publisher.stop()
        if self._stream_oculus:
            self.rgb_viz_publisher.stop()
                
        
    def run(self):
        rospy.spin()  # 阻塞等待消息到达


if __name__ == '__main__':
    # import rospy
    # from sensor_msgs.msg import Image
    # from cv_bridge import CvBridge, CvBridgeError

    
    # 读取配置文件
    # stream_configs = rospy.get_param("~stream_configs")
    # cam_serial_num = rospy.get_param("~cam_serial_num")
    # cam_id = rospy.get_param("~cam_id")
    # cam_configs = rospy.get_param("~cam_configs")
    # stream_oculus = rospy.get_param("~stream_oculus", False)

    stream_configs = {'host':'192.168.1.119','port':int(10005)}
    cam_serial_num = "145645318"
    cam_id = 0
    stream_oculus = False
    if not rospy.core.is_initialized():
        rospy.init_node('dabai_camera_node')

    class cam_configs:
        def __init__(self):
            self.rotation_angle = 0
            self.width = 640
            self.height = 360

    # 实例化DabaiCamera类
    dabai_camera = DabaiCamera(stream_configs, cam_serial_num, cam_id, cam_configs, stream_oculus)

    stream_thread = threading.Thread(target=dabai_camera.stream)
    stream_thread.daemon = True  # 设置为守护线程，主程序退出时线程也退出
    stream_thread.start()

    print("ROS node spinning...")
    rospy.spin()

    # 启动DabaiCamera的stream方法
    # dabai_camera.stream()