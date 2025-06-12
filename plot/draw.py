import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import collections

"""
用于画图分析坐标轴方向
"""

# 根据输入的三维坐标，绘制其三个坐标随时间变化的曲线，将曲线绘制在三维坐标系中
def draw_3d_curve(array, title='3D Curve', xlabel='X', ylabel='Y', zlabel='Z'):
    x, y, z = array[0], array[1], array[2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, label='3D Curve')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    # 显示图例
    ax.legend()
    # 设置网格
    ax.grid(True)
    plt.show()


def plot_coordinates_over_time(x_coords, y_coords, z_coords):
    """
    根据输入的时间和三维坐标数据，绘制三个坐标随时间变化的曲线。

    参数:
    t (list or np.array): 时间数据点。
    x_coords (list or np.array): 对应每个时间点的X坐标。
    y_coords (list or np.array): 对应每个时间点的Y坐标。
    z_coords (list or np.array): 对应每个时间点的Z坐标。
    """
    # --- 1. 创建画布和坐标系 ---
    # figsize可以设置图形的大小
    plt.figure(figsize=(12, 7))

    # print("x_coords", x_coords.shape, len(x_coords))

    t = np.arange(len(x_coords))
    # print("t", t, t.shape)

    # --- 2. 绘制三条曲线 ---
    # 绘制 X 坐标随时间变化的曲线
    plt.plot(t, x_coords, label='X Coordinate', color='r', linestyle='-')
    
    # 绘制 Y 坐标随时间变化的曲线
    plt.plot(t, y_coords, label='Y Coordinate', color='g', linestyle='--')
    
    # 绘制 Z 坐标随时间变化的曲线
    plt.plot(t, z_coords, label='Z Coordinate', color='b', linestyle=':')

    # --- 3. 设置图表属性 ---
    # 设置图表标题
    plt.title('Coordinates vs. Time', fontsize=16)
    
    # 设置X轴和Y轴的标签
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Position', fontsize=12)
    
    # 显示图例，loc='best'表示自动选择最佳位置
    plt.legend(loc='best')
    
    # 显示网格
    plt.grid(True)
    
    # --- 4. 显示图形 ---
    plt.show()



def plot_realtime_coordinates(array_data, t):
    # 设置固定长度的数据缓冲区，例如只显示最近的1000个点
    # 如果想显示所有点，可以去掉 maxlen 参数
    history_len = 1000
    t_data = collections.deque(maxlen=history_len)
    x_data = collections.deque(maxlen=history_len)
    y_data = collections.deque(maxlen=history_len)
    z_data = collections.deque(maxlen=history_len)

    # 创建图表和坐标系
    fig, ax = plt.subplots(figsize=(12, 7))

    # 初始化三条空的曲线，我们稍后会更新它们的数据
    # 'r-' 表示红色实线, 'g--' 表示绿色虚线, 'b:' 表示蓝色点线
    line_x, = ax.plot([], [], 'r-', label='X Coordinate')
    line_y, = ax.plot([], [], 'g--', label='Y Coordinate')
    line_z, = ax.plot([], [], 'b:', label='Z Coordinate')

    # 设置图表的基本属性
    ax.set_xlabel('Time')
    ax.set_ylabel('Position')
    ax.set_title('Real-time Coordinates vs. Time')
    ax.legend()
    ax.grid(True)

    # 解析新数据
    # t, x, y, z = array_data
    def update(array_data):
        x, y, z = array_data[0], array_data[1], array_data[2]
        
        # 将新数据添加到我们的数据缓冲区
        t_data.append(t)
        x_data.append(x)
        y_data.append(y)
        z_data.append(z)
        
        # 更新曲线的数据
        line_x.set_data(t_data, x_data)
        line_y.set_data(t_data, y_data)
        line_z.set_data(t_data, z_data)
        
        # 重新计算并调整坐标轴的范围
        ax.relim()
        ax.autoscale_view()
        
        # 返回已更新的艺术家对象
        return line_x, line_y, line_z

    # --- 4. 创建并启动动画 ---
    # FuncAnimation 将会不断调用 update 函数
    # - fig: 更新的图表对象
    # - update: 更新函数
    # - frames: 数据源，这里是我们的生成器
    # - interval: 更新间隔（毫秒），例如 50ms 更新一次
    # - blit: 优化绘图性能，只重绘变化的部分
    ani = animation.FuncAnimation(fig, update, frames=array_data, interval=10, blit=True, cache_frame_data=False)

    # 显示图表
    plt.show()



# def plot_realtime_coordinates(
#     data_source: Generator[Tuple[float, float, float, float], None, None],
#     history_len: int = 100,
#     interval: int = 50
# ):
#     """
#     接收一个实时数据源，动态绘制三维坐标随时间变化的曲线。

#     参数:
#     data_source (Generator): 一个生成器函数。该生成器每次被调用时，
#                              必须 yield 一个包含 (时间, x, y, z) 的元组。
#     history_len (int): 图表上保留的数据点数量（滑动窗口大小）。
#     interval (int): 图表刷新间隔，单位为毫秒。
#     """
    
#     # --- 1. 初始化图表和数据存储 ---
#     t_data = collections.deque(maxlen=history_len)
#     x_data = collections.deque(maxlen=history_len)
#     y_data = collections.deque(maxlen=history_len)
#     z_data = collections.deque(maxlen=history_len)

#     fig, ax = plt.subplots(figsize=(12, 7))
#     line_x, = ax.plot([], [], 'r-', label='X Coordinate')
#     line_y, = ax.plot([], [], 'g--', label='Y Coordinate')
#     line_z, = ax.plot([], [], 'b:', label='Z Coordinate')

#     ax.set_xlabel('Time')
#     ax.set_ylabel('Position')
#     ax.set_title('Real-time Coordinates vs. Time')
#     ax.legend()
#     ax.grid(True)
    
#     # --- 2. 定义内部更新函数 ---
#     # 这个函数在动画的每一帧被调用
#     def update(frame_data: Tuple[float, float, float, float]):
#         """内部函数，用于更新曲线数据。"""
#         t, x, y, z = frame_data
        
#         t_data.append(t)
#         x_data.append(x)
#         y_data.append(y)
#         z_data.append(z)
        
#         line_x.set_data(t_data, x_data)
#         line_y.set_data(t_data, y_data)
#         line_z.set_data(t_data, z_data)
        
#         # 自动调整坐标轴范围
#         ax.relim()
#         ax.autoscale_view()
        
#         return line_x, line_y, line_z

#     # --- 3. 创建并启动动画 ---
#     # FuncAnimation 会从 data_source 中获取数据并传递给 update 函数
#     ani = animation.FuncAnimation(
#         fig=fig, 
#         func=update, 
#         frames=data_source, 
#         interval=interval, 
#         blit=True,
#         cache_frame_data=False
#     )
    
#     # 显示图表
#     plt.show()


if __name__ == '__main__':
    # # 三维坐标
    # t = np.linspace(0, 20, 500) # 从0到20生成500个点

    # # 根据时间生成 x, y, z 坐标
    # x_coords = np.cos(t).reshape(-1, 1)
    # y_coords = np.sin(t).reshape(-1, 1)
    # z_coords = t / 4
    # z_coords = z_coords.reshape(-1, 1)

    # input_array = np.concatenate([x_coords, y_coords, z_coords], axis=1)
    # # print(input_array.shape)

    # # print(x_coords.shape, y_coords.shape, z_coords.shape)   
    # draw_3d_curve(input_array, title='3D Curve', xlabel='X', ylabel='Y', zlabel='Z')
    # plot_coordinates_over_time(t, x_coords, y_coords, z_coords)

    x_data = []
    y_data = []
    z_data = []
    # 按行读取txt文件中的数据
    with open('/home/eigindustry/workspace/Open-Teach/move.txt', 'r') as f:
        for line in f.readlines():
            # 去掉每行末尾的换行符
            line = line.strip()
            # 按空格分割每行数据
            line = line[1:-1]
            # print(line)
            data = line.split()
            # print(data[0])
            # break
            # 转换数据类型
            x = float(data[0])
            y = float(data[1])  
            z = float(data[2])
            # 保存数据
            x_data.append(x)
            y_data.append(y)
            z_data.append(z)

    print(len(x_data), len(y_data), len(z_data))
    # # 绘制三维曲线
    # plot_coordinates_over_time(np.array(x_data), np.array(y_data), np.array(z_data))



    n = np.linalg.norm(np.array([3, 3, 3]))
    print(n)



