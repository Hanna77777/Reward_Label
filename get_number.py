import pyrealsense2 as rs

# 初始化 RealSense 相机
ctx = rs.context()
devices = ctx.query_devices()

# 获取所有相机的序列号
for device in devices:
    serial_number = device.get_info(rs.camera_info.serial_number)
    print("Camera Serial Number:", serial_number)

# 关闭相机设备
ctx.stop()