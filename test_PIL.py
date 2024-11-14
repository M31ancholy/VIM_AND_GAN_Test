from PIL import Image
import numpy as np

# 创建一个简单的测试图像
image_data = np.random.rand(100, 100, 3) * 255  # 生成随机颜色图像
image_data = image_data.astype(np.uint8)  # 确保数据类型为 uint8

# 转换为 PIL 图像
image = Image.fromarray(image_data)

# 显示图像
image.show()
