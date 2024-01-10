import cv2 as cv
import numpy as np

class ImageBlender:
    def __init__(self, gamma, blend_ratio=0.25):
        self.gamma = gamma
        self.blend_ratio = blend_ratio  # 融合区域占总宽度的比例

    def blend_images(self, left_image, right_image):
        height, left_width, _ = left_image.shape
        _, right_width, _ = right_image.shape

        # 计算左图和右图融合区域的宽度
        left_blend_width = int(self.blend_ratio * left_width)
        right_blend_width = int(self.blend_ratio * right_width)

        # 确保两张图像的高度相同
        if left_image.shape[0] != right_image.shape[0]:
            raise ValueError("左右图像的高度不同，无法进行融合。")

        # 计算alpha混合值
        alpha_left = np.linspace(1, 0, left_blend_width).reshape(1, left_blend_width, 1)
        alpha_left = np.repeat(alpha_left, height, axis=0)
        alpha_right = np.linspace(0, 1, right_blend_width).reshape(1, right_blend_width, 1)
        alpha_right = np.repeat(alpha_right, height, axis=0)

        # 应用alpha混合到左右图像的融合区域
        blended_region_left = left_image[:, -left_blend_width:] * alpha_left
        blended_region_right = right_image[:, :right_blend_width] * alpha_right
        blended_region = blended_region_left + blended_region_right

        # 创建融合后的图像
        blended_image = np.hstack((left_image[:, :-left_blend_width], blended_region, right_image[:, right_blend_width:]))

        return blended_image


    def gamma_correction(self, image):
        # 应用gamma校正
        lut = np.array([((i / 255.0) ** (1 / self.gamma)) * 255 for i in np.arange(256)]).astype(np.uint8)
        return cv.LUT(image, lut)

    def process_images(self, left_image_path, right_image_path):
        # 读取图像
        left_image = cv.imread(left_image_path, cv.IMREAD_COLOR)
        right_image = cv.imread(right_image_path, cv.IMREAD_COLOR)

        if left_image is None or right_image is None:
            raise FileNotFoundError("One of the input images is not found.")

        # 应用gamma校正
        gamma_corrected_left = self.gamma_correction(left_image)
        gamma_corrected_right = self.gamma_correction(right_image)

        # 混合图像
        adjusted_left, adjusted_right = self.blend_images(gamma_corrected_left, gamma_corrected_right)

        # 创建一个全新的画布，将左右图像拼接起来形成完整的融合图像
        blended_image = np.hstack((adjusted_left, adjusted_right))

        return adjusted_left, adjusted_right, blended_image

# 配置参数
gamma = 1.2  # 伽马校正值

# 图像路径
left_image_path = '/Users/josieq/Desktop/pbl 2nd/ImageLeft0.png'  # 替换为左图像的路径
right_image_path = '/Users/josieq/Desktop/pbl 2nd/ImageRight0.png'  # 替换为右图像的路径

# 创建图像混合器实例
blender = ImageBlender(gamma)

# 处理图像并获取调整后的左右图像以及融合后的图像
adjusted_left, adjusted_right, blended_image = blender.process_images(left_image_path, right_image_path)

# 保存和显示结果
cv.imwrite('adjusted_left.jpg', adjusted_left)
cv.imwrite('adjusted_right.jpg', adjusted_right)
cv.imwrite('blended_image.jpg', blended_image)

cv.imshow('Adjusted Left Image', adjusted_left)
cv.imshow('Adjusted Right Image', adjusted_right)
cv.imshow('Blended Image', blended_image)
cv.waitKey(0)
cv.destroyAllWindows()