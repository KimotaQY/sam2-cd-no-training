import cv2
import numpy as np
import os
from glob import glob

def get_mask_bboxes(mask_path, class_value=255, min_area=10):
    """
    获取二值化掩码图中所有目标的最小外接矩形坐标
    
    参数:
        mask_path (str): 标签图片路径
        class_value (int): 目标类别的像素值（默认255，表示建筑物）
        min_area (int): 忽略面积小于此值的掩码（过滤噪声）
    
    返回:
        list: 每个元素是一个 (x, y, w, h) 的矩形框
    """
    # 读取标签图（单通道）
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # 二值化：确保目标像素=255，背景=0
    _, binary = cv2.threshold(mask, class_value - 1, 255, cv2.THRESH_BINARY)
    
    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bboxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:  # 过滤小面积噪声
            x, y, w, h = cv2.boundingRect(cnt)
            bboxes.append((x, y, w, h))
    
    return bboxes

def visualize_bboxes(image_path, bboxes, output_path=None):
    """可视化矩形框（可选）"""
    image = cv2.imread(image_path)
    for (x, y, w, h) in bboxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    if output_path:
        cv2.imwrite(output_path, image)
    cv2.imshow("BBoxes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 示例用法
if __name__ == "__main__":
    mask_path = "D:/QY/Datasets/WHU-CD/before_label/tile_0_0.png"  # 替换为你的标签图片路径
    
    # 获取所有建筑物的外接矩形
    bboxes = get_mask_bboxes(mask_path)
    print(f"检测到 {len(bboxes)} 个建筑物，坐标如下：")
    for i, (x, y, w, h) in enumerate(bboxes):
        print(f"建筑物 {i + 1}: (x={x}, y={y}, w={w}, h={h})")
    
    # 可视化（需提供原始图像路径）
    visualize_bboxes("D:/QY/Datasets/WHU-CD/train/A/tile_0_0.png", bboxes)