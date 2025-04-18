import cv2
import numpy as np
import os
from glob import glob

def extract_single_masks(mask_path, output_dir=None, class_value=255, min_area=10):
    """
    提取标签图中的每个建筑物掩码，并保存为单独的二进制掩码
    
    参数:
        mask_path (str): 标签图片路径
        output_dir (str): 输出文件夹路径
        class_value (int): 建筑物类别的像素值（默认255）
        min_area (int): 忽略面积小于此值的掩码（过滤噪声）
    
    返回:
        list: 每个元素是一个 (mask, bbox, points) 的字典，其中 bbox=(x,y,w,h)
    """
    if output_dir is not None:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
    
    # 读取标签图（单通道）
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"无法读取文件: {mask_path}")
    
    # 二值化：确保建筑物=255，背景=0
    _, binary = cv2.threshold(mask, class_value - 1, 255, cv2.THRESH_BINARY)
    
    # 查找轮廓（只检测最外层轮廓）
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    single_masks = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue  # 跳过小面积噪声
        
        # 生成单个建筑物的掩码
        single_mask = np.zeros_like(binary)
        cv2.drawContours(single_mask, [cnt], -1, 255, thickness=cv2.FILLED)
        
        # 获取外接矩形（可选）
        x, y, w, h = cv2.boundingRect(cnt)

        # 获取质心（可选）
        centroid = get_centroid(single_mask)
        
        if output_dir is not None:
            # 保存掩码
            output_path = os.path.join(output_dir, f"building_{i}.png")
            cv2.imwrite(output_path, single_mask)
            print(f"保存: {output_path} [bbox: ({x}, {y}, {w}, {h})]")
        
        single_masks.append({
            "mask": single_mask, 
            "bbox": (x, y, w, h), 
            "points": [centroid]
            })
    
    return single_masks


def get_centroid(mask):
    """计算二值掩码的质心 (x, y)"""
    y, x = np.where(mask == 255)
    if len(x) == 0:
        return None  # 空掩码
    return [int(np.mean(x)), int(np.mean(y))]


def visualize_mask(mask, bbox=None):
    """可视化掩码和矩形框"""
    vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("Mask", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def _visualize_mask(mask, bbox=None):
    """可视化掩码和边界框（修正版）"""
    # 确保掩码是二值化的
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # 转换为三通道彩色图像
    vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # 绘制边界框（如果提供）
    if bbox is not None:
        x, y, w, h = bbox
        # 确保坐标在图像范围内
        h_img, w_img = vis.shape[:2]
        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        w = min(w, w_img - x)
        h = min(h, h_img - y)
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 绿色框
    
    # 显示图像
    cv2.imshow("Mask", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 示例用法
if __name__ == "__main__":
    label_path = "D:/QY/Datasets/WHU-CD/before_label/tile_0_0.png"      # 替换为你的标签图片路径
    output_dir = "single_masks"   # 输出文件夹
    
    # 提取所有建筑物掩码
    masks = extract_single_masks(label_path, output_dir=None)
    print(f"共提取 {len(masks)} 个建筑物掩码")

    for item in masks:
        mask, bbox, points = item.values()

        # mask = masks[3]["mask"]
        # bbox = masks[3]["bbox"]
        visualize_mask(mask, bbox)