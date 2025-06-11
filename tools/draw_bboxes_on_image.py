import json
import os
import cv2
import numpy as np


def draw_bboxes_on_image(image_path, json_path, output_path=None):
    """
    在图像上绘制边界框

    参数:
        image_path: 底图路径
        json_path: 包含bbox信息的JSON文件路径
        output_path: 结果保存路径(可选)
    """
    # 1. 读取JSON文件
    with open(json_path, "r") as f:
        data = json.load(f)

    # 2. 加载底图
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法加载图像: {image_path}")

    # 3. 遍历所有对象并绘制bbox
    for obj in data.get("objects", []):
        bbox = obj.get("bbox")
        if bbox is None:  # 跳过没有bbox的对象
            continue

        # 解析bbox坐标 [x1, y1, x2, y2]
        x1, y1, x2, y2 = map(int, bbox)

        # 4. 绘制矩形框
        color = (0, 255, 0)  # 绿色 (BGR格式)
        thickness = 2
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # 绘制标签和分数
        label = f"{obj.get('category', 'unknown')} {obj.get('score', 0):.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        text_color = (255, 255, 255)  # 白色文字
        bg_color = (0, 0, 0)  # 黑色背景

        # 计算文本大小
        (text_width, text_height), _ = cv2.getTextSize(
            label, font, font_scale, thickness
        )

        # 绘制文本背景
        cv2.rectangle(
            image, (x1, y1 - text_height - 5), (x1 + text_width, y1), bg_color, -1
        )

        # 绘制文本
        cv2.putText(image, label, (x1, y1 - 5), font, font_scale, text_color, thickness)

    # 5. 显示或保存结果
    if output_path:
        cv2.imwrite(output_path, image)
        print(f"结果已保存到: {output_path}")
    else:
        cv2.imshow("Result", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# 使用示例
if __name__ == "__main__":
    img_dir = r"E:/CD_datasets/LEVIR-CD/test/B"
    json_dir = r"E:/CD_datasets/LEVIR-CD/test/before_label"
    output_dir = r"E:/CD_datasets/LEVIR-CD/test/A_with_box"

    # os.makedirs(output_dir, exist_ok=True)

    # # get all filenames
    # img_names = [p for p in os.listdir(img_dir) if os.path.splitext(p)[-1] in [".png"]]

    # for idx, img_name in enumerate(img_names):
    #     img_path = os.path.join(img_dir, img_name)
    #     json_path = os.path.join(json_dir, os.path.splitext(img_name)[0] + ".json")
    #     output_path = os.path.join(output_dir, os.path.splitext(img_name)[0] + ".jpg")

    #     print(f"{img_name}")
    #     draw_bboxes_on_image(img_path, json_path, output_path)

    img_path = os.path.join(img_dir, "test_13.png")
    draw_bboxes_on_image(img_path, "test_13.json")
