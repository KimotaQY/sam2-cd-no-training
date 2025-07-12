import os
import cv2
import numpy as np


def add_images_and_binarize(image_path1, image_path2, output_path):
    # 读取两张图片（以灰度图形式读取）
    img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
    # 将白色背景（255）设置为0
    img1[img1 == 255] = 0
    img2[img2 == 255] = 0

    # 确保图片尺寸一致
    if img1.shape != img2.shape:
        raise ValueError("两张图片的尺寸不一致，无法相加")

    # 图片相加
    added_image = cv2.add(img1, img2)

    # 将相加后的图片二值化（阈值为255的一半，即127）
    _, binary_image = cv2.threshold(added_image, 1, 255, cv2.THRESH_BINARY)

    # 保存结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, binary_image)
    print(f"图片相加完成，保存在：{output_path}")


if __name__ == "__main__":
    input_dir_1 = "E:/CD_datasets/SECOND/test/label1"
    input_dir_2 = "E:/CD_datasets/SECOND/test/label2"
    output_dir = "E:/CD_datasets/SECOND/test/label"
    img_list_1 = os.listdir(input_dir_1)
    img_list_2 = os.listdir(input_dir_2)

    for img in img_list_1:
        img_path_1 = os.path.join(input_dir_1, img)
        img_path_2 = os.path.join(input_dir_2, img)
        output_path = os.path.join(output_dir, img)
        add_images_and_binarize(img_path_1, img_path_2, output_path)
