from PIL import Image
import numpy as np
import os


def combine_images(image_paths, output_path):
    # 创建一个1024x1024的空白图像，用0值填充（黑色）
    combined_image = Image.new("RGB", (1024, 1024), (0, 0, 0))

    # 如果图像路径数量不足4，则用None填充列表以确保有四个元素
    image_paths += [None] * (4 - len(image_paths))

    for i, image_path in enumerate(image_paths):
        if image_path and os.path.exists(image_path):
            img = Image.open(image_path)
            img = img.resize((512, 512))  # 确保图像是512x512
        else:
            # 如果图像路径不存在或为None，则创建一个512x512的黑色图像
            img = Image.new("RGB", (512, 512), (0, 0, 0))

        # 计算图像放置的位置
        row = i // 2
        col = i % 2
        x = col * 512
        y = row * 512

        # 将图像粘贴到大图上
        combined_image.paste(img, (x, y))

    # 保存合成后的图像
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined_image.save(output_path)
    print(f"合成完成: {output_path}")


if __name__ == "__main__":
    input_dir = "E:/CD_datasets/SECOND/test/label"
    output_dir = "E:/CD_datasets/SECOND/test/label_1024"
    img_list = os.listdir(input_dir)
    # 每次读取4张图片
    for i in range(0, len(img_list), 4):
        batch_img_list = img_list[i : i + 4]

        # 将list中的图片名使用_拼接,去掉文件扩展名
        output_filename = "_".join([os.path.splitext(img)[0] for img in batch_img_list])
        output_path = os.path.join(output_dir, output_filename + ".png")

        image_paths = [os.path.join(input_dir, img) for img in batch_img_list]
        combine_images(image_paths, output_path)
