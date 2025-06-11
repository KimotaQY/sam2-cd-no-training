import json
from pathlib import Path
import xml.etree.ElementTree as ET
import os


def json_to_voc(bbox_list, image_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 创建XML根节点
    root = ET.Element("annotation")

    # 添加图片信息
    ET.SubElement(root, "filename").text = os.path.basename(image_path)
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = "1024"  # 替换为实际图片宽度
    ET.SubElement(size, "height").text = "1024"  # 替换为实际图片高度

    # 添加每个bbox
    for bbox in bbox_list:
        score = bbox.get("score")
        if score < 0.7:
            continue
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = "building"  # 你的类别名称
        bndbox = ET.SubElement(obj, "bndbox")
        x1, y1, x2, y2 = bbox["bbox"]
        ET.SubElement(bndbox, "xmin").text = str(x1)
        ET.SubElement(bndbox, "ymin").text = str(y1)
        ET.SubElement(bndbox, "xmax").text = str(x2)
        ET.SubElement(bndbox, "ymax").text = str(y2)

    # 保存XML文件
    tree = ET.ElementTree(root)
    xml_path = os.path.join(
        output_dir, os.path.splitext(os.path.basename(image_path))[0] + ".xml"
    )
    tree.write(xml_path)


def voc_to_json(xml_dir, output_json_dir):
    """
    将VOC格式的XML文件夹转换为目标JSON格式
    :param xml_dir: 存放VOC XML文件的目录路径
    :param output_json_dir: 输出的JSON文件路径
    """
    os.makedirs(output_json_dir, exist_ok=True)

    box_sum = 0

    # 遍历XML目录
    for xml_file in os.listdir(xml_dir):
        if not xml_file.endswith(".xml"):
            continue

        json_data = []

        xml_path = os.path.join(xml_dir, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 提取图片基础信息（可选）
        filename = root.find("filename").text
        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)

        # 处理每个检测框
        for obj in root.findall("object"):
            cls_name = obj.find("name").text
            bndbox = obj.find("bndbox")
            xmin = int(float(bndbox.find("xmin").text))
            ymin = int(float(bndbox.find("ymin").text))
            xmax = int(float(bndbox.find("xmax").text))
            ymax = int(float(bndbox.find("ymax").text))

            # 转换为你的JSON格式
            json_data.append(
                {
                    "image_name": filename,  # 可选字段
                    "bbox": [xmin, ymin, xmax, ymax],
                    "score": 0.5,  # 默认值（需从额外字段读取或自定义逻辑）
                    "class": cls_name,  # 可选字段
                }
            )

        # 保存为JSON文件
        output_json_path = os.path.join(output_json_dir, Path(filename).stem + ".json")
        with open(output_json_path, "w") as f:
            json.dump({"objects": json_data}, f, indent=4)

        print(f"{filename}转换成功，包含{len(json_data)}个标注框")

        box_sum += len(json_data)

    print(f"转换完成，共处理 {box_sum} 个标注框 -> 保存至 {output_json_dir}")


if __name__ == "__main__":
    # img_dir = "E:\CD_datasets\LEVIR-CD\\test\A"
    # label_dir = "E:\CD_datasets\LEVIR-CD\\test\A_sampoly_merge\\result"
    # output_dir = "E:\CD_datasets\LEVIR-CD\\test\A_sampoly_merge\\anno"

    # # 读取文件夹中图片名称
    # img_names = [p for p in os.listdir(img_dir) if os.path.splitext(p)[-1] in [".png"]]

    # for i, img_name in enumerate(img_names):
    #     img_path = os.path.join(img_dir, img_name)

    #     json_path = os.path.join(label_dir, Path(img_name).stem + ".json")
    #     with open(json_path, "r") as f:
    #         json_result = json.load(f)

    #     json_to_voc(json_result["objects"], img_path, output_dir)

    #     print(f"处理进度：{i+1}/{len(img_names)}")

    xml_dir = "E:\CD_datasets\LEVIR-CD\\test\B_sampoly_merge\\anno"
    output_dir = "E:\CD_datasets\LEVIR-CD\\test\B_sampoly_merge\\result_fixed"
    voc_to_json(xml_dir, output_dir)
