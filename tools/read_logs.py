import csv
import os
import re


if __name__ in "__main__":
    iou_list = [0.3, 0.4, 0.5, 0.6, 0.7]
    mid_list = [0, 1, 2, 3]
    diff_frame_num_list = [1, -1]
    model_obj = ["t", "s", "b+", "l"]
    prompt_type_list = ["points", "box", "mask"]
    output_csv = "output_acc.csv"  # 输出的csv文件路径
    with open(output_csv, "w", newline="") as csvfile:
        # 写入CSV文件
        writer = csv.writer(csvfile)

        # 写入表头
        writer.writerow(
            [
                "Model_type",
                "mid_frame",
                "iou_threshold",
                "prompt_type",
                "f1",
                "iou",
                "pre",
                "rec",
                "acc",
            ]
        )

        for model_type in model_obj:
            for mid_frame in mid_list:
                for iou_threshold in iou_list:
                    for prompt_type in prompt_type_list:
                        log_dir = f"E:/CD_projects/sam2-cd-no-training/logs/whu_sam{model_type}_{prompt_type}_mid{mid_frame}_{1}_iou{iou_threshold}"
                        # 定义输入和输出文件路径
                        log_path = os.path.join(log_dir, "log.txt")

                        if not os.path.isdir(log_dir):
                            print(
                                f"whu_sam{model_type}_{prompt_type}_mid{mid_frame}_{1}_iou{iou_threshold} 路径不存在"
                            )
                            continue

                        # 读取txt文件的最后一行
                        with open(log_path, "r", encoding="utf-8") as f:
                            lines = f.readlines()
                            last_line = lines[-1].strip()  # 获取最后一行并去除首尾空白

                        # 使用正则表达式提取数值
                        # 匹配模式示例：【平均值 iou: 0.7485393123413728 f1: 0.8158398633122981 pre: 0.8272123796246305 rec: 0.8383807316377816】
                        pattern = r"iou: (\d+\.\d+).*f1: (\d+\.\d+).*pre: (\d+\.\d+).*rec: (\d+\.\d+).*acc: (\d+\.\d+)"
                        match = re.search(pattern, last_line)

                        if match:
                            iou, f1, pre, rec, acc = match.groups()
                            if not acc:
                                continue

                            # 写入数据
                            writer.writerow(
                                [
                                    model_type,
                                    mid_frame,
                                    iou_threshold,
                                    prompt_type,
                                    f1,
                                    iou,
                                    pre,
                                    rec,
                                    acc,
                                ]
                            )

                            print(
                                f"数据已成功写入 {output_csv} iou: {iou} f1: {f1} pre: {pre} rec: {rec}"
                            )
                        else:
                            print(
                                f"whu_sam{model_type}_{prompt_type}_mid{mid_frame}_{1}_iou{iou_threshold} 未能从最后一行中提取所需数据，请检查文件格式。"
                            )
