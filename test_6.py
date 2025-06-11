# 测试不同帧数和不同追踪目标数对性能的影响（运行时间、FLOPs等）

import csv
import json
import os
from pathlib import Path
import shutil

# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )


def linear_color_interpolation(img1, img2, alpha):
    """
    线性颜色插值
    :param img1: T1图像（BGR格式）
    :param img2: T2图像（BGR格式）
    :param alpha: 插值权重（0为全T1，1为全T2）
    :return: 中间帧图像
    """
    # 提取RGB通道（忽略Alpha）
    img1_rgb = img1[:, :, :3]
    img2_rgb = img2[:, :, :3]

    # 线性插值
    interpolated_rgb = (1 - alpha) * img1_rgb + alpha * img2_rgb
    interpolated_rgb = interpolated_rgb.astype(np.uint8)
    return interpolated_rgb


def gen_frame(folder_paths, filename, output_dir="output_jpg", sort="asc", mid_frame=0):
    # 根据排序方式决定遍历顺序
    paths_to_process = folder_paths if sort == "asc" else list(reversed(folder_paths))

    # 清空文件夹内容
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # 确保输出文件夹存在（只需要检查一次）
    os.makedirs(output_dir, exist_ok=True)

    for idx, folder_path in enumerate(paths_to_process):
        # 构造输入和输出路径
        input_path = os.path.join(folder_path, filename)
        output_filename = f"{idx + 1}.jpg" if idx == 0 else f"{idx + mid_frame + 1}.jpg"
        output_path = os.path.join(output_dir, output_filename)

        # 打开PNG图片并转换为RGB模式（JPG不支持PNG的RGBA透明度）
        try:
            with Image.open(input_path) as img:
                if img.mode in ("RGBA", "LA"):
                    # 创建一个白色背景的RGB图像
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1])  # 使用alpha通道作为mask
                    img = background
                elif img.mode != "RGB":
                    img = img.convert("RGB")

                # 保存为JPG
                img.save(output_path, "JPEG", quality=90)
                print(f"转换成功: {filename} -> {os.path.basename(output_path)}")
        except Exception as e:
            print(f"转换失败 {filename}: {str(e)}")

    def generate_uniform_alphas(num_frames):
        """生成均匀间隔的alpha值"""
        return [i / (num_frames + 1) for i in range(1, num_frames + 1)]

    # 生成中间帧
    alphas = generate_uniform_alphas(mid_frame)
    for idx, alpha in enumerate(alphas):
        folder_path = paths_to_process[0]
        # 构造输入和输出路径
        input_path = os.path.join(folder_path, filename)
        output_filename = f"{idx + 2}.jpg"
        output_path = os.path.join(output_dir, output_filename)

        # 打开PNG图片并转换为RGB模式（JPG不支持PNG的RGBA透明度）
        try:
            with Image.open(input_path) as img:
                if img.mode in ("RGBA", "LA"):
                    # 创建一个白色背景的RGB图像
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1])  # 使用alpha通道作为mask
                    img = background
                elif img.mode != "RGB":
                    img = img.convert("RGB")

                first_frame = os.path.join(paths_to_process[0], filename)
                final_frame = os.path.join(paths_to_process[-1], filename)
                img = linear_color_interpolation(
                    cv2.imread(first_frame, cv2.IMREAD_UNCHANGED),
                    cv2.imread(final_frame, cv2.IMREAD_UNCHANGED),
                    alpha=alpha,
                )
                # 保存为JPG
                cv2.imwrite(output_path, img)
                # img.save(output_path, "JPEG", quality=90)
                print(f"转换成功: {filename} -> {os.path.basename(output_path)}")
        except Exception as e:
            print(f"转换失败 {filename}: {str(e)}")

    return os.path.dirname(output_path)


def add_new_obj(
    ann_frame_idx,
    ann_obj_id,
    points=None,
    labels=None,
    box=None,
    mask=None,
    predictor=None,
    inference_state=None,
):
    try:
        ann_frame_idx = ann_frame_idx  # the frame index we interact with
        ann_obj_id = ann_obj_id  # give a unique id to each object we interact with (it can be any integers)

        if points is not None or box is not None:
            # Let's add a positive click at (x, y) to get started
            points = np.array(points, dtype=np.float32) if points is not None else None
            # for labels, `1` means positive click and `0` means negative click
            labels = np.array(labels, np.int32) if labels is not None else None

            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                points=points,
                labels=labels,
                box=box,
            )

        if mask is not None:
            # 1. 将 OpenCV 掩码 (0,255) 转换为二进制 (0,1)
            binary_mask = (mask > 128).astype(np.uint8)  # 阈值化

            # 2. 转换为 PyTorch 张量，并转为布尔类型
            mask_tensor = torch.from_numpy(binary_mask).to(torch.bool)

            # 检查形状是否为 (H, W)
            assert mask_tensor.dim() == 2, f"Mask must be 2D, got {mask_tensor.shape}"

            _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                mask=mask_tensor,
            )
    except Exception as e:
        raise e  # 主动抛出错误

    return _, out_obj_ids, out_mask_logits


def compute_mask_iou(mask1, mask2):
    """
    计算两mask的IoU（交并比）差异
    返回:
        iou: 相似度（0~1，1表示完全相同）
        diff_mask: 差异区域（1表示不同，0表示相同）
    """
    intersection = np.logical_and(mask1 > 0, mask2 > 0)
    union = np.logical_or(mask1 > 0, mask2 > 0)
    sum_union = np.sum(union)
    if sum_union == 0:  # 两个 mask 都是全 0，认为完全相同
        return 1.0
    iou = np.sum(intersection) / sum_union
    # diff_mask = np.logical_xor(mask1 > 0, mask2 > 0).astype(np.uint8)
    return iou


def merge_masks(masks_dict, compare_masks_dict=None, iou_threshold=0.5):
    """
    合并当前帧的masks，但跳过与对比帧中高IoU的物体

    参数:
        masks_dict (dict): 当前帧的masks {obj_id: mask}
        compare_masks_dict (dict): 对比帧的masks {obj_id: mask}（可选）
        iou_threshold (float): IoU阈值，大于此值则跳过合并

    返回:
        merged_mask (dict): 保留下来的mask
    """
    merged_mask = {}

    # 如果没有对比帧，直接返回masks_dict
    if compare_masks_dict is None:
        return masks_dict

    # 遍历当前帧的每个物体
    for obj_id, mask in masks_dict.items():
        mask_binary = (mask > 0).astype(np.uint8)

        # 检查对比帧中是否存在高IoU的物体
        compare_mask = compare_masks_dict.get(obj_id)
        compare_binary = (compare_mask > 0).astype(np.uint8)

        # 计算IoU（忽略全零mask的情况）
        if np.any(compare_binary) or np.any(mask_binary):
            iou = compute_mask_iou(compare_binary.flatten(), mask_binary.flatten())
            # if iou < 1.0:
            #     print(f"iou: {iou}")
            if iou <= iou_threshold:
                # 仅合并低IoU的物体
                # print("合并")
                merged_mask[obj_id] = mask
                # 显示每个obj的iou
                # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
                # show_mask(mask, ax1, obj_id=obj_id)
                # ax1.set_title(f"IoU {iou} (Masks)")
                # plt.tight_layout()
                # plt.show()

    return merged_mask


from tools.extract_single_masks import extract_single_masks

logs = []
sub_log = {
    "model_type": "",
    "mid_frame": "",
    "prompt_type": "",
    "obj_num": 1,
    "time_cost": "",
    # "FLOPs": "",
    "memory_usage": "",
    "max_memory_usage": "",
}


def step_one(
    img_name,
    T1_dir,
    T2_dir,
    T1_label_dir,
    T2_label_dir,
    predictor=None,
    label_type="whu",
    mid_frame=0,
    diff_frame_num=1,
    iou_threshold=0.5,
    prompt_type="box",
):
    global sub_log
    max_obj_num = sub_log["obj_num"]
    #
    diff_mask_list = []

    for i, label_dir in enumerate([T1_label_dir, T2_label_dir]):
        # 生成顺序jpg
        video_dir = gen_frame(
            [T1_dir, T2_dir],
            img_name,
            sort="asc" if i == 0 else "desc",
            mid_frame=mid_frame,
        )

        # scan all the JPEG frame names in this directory
        frame_names = [
            p
            for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        inference_state = predictor.init_state(video_path=video_dir)

        # track objects
        predictor.reset_state(inference_state)

        if label_type != "whu":
            label_path = os.path.join(label_dir, Path(img_name).stem + ".json")
            with open(label_path, "r") as f:
                json_result = json.load(f)
                masks = json_result["objects"]
        else:
            # 获取label中建筑物mask、box、points，并逐个赋予id进行追踪
            masks = extract_single_masks(os.path.join(label_dir, img_name))
            print(f"{img_name}建筑物mask数量: {len(masks)}")
            if len(masks) >= max_obj_num:
                masks = masks[:max_obj_num]
        print(f"建筑物mask数量: {len(masks)}")

        # 获取追踪结果
        # run propagation throughout the video and collect the results in a dict
        video_segments = {}
        # 每栋建筑单独预测（**测试**）
        # for idx, item in enumerate(masks):
        #     if label_type == "whu":
        #         mask, (x, y, w, h), points = item.values()
        #     else:
        #         box = item.get("bbox")
        #         score = item.get("score")
        #         # if score < 0.7:
        #         #     continue

        #     ann_list = []
        #     for frame_idx in range(mid_frame + 1):
        #         if prompt_type == "points":
        #             # 使用points
        #             labels = [1]
        #             ann_list.append(
        #                 {
        #                     "ann_frame_idx": frame_idx,
        #                     "ann_obj_id": idx + 1,
        #                     "points": points,
        #                     "labels": labels,
        #                     # "box": np.array([x, y, x + w, y + h]),
        #                 }
        #             )
        #         elif prompt_type == "box":
        #             # 使用box
        #             ann_list.append(
        #                 {
        #                     "ann_frame_idx": frame_idx,
        #                     "ann_obj_id": idx + 1,
        #                     "box": (
        #                         np.array([x, y, x + w, y + h])
        #                         if label_type == "whu"
        #                         else np.array(box)
        #                     ),
        #                 }
        #             )
        #         else:
        #             # 使用mask
        #             ann_list.append(
        #                 {
        #                     "ann_frame_idx": frame_idx,
        #                     "ann_obj_id": idx + 1,
        #                     "mask": mask,
        #                 }
        #             )

        #     # 将ann_list导入predictor
        #     try:
        #         for index, item in enumerate(ann_list):
        #             _, out_obj_ids, out_mask_logits = add_new_obj(
        #                 **item, predictor=predictor, inference_state=inference_state
        #             )

        #     except Exception as e:
        #         raise e

        #     if len(ann_list) != 0:
        #         for (
        #             out_frame_idx,
        #             out_obj_ids,
        #             out_mask_logits,
        #         ) in predictor.propagate_in_video(inference_state):
        #             if out_frame_idx not in video_segments:
        #                 video_segments[out_frame_idx] = {}
        #             for i, out_obj_id in enumerate(out_obj_ids):
        #                 video_segments[out_frame_idx][out_obj_id] = (
        #                     (out_mask_logits[i] > 0.0).cpu().numpy()
        #                 )

        #     predictor.reset_state(inference_state)

        # 建筑合并预测（**测试**）
        ann_list = []
        import time

        time_cost = 0
        for frame_idx in range(mid_frame + 1):
            for idx, item in enumerate(masks):
                if label_type == "whu":
                    mask, (x, y, w, h), points = item.values()
                else:
                    box = item.get("bbox")
                    score = item.get("score")
                    # if score < 0.7:
                    #     continue

                if prompt_type == "points":
                    # 使用points
                    labels = [1]
                    ann_list.append(
                        {
                            "ann_frame_idx": frame_idx,
                            "ann_obj_id": idx + 1,
                            "points": points,
                            "labels": labels,
                            # "box": np.array([x, y, x + w, y + h]),
                        }
                    )
                elif prompt_type == "box":
                    # 使用box
                    ann_list.append(
                        {
                            "ann_frame_idx": frame_idx,
                            "ann_obj_id": idx + 1,
                            "box": (
                                np.array([x, y, x + w, y + h])
                                if label_type == "whu"
                                else np.array(box)
                            ),
                        }
                    )
                else:
                    # 使用mask
                    ann_list.append(
                        {
                            "ann_frame_idx": frame_idx,
                            "ann_obj_id": idx + 1,
                            "mask": mask,
                        }
                    )

            start_time = time.time()
            # 将ann_list导入predictor
            try:
                for index, item in enumerate(ann_list):
                    _, out_obj_ids, out_mask_logits = add_new_obj(
                        **item, predictor=predictor, inference_state=inference_state
                    )

            except Exception as e:
                raise e

            end_time = time.time()

            if len(ann_list) != 0:
                for (
                    out_frame_idx,
                    out_obj_ids,
                    out_mask_logits,
                ) in predictor.propagate_in_video(inference_state):
                    if out_frame_idx not in video_segments:
                        video_segments[out_frame_idx] = {}
                    for i, out_obj_id in enumerate(out_obj_ids):
                        video_segments[out_frame_idx][out_obj_id] = (
                            (out_mask_logits[i] > 0.0).cpu().numpy()
                        )
            time_cost += end_time - start_time

            predictor.reset_state(inference_state)

        print(f"{img_name} 耗时: {time_cost}")
        print(f"已分配显存: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        print(f"最大显存使用: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")
        sub_log["time_cost"] = time_cost
        sub_log["memory_usage"] = torch.cuda.memory_allocated() / 1024**2
        sub_log["max_memory_usage"] = torch.cuda.max_memory_allocated() / 1024**2

        filename = "log.csv"
        file_exists = os.path.isfile(filename)
        with open(filename, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sub_log.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(sub_log)

        # mask合并显示
        segments_len = len(video_segments)
        if segments_len == 0:
            diff_mask = {}
        else:
            # 首尾帧比较
            diff_mask = merge_masks(
                video_segments[0 if diff_frame_num == 1 else segments_len - 2],
                compare_masks_dict=video_segments[segments_len - 1],
                iou_threshold=iou_threshold,
            )

        diff_mask_list.append(diff_mask)

        del video_segments
        torch.cuda.empty_cache()  # 清理 PyTorch 的 CUDA 缓存

    # 显式释放 predictor
    del predictor

    return diff_mask_list


from sam2.build_sam import build_sam2_video_predictor


def main(
    model_type,
    mid_frame,
    diff_frame_num,
    iou_threshold,
    prompt_type="box",
    label_type="whu",
    obj_num=1,
):
    global sub_log
    sub_log["model_type"] = model_type
    sub_log["mid_frame"] = mid_frame
    sub_log["prompt_type"] = prompt_type
    sub_log["obj_num"] = obj_num
    model_obj = {
        "t": {
            "checkpoint": "sam2.1_hiera_tiny.pt",
            "config": "sam2.1_hiera_t.yaml",
        },
        "s": {
            "checkpoint": "sam2.1_hiera_small.pt",
            "config": "sam2.1_hiera_s.yaml",
        },
        "b+": {
            "checkpoint": "sam2.1_hiera_base_plus.pt",
            "config": "sam2.1_hiera_b+.yaml",
        },
        "l": {
            "checkpoint": "sam2.1_hiera_large.pt",
            "config": "sam2.1_hiera_l.yaml",
        },
    }

    checkpoint = model_obj[model_type]["checkpoint"]
    config = model_obj[model_type]["config"]
    # 加载SAM2 video predictor
    sam2_checkpoint = os.path.join("E:\CD_Checkpoints", checkpoint)
    model_cfg = os.path.join(
        "E:\CD_projects\sam2-cd-no-training\sam2\configs\sam2.1", config
    )

    # 输入前后时相图片
    if label_type == "whu":
        T1 = "E:\CD_datasets\WHU-CD\\test\A"
        T2 = "E:\CD_datasets\WHU-CD\\test\B"
        diff_label_dir = "E:\CD_datasets\WHU-CD\\test\label"
        T1_label = "E:\CD_datasets\WHU-CD\\before_label"
        T2_label = "E:\CD_datasets\WHU-CD\\after_label"
    else:
        T1 = "E:\CD_datasets\LEVIR-CD\\test\A"
        T2 = "E:\CD_datasets\LEVIR-CD\\test\B"
        diff_label_dir = "E:\CD_datasets\LEVIR-CD\\test\label"
        T1_label = "E:\CD_datasets\LEVIR-CD\\test\A_sampoly_merge\\result_fixed"
        T2_label = "E:\CD_datasets\LEVIR-CD\\test\B_sampoly_merge\\result_fixed"

    # 读取前后时相路径中的所有文件名
    # img_names = [p for p in os.listdir(T1) if os.path.splitext(p)[-1] in [".png"]]

    img_names = ["tile_7168_8192.png"]

    for idx, img_name in enumerate(img_names):
        predictor = build_sam2_video_predictor(
            model_cfg, sam2_checkpoint, device=device
        )

        step_one(
            img_name,
            T1,
            T2,
            T1_label,
            T2_label,
            predictor,
            label_type=label_type,
            mid_frame=mid_frame,
            diff_frame_num=diff_frame_num,
            iou_threshold=iou_threshold,
            prompt_type=prompt_type,
        )

        if "predictor" in locals():
            del predictor
        torch.cuda.empty_cache()


import copy

if __name__ == "__main__":
    model_types = ["t"]
    # model_types = ["t", "s", "b+", "l"]
    prompt_types = ["points", "box", "mask"]
    mid_frames = [0, 1, 2]
    obj_nums = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # print(sub_log)
    for model_type in model_types:
        for prompt_type in prompt_types:
            for mid_frame in mid_frames:
                for obj_num in obj_nums:
                    sub_log["model_type"] = model_type
                    sub_log["mid_frame"] = mid_frame
                    sub_log["prompt_type"] = prompt_type
                    sub_log["obj_num"] = obj_num
                    main(model_type, mid_frame, 1, 0.5, prompt_type, label_type="whu")
                    print(sub_log)
                    logs.append(copy.deepcopy(sub_log))
    # main("b+", 1, 1, 0.5, "box", label_type="whu")
    # logs.append(sub_log)
    # 定义CSV文件的列名，基于 sub_log 的键
    fieldnames = [
        "model_type",
        "mid_frame",
        "prompt_type",
        "obj_num",
        "time_cost",
        "memory_usage",
        "max_memory_usage",
    ]

    # 写入CSV文件
    with open("output_logs.csv", mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # 写入表头
        writer.writeheader()

        # 写入日志数据
        for log in logs:
            writer.writerow(log)

    print("日志已成功写入 output_logs.csv")
