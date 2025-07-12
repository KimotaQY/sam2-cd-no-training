import gc
import os
import cv2
import numpy as np
import torch
import statistics

from BiSAM_CD import step_one, compute_mask_iou, compute_mask_iou_batch
from sam2.build_sam import build_sam2_video_predictor
from tools.misc import binary_accuracy, AverageMeter
from tools.extract_single_masks import extract_single_masks

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


def sum_masks_dict(masks_A, masks_B=None, iou_threshold=0.5):
    """
    直接对两mask求和（值可能超过1或255）,并执行归一化
    返回:
        sum_mask: 相同shape的矩阵，值为 mask1 + mask2
    """
    # 处理空输入
    if not masks_A and (masks_B is None or not masks_B):
        # 获取参考shape（若无法获取，抛出异常或指定默认shape）
        try:
            ref_shape = next(iter(masks_A.values())).shape
        except StopIteration:
            ref_shape = (1, 1024, 1024)  # 默认shape
        return np.zeros(ref_shape, dtype=np.uint8)

    try:
        merged_mask = np.zeros_like(next(iter(masks_A.values())), dtype=np.uint8)
    except StopIteration:
        ref_shape = (1, 1024, 1024)  # 默认shape
        merged_mask = np.zeros(ref_shape, dtype=np.uint8)

    # 没有对比的masks，直接返回合并后的mask
    if masks_B is None:
        for mask in masks_A.values():
            merged_mask = np.logical_or(merged_mask, mask > 0).astype(np.uint8)
        return merged_mask

    # # 逐个对比masks中的mask的iou，过高的移除
    # keys_to_remove = {"A": [], "B": []}
    # for obj_id_A, mask_A in masks_A.items():
    #     mask_A_binary = (mask_A > 0).astype(np.uint8)
    #     for obj_id_B, mask_B in masks_B.items():
    #         mask_B_binary = (mask_B > 0).astype(np.uint8)

    #         # 计算IoU（忽略全零mask的情况）
    #         if np.any(mask_B_binary) or np.any(mask_A_binary):
    #             iou = compute_mask_iou(mask_B_binary.flatten(), mask_A_binary.flatten())
    #             if iou > iou_threshold:
    #                 if obj_id_A not in keys_to_remove["A"]:
    #                     keys_to_remove["A"].append(obj_id_A)
    #                 if obj_id_B not in keys_to_remove["B"]:
    #                     keys_to_remove["B"].append(obj_id_B)
    #                 # # 删除字典中对应key
    #                 # if obj_id_A in masks_A:
    #                 #     del masks_A[obj_id_A]
    #                 # if obj_id_B in masks_B:
    #                 #     del masks_B[obj_id_B]

    # 将 masks_A 和 masks_B 转换为 NumPy 数组
    mask_array_A = np.array([m > 0 for m in masks_A.values()])
    mask_array_B = np.array([m > 0 for m in masks_B.values()])

    # 计算所有 mask 对的 IoU
    iou_matrix = compute_mask_iou_batch(mask_array_A, mask_array_B)

    # 找出需要删除的 key
    keys_to_remove = {"A": [], "B": []}
    for idx_A, obj_id_A in enumerate(masks_A.keys()):
        for idx_B, obj_id_B in enumerate(masks_B.keys()):
            if iou_matrix[idx_A, idx_B] > iou_threshold:
                if obj_id_A not in keys_to_remove["A"]:
                    keys_to_remove["A"].append(obj_id_A)
                if obj_id_B not in keys_to_remove["B"]:
                    keys_to_remove["B"].append(obj_id_B)

    for obj_id, mask in masks_A.items():
        if obj_id not in keys_to_remove["A"]:
            merged_mask = np.logical_or(merged_mask, mask > 0).astype(np.uint8)

    for obj_id, mask in masks_B.items():
        if obj_id not in keys_to_remove["B"]:
            merged_mask = np.logical_or(merged_mask, mask > 0).astype(np.uint8)

    # sum_mask = mask1.astype(np.float32) + mask2.astype(np.float32)
    # return sum_mask / np.max(sum_mask)
    return merged_mask


def main(
    model_type,
    mid_frame,
    diff_frame_num,
    iou_threshold,
    prompt_type="box",
    label_type="whu",
    obj_num=1,
    **kwargs,
):
    # global sub_log
    # sub_log["model_type"] = model_type
    # sub_log["mid_frame"] = mid_frame
    # sub_log["prompt_type"] = prompt_type
    # sub_log["obj_num"] = obj_num
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
    sam2_checkpoint = os.path.join("E:/CD_Checkpoints", checkpoint)
    model_cfg = os.path.join(
        "E:/CD_projects/sam2-cd-no-training/sam2/configs/sam2.1", config
    )

    # 输入前后时相图片
    T1 = kwargs.get("T1", None)
    T2 = kwargs.get("T2", None)
    diff_label_dir = kwargs.get("diff_label_dir", None)
    T1_label = kwargs.get("T1_label", None)
    T2_label = kwargs.get("T2_label", None)

    if None in [T1, T2, diff_label_dir, T1_label, T2_label]:
        # print("请输入前后时相图片路径和标签路径")
        # return
        T1 = "E:/CD_datasets/WHU-CD//test/A"
        T2 = "E:/CD_datasets/WHU-CD//test/B"
        diff_label_dir = "E:/CD_datasets/WHU-CD/test/label"
        T1_label = "E:/CD_datasets/WHU-CD//before_label"
        T2_label = "E:/CD_datasets/WHU-CD//after_label"

    # if label_type == "whu":
    #     T1 = "E:/CD_datasets/WHU-CD//test/A"
    #     T2 = "E:/CD_datasets/WHU-CD//test/B"
    #     diff_label_dir = "E:/CD_datasets/WHU-CD/test/label"
    #     T1_label = "E:/CD_datasets/WHU-CD//before_label"
    #     T2_label = "E:/CD_datasets/WHU-CD//after_label"
    # else:
    #     T1 = "E:/CD_datasets/LEVIR-CD/test/A"
    #     T2 = "E:/CD_datasets/LEVIR-CD/test/B"
    #     diff_label_dir = "E:/CD_datasets/LEVIR-CD/test/label"
    #     T1_label = "E:/CD_datasets/LEVIR-CD//test/A_sampoly_merge/result_fixed"
    #     T2_label = "E:/CD_datasets/LEVIR-CD//test/B_sampoly_merge/result_fixed"

    # 读取前后时相路径中的所有文件名
    img_names = [p for p in os.listdir(T1) if os.path.splitext(p)[-1] in [".png"]]

    # img_names = ["tile_7168_8192.png"]

    # output_dir = f"./logs/temp/test"
    output_dir = f"./logs/temp/{label_type}_sam{model_type}_{prompt_type}_mid{mid_frame}_{diff_frame_num}_iou{iou_threshold}"

    # 存在的文件夹则跳过
    if os.path.isdir(output_dir):
        print(f"{output_dir} 已存在")
        return
    else:
        os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "log.txt"), "w", encoding="utf-8") as f:
        F1_meter = AverageMeter()
        IoU_meter = AverageMeter()
        Acc_meter = AverageMeter()
        Pre_meter = AverageMeter()
        Rec_meter = AverageMeter()

        for idx, img_name in enumerate(img_names):
            # # 获取前后时相的prompts
            # prompts = {}
            # T1_prompts = None
            # T2_prompts = None
            # for i, label_dir in enumerate([T1_label, T2_label]):
            #     if label_type != "whu":
            #         label_path = os.path.join(label_dir, Path(img_name).stem + ".json")
            #         with open(label_path, "r") as f:
            #             json_result = json.load(f)
            #             masks = json_result["objects"]
            #     else:
            #         # 获取label中建筑物mask、box、points，并逐个赋予id进行追踪
            #         masks = extract_single_masks(os.path.join(label_dir, img_name))
            #     print(f"建筑物mask数量: {len(masks)}")

            #     if i == 0:
            #         T1_prompts = masks
            #     else:
            #         T2_prompts = masks
            #     # prompts["T1" if i == 0 else "T2"] = masks

            # # 将prompts进行分段处理
            # prompts = {}
            # segment_size = kwargs.get("segment_size", 1)
            # mask_list = []
            # for idx in range(0, max(len(T1_prompts), len(T2_prompts)), segment_size):
            #     prompts[f"T1"] = T1_prompts[idx : idx + segment_size]
            #     prompts[f"T2"] = T2_prompts[idx : idx + segment_size]

            predictor = build_sam2_video_predictor(
                model_cfg, sam2_checkpoint, device=device
            )

            diff_mask_list = step_one(
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
                # prompts=prompts,
            )

            diff_mask = sum_masks_dict(*diff_mask_list, iou_threshold=iou_threshold)
            # mask_list.append(diff_mask)

            # # 合并diff_masks
            # # 初始化 merged_mask 为全零矩阵，形状与 mask 相同
            # merged_mask = np.zeros_like(mask_list[0], dtype=np.uint8)

            # # 遍历 mask_list 并依次合并
            # for mask in mask_list:
            #     binary_mask = (mask > 0).astype(np.uint8)  # 确保是二值化的 mask
            #     merged_mask = np.logical_or(merged_mask, binary_mask).astype(np.uint8)

            # 读取标签图（单通道）
            label_mask = cv2.imread(
                os.path.join(diff_label_dir, img_name), cv2.IMREAD_GRAYSCALE
            )
            # iou = compute_mask_iou(diff_mask, label_mask)
            acc, precision, recall, f1, iou = binary_accuracy(diff_mask, label_mask)

            F1_meter.update(f1)
            Acc_meter.update(acc)
            IoU_meter.update(iou)
            Pre_meter.update(precision)
            Rec_meter.update(recall)

            print(
                f"{idx+1}/{len(img_names)} iou: {iou} f1: {f1} pre: {precision} rec: {recall} acc:{format(acc*100,'.2f')}"
            )
            f.write(
                f"{idx+1}/{len(img_names)} f1: {format(f1*100,'.2f')} iou: {format(iou*100,'.2f')} pre: {format(precision*100,'.2f')} rec: {format(recall*100,'.2f')} acc:{format(acc*100,'.2f')} name: {img_name}\n"
            )

            # 保存mask
            h, w = diff_mask.shape[-2:]
            mask_image = diff_mask.reshape(h, w, 1)
            cv2.imwrite(os.path.join(output_dir, img_name), mask_image * 255)

            if "predictor" in locals():
                del predictor
            torch.cuda.empty_cache()

        try:
            print(
                f"平均值 iou: {IoU_meter.avg} f1: {F1_meter.avg} pre: {Pre_meter.avg} rec: {Rec_meter.avg} acc:{Acc_meter.avg}"
            )
            f.write(
                f"平均值 iou: {IoU_meter.avg} f1: {F1_meter.avg} pre: {Pre_meter.avg} rec: {Rec_meter.avg} acc:{Acc_meter.avg}"
            )
        except statistics.StatisticsError:
            print("列表为空，无法计算平均值")


def test(
    model_type,
    mid_frame,
    diff_frame_num,
    iou_threshold,
    T1,
    T2,
    diff_label_dir,
    T1_label,
    T2_label,
    dataset_name,
    label_type="levir",
    prompt_type="box",
):
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
    sam2_checkpoint = os.path.join("E:/CD_Checkpoints", checkpoint)
    model_cfg = os.path.join(
        "E:/CD_projects/sam2-cd-no-training/sam2/configs/sam2.1", config
    )

    # 输入前后时相图片
    # T1 = kwargs.get("T1", None)
    # T2 = kwargs.get("T2", None)
    # diff_label_dir = kwargs.get("diff_label_dir", None)
    # T1_label = kwargs.get("T1_label", None)
    # T2_label = kwargs.get("T2_label", None)

    if None in [T1, T2, diff_label_dir, T1_label, T2_label]:
        print("请输入前后时相图片路径和标签路径")
        return

    # 读取前后时相路径中的所有文件名
    img_names = [p for p in os.listdir(T1) if os.path.splitext(p)[-1] in [".png"]]

    # 获取prompt的文件夹名称
    if label_type != "sam2":
        prompt_folder_name = os.path.normpath(T2_label).split(os.sep)[-2]
        prompt_folder_name = prompt_folder_name.split("_", 1)[1]
    else:
        prompt_folder_name = os.path.normpath(T2_label).split(os.sep)[-1]
        prompt_folder_name = prompt_folder_name.split("_", 1)[1]

    output_dir = f"./logs/{dataset_name}/generate_{model_type}_{prompt_type}_mid{mid_frame}_{diff_frame_num}_iou{iou_threshold}/{prompt_folder_name}"

    # 存在的文件夹则跳过
    if os.path.isdir(output_dir):
        print(f"{output_dir} 已存在")
        return
    else:
        os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "log.txt"), "w", encoding="utf-8") as f:
        F1_meter = AverageMeter()
        IoU_meter = AverageMeter()
        Acc_meter = AverageMeter()
        Pre_meter = AverageMeter()
        Rec_meter = AverageMeter()

        for idx, img_name in enumerate(img_names):
            predictor = build_sam2_video_predictor(
                model_cfg, sam2_checkpoint, device=device
            )

            diff_mask_list = step_one(
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
                # prompts=prompts,
            )

            diff_mask = sum_masks_dict(*diff_mask_list, iou_threshold=iou_threshold)

            # 读取标签图（单通道）
            label_mask = cv2.imread(
                os.path.join(diff_label_dir, img_name), cv2.IMREAD_GRAYSCALE
            )
            # iou = compute_mask_iou(diff_mask, label_mask)
            acc, precision, recall, f1, iou = binary_accuracy(diff_mask, label_mask)

            F1_meter.update(f1)
            Acc_meter.update(acc)
            IoU_meter.update(iou)
            Pre_meter.update(precision)
            Rec_meter.update(recall)

            print(
                f"{idx+1}/{len(img_names)} iou: {iou} f1: {f1} pre: {precision} rec: {recall} acc:{format(acc*100,'.2f')}"
            )
            f.write(
                f"{idx+1}/{len(img_names)} f1: {format(f1*100,'.2f')} iou: {format(iou*100,'.2f')} pre: {format(precision*100,'.2f')} rec: {format(recall*100,'.2f')} acc:{format(acc*100,'.2f')} name: {img_name}\n"
            )

            # 保存mask
            h, w = diff_mask.shape[-2:]
            mask_image = diff_mask.reshape(h, w, 1)
            cv2.imwrite(os.path.join(output_dir, img_name), mask_image * 255)

            if "predictor" in locals():
                del predictor
            torch.cuda.empty_cache()
            gc.collect()

        try:
            print(
                f"平均值 iou: {IoU_meter.avg} f1: {F1_meter.avg} pre: {Pre_meter.avg} rec: {Rec_meter.avg} acc:{Acc_meter.avg}"
            )
            f.write(
                f"平均值 iou: {IoU_meter.avg} f1: {F1_meter.avg} pre: {Pre_meter.avg} rec: {Rec_meter.avg} acc:{Acc_meter.avg}"
            )
        except statistics.StatisticsError:
            print("列表为空，无法计算平均值")


# if __name__ == "__main__":
#     # 读取文件夹中所有子文件夹的名称
#     input_dir = "E:/CD_datasets/WHU-CD/test"
#     exclude_folders = ["A", "B", "label"]
#     folder_names = [
#         name
#         for name in os.listdir(input_dir)
#         if os.path.isdir(os.path.join(input_dir, name)) and name not in exclude_folders
#     ]
#     # 将文件夹名称去掉第一个_之后的所有字符进行保存
#     folder_names = [name[name.index("_") + 1 :] for name in folder_names]
#     # 去除掉掉重复的文件夹名称
#     folder_names = list(set(folder_names))
#     print(folder_names)

#     model_type = "l"  # SAM2参数
#     mid_frame = 1  # 插帧数
#     prompt_type = "mask"  # 提示类型

#     T1 = "E:/CD_datasets/WHU-CD//test/A"
#     T2 = "E:/CD_datasets/WHU-CD//test/B"
#     diff_label_dir = "E:/CD_datasets/WHU-CD/test/label"
#     T1_label = "E:/CD_datasets/WHU-CD//before_label"
#     T2_label = "E:/CD_datasets/WHU-CD//after_label"
#     main(
#         model_type,
#         mid_frame,
#         -1,
#         0.5,
#         prompt_type,
#         label_type="whu",
#         T1=T1,
#         T2=T2,
#         T1_label=T1_label,
#         T2_label=T2_label,
#         diff_label_dir=diff_label_dir,
#     )

if __name__ == "__main__":
    test(
        "b+",
        1,
        -1,
        0.5,
        "E:/CD_datasets/LEVIR-CD/test/A",
        "E:/CD_datasets/LEVIR-CD/test/B",
        "E:/CD_datasets/LEVIR-CD/test/label",
        "E:/CD_datasets/LEVIR-CD/test/sam2/A_sam2_coco_rle_b+",
        "E:/CD_datasets/LEVIR-CD/test/sam2/B_sam2_coco_rle_b+",
        dataset_name="LEVIR-CD",
        label_type="sam2",
        prompt_type="mask",
    )
