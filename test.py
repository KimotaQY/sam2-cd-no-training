import os
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




def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def add_new_obj(ann_frame_idx, ann_obj_id, points=None, labels=None, box=None, predictor=None):
    try:
        ann_frame_idx = ann_frame_idx  # the frame index we interact with
        ann_obj_id = ann_obj_id  # give a unique id to each object we interact with (it can be any integers)

        # Let's add a positive click at (x, y) to get started
        points = np.array(points, dtype=np.float32) if labels is not None else None
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
    except Exception as e:
        raise e  # 主动抛出错误
    
    return _, out_obj_ids, out_mask_logits


# ann_list = [
#     {
#         "ann_frame_idx": 0,
#         "ann_obj_id": 1,
#         "points": [[303,883]],
#         "labels": [1]
#     },
#     {
#         "ann_frame_idx": 0,
#         "ann_obj_id": 2,
#         "points": [[372.4,851.9]],
#         "labels": [1]
#     },
#     {
#         "ann_frame_idx": 0,
#         "ann_obj_id": 3,
#         "points": [[708.1,163.6]],
#         "labels": [1]
#     },
#     {
#         "ann_frame_idx": 0,
#         "ann_obj_id": 4,
#         "points": [[691,120.3]],
#         "labels": [1]
#     },
# ]


def _merge_masks(masks_dict):
    """
    合并同一帧的所有 masks（逻辑或操作）
    输入: 
        masks_dict (dict): {obj_id: mask, ...}
    返回: 
        merged_mask (np.ndarray): 合并后的二值化mask（0/1）
    """
    merged_mask = np.zeros_like(next(iter(masks_dict.values())), dtype=np.uint8)
    for mask in masks_dict.values():
        merged_mask = np.logical_or(merged_mask, mask > 0).astype(np.uint8)
    return merged_mask
    

def _merge_masks(masks_dict, compare_masks_dict=None, iou_threshold=0.5):
    """
    合并当前帧的masks，但跳过与对比帧中高IoU的物体
    
    参数:
        masks_dict (dict): 当前帧的masks {obj_id: mask}
        compare_masks_dict (dict): 对比帧的masks {obj_id: mask}（可选）
        iou_threshold (float): IoU阈值，大于此值则跳过合并
    
    返回:
        merged_mask (np.ndarray): 合并后的二值mask
    """
    merged_mask = np.zeros_like(next(iter(masks_dict.values())), dtype=np.uint8)
    
    # 如果没有对比帧，直接合并所有masks
    if compare_masks_dict is None:
        for mask in masks_dict.values():
            merged_mask = np.logical_or(merged_mask, mask > 0).astype(np.uint8)
        return merged_mask
    
    # 遍历当前帧的每个物体
    for obj_id, mask in masks_dict.items():
        mask_binary = (mask > 0).astype(np.uint8)
        
        # 检查对比帧中是否存在高IoU的物体
        compare_mask = compare_masks_dict.get(obj_id)
        compare_binary = (compare_mask > 0).astype(np.uint8)
            
        # 计算IoU（忽略全零mask的情况）
        if np.any(compare_binary) or np.any(mask_binary):
            iou = compute_mask_iou(compare_binary.flatten(), mask_binary.flatten())
            if iou < 1.0:
                print(f"iou: {iou}")
            if iou <= iou_threshold:
                # 仅合并低IoU的物体
                print("合并")
                merged_mask = np.logical_or(merged_mask, mask_binary).astype(np.uint8)
    
    return merged_mask


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
    
    # 如果没有对比帧，直接合并所有masks
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
            if iou < 1.0:
                print(f"iou: {iou}")
            if iou <= iou_threshold:
                # 仅合并低IoU的物体
                print("合并")
                merged_mask[obj_id] = mask
    
    return merged_mask


def compute_mask_diff(mask1, mask2):
    """
    计算两个mask的差异（变化区域）
    返回: 
        diff_mask (np.ndarray): 差异区域（1表示变化，0表示未变化）
    """
    # 确保mask是布尔类型或0/1二值
    mask1_bin = (mask1 > 0).astype(np.uint8)
    mask2_bin = (mask2 > 0).astype(np.uint8)
    
    # 求差集：在mask1但不在mask2，或反之
    diff_mask = np.bitwise_xor(mask1_bin, mask2_bin)
    return diff_mask

    # diff_mask = mask1_bin.astype(np.float32) - mask2_bin.astype(np.float32)
    # return np.where(diff_mask < 0, 0, diff_mask)


def compute_mask_iou(mask1, mask2):
    """
    计算两mask的IoU（交并比）差异
    返回:
        iou: 相似度（0~1，1表示完全相同）
        diff_mask: 差异区域（1表示不同，0表示相同）
    """
    intersection = np.logical_and(mask1 > 0, mask2 > 0)
    union = np.logical_or(mask1 > 0, mask2 > 0)
    iou = np.sum(intersection) / max(np.sum(union), 1e-6)
    # diff_mask = np.logical_xor(mask1 > 0, mask2 > 0).astype(np.uint8)
    return iou


def sum_masks(mask1, mask2):
    """
    直接对两mask求和（值可能超过1或255）,并执行归一化
    返回:
        sum_mask: 相同shape的矩阵，值为 mask1 + mask2
    """
    sum_mask = mask1.astype(np.float32) + mask2.astype(np.float32)
    # return sum_mask / np.max(sum_mask)
    return np.where(sum_mask >= 2, 0, sum_mask)


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
    
    # 逐个对比masks中的mask的iou，过高的移除
    keys_to_remove = {
        "A": [],
        "B": []
    }
    for obj_id_A, mask_A in masks_A.items():
        mask_A_binary = (mask_A > 0).astype(np.uint8)
        for obj_id_B, mask_B in masks_B.items():
            mask_B_binary = (mask_B > 0).astype(np.uint8)

            # 计算IoU（忽略全零mask的情况）
            if np.any(mask_B_binary) or np.any(mask_A_binary):
                iou = compute_mask_iou(mask_B_binary.flatten(), mask_A_binary.flatten())
                if iou > iou_threshold:
                    if obj_id_A not in keys_to_remove["A"]:
                        keys_to_remove["A"].append(obj_id_A)
                    if obj_id_B not in keys_to_remove["B"]:
                        keys_to_remove["B"].append(obj_id_B)
                    # # 删除字典中对应key
                    # if obj_id_A in masks_A:
                    #     del masks_A[obj_id_A]
                    # if obj_id_B in masks_B:
                    #     del masks_B[obj_id_B]

    for obj_id, mask in masks_A.items():
        if obj_id not in keys_to_remove["A"]:
            merged_mask = np.logical_or(merged_mask, mask > 0).astype(np.uint8)
    
    for obj_id, mask in masks_B.items():
        if obj_id not in keys_to_remove["B"]:
            merged_mask = np.logical_or(merged_mask, mask > 0).astype(np.uint8)

    # sum_mask = mask1.astype(np.float32) + mask2.astype(np.float32)
    # return sum_mask / np.max(sum_mask)
    return merged_mask


def gen_frame(folder_paths, filename, output_dir="output_jpg", sort="asc"):
    if sort == "asc":
        for idx, folder_path in enumerate(folder_paths):
            # 构造输入和输出路径
            input_path = os.path.join(folder_path, filename)
            output_path = os.path.join("D:\QY\Datasets\sam2_test\WHU", output_dir)
            # 确保输出文件夹存在
            os.makedirs(output_path, exist_ok=True)
            output_path = os.path.join(output_path, str(idx+1) + '.jpg')
            
            # 打开PNG图片并转换为RGB模式（JPG不支持PNG的RGBA透明度）
            try:
                with Image.open(input_path) as img:
                    if img.mode in ('RGBA', 'LA'):
                        # 创建一个白色背景的RGB图像
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        background.paste(img, mask=img.split()[-1])  # 使用alpha通道作为mask
                        img = background
                    elif img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # 保存为JPG
                    img.save(output_path, 'JPEG', quality=90)
                    print(f"转换成功: {filename} -> {os.path.basename(output_path)}")
            except Exception as e:
                print(f"转换失败 {filename}: {str(e)}")
    elif sort == "desc":
        for idx, folder_path in enumerate(reversed(folder_paths)):
            # 构造输入和输出路径
            input_path = os.path.join(folder_path, filename)
            output_path = os.path.join("D:\QY\Datasets\sam2_test\WHU", output_dir)
            # 确保输出文件夹存在
            os.makedirs(output_path, exist_ok=True)
            output_path = os.path.join(output_path, str(idx+1) + '.jpg')
            
            # 打开PNG图片并转换为RGB模式（JPG不支持PNG的RGBA透明度）
            try:
                with Image.open(input_path) as img:
                    if img.mode in ('RGBA', 'LA'):
                        # 创建一个白色背景的RGB图像
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        background.paste(img, mask=img.split()[-1])  # 使用alpha通道作为mask
                        img = background
                    elif img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # 保存为JPG
                    img.save(output_path, 'JPEG', quality=90)
                    print(f"转换成功: {filename} -> {os.path.basename(output_path)}")
            except Exception as e:
                print(f"转换失败 {filename}: {str(e)}")


def clean_mask(mask, kernel_size=3, iterations=1):
    """
    通过形态学操作清理细碎区域
    :param mask: 输入二值化mask (0/1 或 0/255)
    :param kernel_size: 核大小（奇数，如3,5）
    :param iterations: 操作次数
    :return: 清理后的mask
    """
    # 保存原始形状
    orig_shape = mask.shape
     
    # 转换为OpenCV兼容格式
    if mask.dtype == bool:
        mask = mask.astype(np.uint8) * 255
    elif mask.dtype == np.float32 or mask.dtype == np.float64:
        mask = (mask * 255).astype(np.uint8)
    
    # 处理可能的3D输入（兼容 (H,W,1) 或 (H,W,3)）
    if len(mask.shape) == 3:
        mask = mask.squeeze()  # 移除单维度 (H,W,1) → (H,W)
        if len(mask.shape) == 3:  # 如果仍是3D（如RGB）
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # 开运算：先腐蚀后膨胀（去除小噪点）
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    
    # 闭运算：先膨胀后腐蚀（填充小孔洞）
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=iterations)

    # 恢复原始形状
    if len(orig_shape) == 3:
        cleaned = np.expand_dims(cleaned, 0)  # (1024,1024) -> (1,1024,1024)
    
    return cleaned


def filter_by_area(mask, min_area=100):
    """
    移除面积小于min_area的连通域
    :param mask: 输入二值化mask
    :param min_area: 最小保留面积（像素单位）
    :return: 过滤后的mask
    """
    # 保存原始形状
    orig_shape = mask.shape
     
    # 转换为OpenCV兼容格式
    if mask.dtype == bool:
        mask = mask.astype(np.uint8) * 255
    elif mask.dtype == np.float32 or mask.dtype == np.float64:
        mask = (mask * 255).astype(np.uint8)
    
    # 处理可能的3D输入（兼容 (H,W,1) 或 (H,W,3)）
    if len(mask.shape) == 3:
        mask = mask.squeeze()  # 移除单维度 (H,W,1) → (H,W)
        if len(mask.shape) == 3:  # 如果仍是3D（如RGB）
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(mask)
    
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(filtered_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    # 恢复原始形状
    if len(orig_shape) == 3:
        filtered_mask = np.expand_dims(filtered_mask, 0)  # (1024,1024) -> (1,1024,1024)
    
    return filtered_mask


def postprocess_mask(mask):
    # 形态学处理
    mask = clean_mask(mask)

    # 过滤小面积部分
    mask = filter_by_area(mask, min_area=50)

    return mask / np.max(mask)


from sam2.build_sam import build_sam2_video_predictor
from tools.get_mask_bboxes import get_mask_bboxes

if __name__ == "__main__":
    # 加载SAM2 video predictor
    sam2_checkpoint = "D:\QY\Checkpoints\sam2.1_hiera_large.pt"
    model_cfg = "D:\QY\sam2-cd-no-training\sam2\configs\sam2.1\sam2.1_hiera_l.yaml"

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    # 输入前后时相图片
    img_name = "tile_0_0.png"
    T1 = "D:\QY\Datasets\WHU-CD\\train\A"
    T2 = "D:\QY\Datasets\WHU-CD\\train\B"
    T1_label = "D:\QY\Datasets\WHU-CD\\before_label"
    T2_label = "D:\QY\Datasets\WHU-CD\\after_label"

    diff_mask_list = []

    for i, label in enumerate([T1_label, T2_label]):
        # 生成顺序jpg
        gen_frame([T1, T2], img_name, sort="asc" if i==0 else "desc")

        # 读取帧图片
        # `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
        video_dir = os.path.join("D:\QY\Datasets\sam2_test\WHU", "output_jpg")

        # scan all the JPEG frame names in this directory
        frame_names = [
            p for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        inference_state = predictor.init_state(video_path=video_dir)

        # track objects
        predictor.reset_state(inference_state)

        # 获取label中建筑物box，并逐个赋予id进行追踪
        bboxes = get_mask_bboxes(os.path.join(label, img_name))

        ann_list = []
        for idx, (x, y, w, h) in enumerate(bboxes):
            ann_list.append({
                "ann_frame_idx": 0,
                "ann_obj_id": idx+1,
                "box": np.array([x, y, x+w, y+h])
            })

        # 将ann_list导入predictor
        for item in ann_list:
            _, out_obj_ids, out_mask_logits = add_new_obj(**item, predictor=predictor) 

        # 获取追踪结果
        # run propagation throughout the video and collect the results in a dict
        video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        # render the segmentation results every few frames
        vis_frame_stride = 1
        plt.close("all")
        # merged_mask_list = []
        # for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
        #     # plt.figure(figsize=(6, 4))
        #     # plt.title(f"frame {out_frame_idx}")
        #     # plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))

        #     # mask分开显示
        #     # for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        #     #     show_mask(out_mask, plt.gca(), obj_id=out_obj_id)

        #     # mask合并显示
        #     segments_len = len(video_segments)
        #     # 非尾帧与尾帧比较，尾帧与首帧比较 todo
        #     merged_mask = merge_masks(
        #         video_segments[out_frame_idx], 
        #         compare_masks_dict=video_segments[segments_len-1] if out_frame_idx < (segments_len-1) else video_segments[0])
        #     merged_mask_list.append(merged_mask)
        #     # show_mask(merged_mask, plt.gca(), obj_id=1)

        #     # plt.show()

        # # 计算差分mask（变化区域）
        # diff_mask = compute_mask_diff(merged_mask_list[0], merged_mask_list[1])

        # diff_mask_list.append(diff_mask)

        # 
        # mask合并显示
        segments_len = len(video_segments)
        # 首尾帧比较
        diff_mask = merge_masks(
            video_segments[0], 
            compare_masks_dict=video_segments[segments_len-1])
        
        diff_mask_list.append(diff_mask)


    # diff_1 = diff_mask_list[0]
    # diff_2 = diff_mask_list[1]
    # diff_mask = sum_masks(diff_1, diff_2)
    # diff_mask = diff_mask

    # diff_1 = postprocess_mask(diff_mask_list[0])
    # diff_2 = postprocess_mask(diff_mask_list[1])
    # diff_mask = sum_masks(diff_1, diff_2)
    # diff_mask = postprocess_mask(diff_mask)

    diff_1 = sum_masks_dict(diff_mask_list[0])
    diff_2 = sum_masks_dict(diff_mask_list[1])
    diff_mask = sum_masks_dict(*diff_mask_list)
    diff_mask = diff_mask

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # 显示第一帧 + 原始masks
    frame1 = Image.open(os.path.join(video_dir, frame_names[0]))
    # ax1.imshow(frame1)
    ax1.set_title(f"Frame {1} (Masks)")
    show_mask(diff_1, ax1, obj_id=1)
    # show_mask(merged_mask_list[0], ax1, obj_id=1)
    # for out_obj_id, out_mask in video_segments[out_frame_idx].items():
    #     show_mask(out_mask, ax1, obj_id=out_obj_id)

    # 显示第二帧 + 原始masks
    frame2 = Image.open(os.path.join(video_dir, frame_names[1]))
    # ax2.imshow(frame2)
    ax2.set_title(f"Frame {1 + 1} (Masks)")
    show_mask(diff_2, ax2, obj_id=2)
    # show_mask(merged_mask_list[1], ax2, obj_id=2)
    # for out_obj_id, out_mask in video_segments[out_frame_idx + 1].items():
    #     show_mask(out_mask, ax2, obj_id=out_obj_id)

    # 显示差分结果（红色高亮变化区域）
    # ax3.imshow(frame2)
    show_mask(diff_mask, ax3, obj_id=3, random_color=True)
    # ax3.imshow(diff_mask, cmap='Reds', alpha=0.5)  # 红色半透明叠加
    ax3.set_title("Changes (Red Regions)")

    plt.tight_layout()
    plt.show()