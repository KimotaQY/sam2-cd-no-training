import os

# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
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


np.random.seed(3)


def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones(
        (
            sorted_anns[0]["segmentation"].shape[0],
            sorted_anns[0]["segmentation"].shape[1],
            4,
        )
    )
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.3]])
        img[m] = color_mask
        if borders:
            import cv2

            contours, _ = cv2.findContours(
                m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            # Try to smooth contours
            contours = [
                cv2.approxPolyDP(contour, epsilon=0.01, closed=True)
                for contour in contours
            ]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

    # cv2.imwrite("B_test_79.png", img * 255)
    ax.imshow(img)


def overlay_mask_on_image_and_save(image, anns, output_path, borders=True):
    if len(anns) == 0:
        return

    sorted_anns = sorted(anns, key=lambda x: x["area"], reverse=True)
    img = np.zeros_like(image)

    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate(
            [np.random.random(3), [0.5]]
        )  # Random color with alpha=0.5
        img[m] = color_mask[:3] * 255  # Apply RGB values without alpha channel

        if borders:
            import cv2

            contours, _ = cv2.findContours(
                m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            # Try to smooth contours
            contours = [
                cv2.approxPolyDP(contour, epsilon=0.01, closed=True)
                for contour in contours
            ]
            cv2.drawContours(img, contours, -1, (0, 0, 255), thickness=1)

    # Overlay the mask on the original image
    overlayed_img = cv2.addWeighted(image, 1, img, 0.5, 0)

    # Save the result
    cv2.imwrite(output_path, overlayed_img)


image = Image.open("E:\CD_datasets\LEVIR-CD\\test\A\\test_79.png")
image = np.array(image.convert("RGB"))

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

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
model_type = "l"
checkpoint = model_obj[model_type]["checkpoint"]
config = model_obj[model_type]["config"]
sam2_checkpoint = os.path.join("E:\CD_Checkpoints", checkpoint)
model_cfg = os.path.join(
    "E:\CD_projects\sam2-cd-no-training\sam2\configs\sam2.1", config
)

sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2,
    points_per_side=64,
    points_per_batch=128,
    pred_iou_thresh=0.7,
    stability_score_thresh=0.92,
    stability_score_offset=0.7,
    crop_n_layers=1,
    box_nms_thresh=0.7,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=25.0,
    use_m2m=True,
)

masks = mask_generator.generate(image)

overlay_mask_on_image_and_save(image, masks, "A_test_79_mask.png")

# plt.figure(figsize=(20, 20))
# plt.imshow(image)
# show_anns(masks)
# plt.axis("off")
# plt.show()
