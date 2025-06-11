# Initialize the client with your API token.
import os
import json
import tempfile
import supervision as sv
import cv2
from dds_cloudapi_sdk import Config
from dds_cloudapi_sdk import Client
from dds_cloudapi_sdk.tasks.v2_task import V2Task
import numpy as np


# TEXT_PROMPT = "house.building.roof"
TEXT_PROMPT = "house"
GROUNDING_MODEL = "GroundingDino-1.5-Pro"
WITH_SLICE_INFERENCE = True
SLICE_WH = (480, 480)
OVERLAP_RATIO = (0.2, 0.2)
IOU_THRESHOLD = 0.8

token = "a95f70bbe8f8d2b3834ab5b3090f0e8b"
config = Config(token)
client = Client(config)

classes = [x.strip().lower() for x in TEXT_PROMPT.split(".") if x]
class_name_to_id = {name: id for id, name in enumerate(classes)}
class_id_to_name = {id: name for name, id in class_name_to_id.items()}


def get_one_result(img_path, output_path):
    # Upload local image to the server and get the URL.
    # infer_image_url = "https://dds-frontend.oss-accelerate.aliyuncs.com/static_files/playground/grounding_DINO-1.6/02.jpg"
    infer_image_url = client.upload_file(
        img_path
    )  # you can also upload local file for processing

    # Create a task with proper parameters.

    task = V2Task(
        api_path="/v2/task/grounding_dino/detection",
        api_body={
            "model": GROUNDING_MODEL,
            "image": infer_image_url,
            "prompt": {"type": "text", "text": TEXT_PROMPT},
            "targets": ["bbox"],
            "bbox_threshold": 0.25,
            "iou_threshold": 0.8,
        },
    )
    # task.set_request_timeout(10)  # set the request timeout in seconds，default is 5 seconds

    # Run the task.
    client.run_task(task)

    # Get the result.
    # print(task.result)

    # Save the result to json
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(task.result, f, indent=4)


def get_one_by_SAHI(img_path):
    def callback(image_slice: np.ndarray) -> sv.Detections:
        print("Inference on image slice")
        # save the img as temp img file for GD-1.5 API usage
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmpfile:
            temp_filename = tmpfile.name
        cv2.imwrite(temp_filename, image_slice)
        image_url = client.upload_file(temp_filename)
        task = V2Task(
            api_path="/v2/task/grounding_dino/detection",
            api_body={
                "model": GROUNDING_MODEL,
                "image": image_url,
                "prompt": {"type": "text", "text": TEXT_PROMPT},
                "targets": ["bbox"],
                "bbox_threshold": 0.25,
                "iou_threshold": 0.8,
            },
        )
        client.run_task(task)
        result = task.result
        # delete the tempfile
        os.remove(temp_filename)

        input_boxes = []
        confidences = []
        class_ids = []
        objects = result["objects"]
        for idx, obj in enumerate(objects):
            input_boxes.append(obj["bbox"])
            confidences.append(obj["score"])
            cls_name = obj["category"].lower().strip()
            class_ids.append(class_name_to_id[cls_name])
        # ensure input_boxes with shape (_, 4)
        input_boxes = np.array(input_boxes).reshape(-1, 4)
        class_ids = np.array(class_ids)
        confidences = np.array(confidences)
        return sv.Detections(
            xyxy=input_boxes, confidence=confidences, class_id=class_ids
        )

    slicer = sv.InferenceSlicer(
        callback=callback,
        slice_wh=SLICE_WH,
        overlap_ratio_wh=OVERLAP_RATIO,
        iou_threshold=IOU_THRESHOLD,
        overlap_filter=sv.OverlapFilter.NON_MAX_SUPPRESSION,
    )
    detections = slicer(cv2.imread(img_path))
    class_names = [class_id_to_name[id] for id in detections.class_id]
    confidences = detections.confidence.tolist()
    class_ids = detections.class_id.tolist()
    input_boxes = detections.xyxy.tolist()

    # result = {
    #     "class_names": class_names,
    #     "confidences": confidences,
    #     "class_ids": class_ids,
    #     "input_boxes": input_boxes,
    # }

    objects = []

    for idx, box in enumerate(input_boxes):
        objects.append(
            {
                "bbox": box,
                "category": class_names[idx],
                "score": confidences[idx],
            }
        )

    result = {"objects": objects}

    # Save the result to json
    with open("test.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    img_dir = r"E:/CD_datasets/LEVIR-CD/test/B"
    # output_dir = r"E:/CD_datasets/LEVIR-CD/test/before_label"

    # os.makedirs(output_dir, exist_ok=True)

    # # get all filenames
    # img_names = [p for p in os.listdir(img_dir) if os.path.splitext(p)[-1] in [".png"]]

    # for idx, img_name in enumerate(img_names):
    #     img_path = os.path.join(img_dir, img_name)
    #     output_path = os.path.join(output_dir, os.path.splitext(img_name)[0] + ".json")

    #     if WITH_SLICE_INFERENCE:
    #         get_one_by_SAHI(img_path)
    #     else:
    #         get_one_result(img_path=img_path, output_path=output_path)

    #     print(f"{(idx+1)}/{len(img_names)} ===> {img_name} 已完成")

    img_path = os.path.join(img_dir, "test_74.png")
    get_one_by_SAHI(img_path)
