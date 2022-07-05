import json
import cv2
import numpy as np
from xycut import bbox2points, recursive_xy_cut, vis_polygons_with_index


def load_data(p):
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    boxes = []
    for it in data["document"]:
        boxes.append(it["box"])
    return np.array(boxes)


if __name__ == "__main__":
    boxes = load_data("./zh_train_51.json")

    image = cv2.imread("./zh_train_51.jpg")
    result = vis_polygons_with_index(image, [bbox2points(it) for it in boxes])
    cv2.imwrite("./zh_train_51_original.jpg", result)

    res = []
    recursive_xy_cut(np.asarray(boxes).astype(int), np.arange(len(boxes)), res)
    assert len(res) == len(boxes)
    sorted_boxes = boxes[np.array(res)].tolist()

    result = vis_polygons_with_index(image, [bbox2points(it) for it in sorted_boxes])
    cv2.imwrite("./zh_train_51_result.jpg", result)
