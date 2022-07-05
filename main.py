import cv2
import numpy as np
from xycut import bbox2points, recursive_xy_cut, vis_polygons_with_index

if __name__ == "__main__":
    boxes = [
        [37, 37, 531, 37 + 68],
        [37, 140, 37 + 92, 140 + 246],
        [152, 161, 152 + 271, 161 + 74],
        [142, 263, 142 + 125, 263 + 123],
        [320, 244, 320 + 65, 244 + 64],
        [320, 332, 320 + 65, 332 + 64],
        [439, 140, 439 + 92, 140 + 246],
    ]

    random_boxes = np.array(boxes)
    np.random.shuffle(random_boxes)
    res = []
    recursive_xy_cut(np.asarray(random_boxes).astype(int), np.arange(len(boxes)), res)
    assert len(res) == len(boxes)
    sorted_boxes = random_boxes[np.array(res)].tolist()
    print(sorted_boxes)

    image = cv2.imread("./xy_cut_example.png")
    result = vis_polygons_with_index(image, [bbox2points(it) for it in sorted_boxes])
    cv2.imwrite("./xy_cut_result.png", result)
