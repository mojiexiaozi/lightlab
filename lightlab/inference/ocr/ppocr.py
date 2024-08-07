import numpy as np
import cv2
from typing import List

from lightlab.inference.inference import Engine
from lightlab.inference.seg.seg_infer import SegInfer
from lightlab.inference.utils import ResizeLimitMax, ToCHWImage, Transform, Normalize


def get_rotate_crop_image(img, points):
    # points = np.array(points, dtype=np.float32)
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])
        )
    )
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])
        )
    )
    pts_std = np.float32(
        [
            [0, 0],
            [img_crop_width, 0],
            [img_crop_width, img_crop_height],
            [0, img_crop_height],
        ]
    )
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M,
        (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img


class PPOcr:
    def __init__(self, det_engine: Engine, det_size=960) -> None:
        det_trans = Transform(
            [
                ResizeLimitMax(det_size),
                Normalize(),
                ToCHWImage(),
            ]
        )
        self.det_infer = SegInfer(det_engine, det_size, det_trans)

    def __call__(self, imgs):
        det_res = self.det_infer(imgs)
        i = 0
        img = cv2.imread("assets/text.jpg")
        for im, det_res in zip(imgs, det_res):
            for det in det_res:
                rotated_rect = det["rotated_rect"]
                w, h = rotated_rect[1]
                if w < h:
                    w = int(2.5 * w)
                    h += 20
                else:
                    h = int(2.5 * h)
                    w += 20
                print(w, h)

                new_rect = cv2.RotatedRect(rotated_rect[0], (w, h), rotated_rect[2])
                cv2.drawContours(
                    img, [cv2.boxPoints(new_rect).astype(np.int32)], -1, (0, 255, 0), 2
                )
                # x, y, w, h = cv2.boundingRect(min_box)
                # crop_im = get_rotate_crop_image(img, min_box)
                # cv2.imwrite(f"crop{i}.png", crop_im)
                i += 1
        cv2.imwrite("text.png", img)


if __name__ == "__main__":
    from lightlab.inference.onnx_engine import OnnxEngine

    engine = OnnxEngine("assets/models/ch_PP-OCRv4_det_infer.onnx")
    ocr = PPOcr(engine)
    import cv2

    im = cv2.imread("assets/text.jpg")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    ocr([im])
