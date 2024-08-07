import numpy as np
import cv2

from lightlab.inference.onnx_engine import OnnxEngine
from lightlab.inference.inference import Inference, Engine
from lightlab.inference.utils import (
    SmallestResize,
    Transform,
    ToCHWImage,
    Normalize,
)


class SegInfer(Inference):
    conf_thres: float = 0.5
    iou_thres: float = 0.5

    def __init__(self, engine: Engine, input_size=640, trans=None) -> None:
        super().__init__(engine, input_size)
        if trans is None:
            trans = Transform(
                [
                    SmallestResize(self.input_size),
                    Normalize(),
                    ToCHWImage(),
                ]
            )
        self.trans = trans

    def postprocess(self, outputs) -> None:
        results = []
        for masks, scale in zip(outputs[0], self.scales):
            shapes = []

            for mask in masks.astype(np.uint8):
                print(mask.shape)
                #  膨胀操作
                # mask = cv2.dilate(mask, np.array([[1, 1], [1, 1]]))
                contours, _ = cv2.findContours(
                    mask * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
                )
                cls_id = mask.max()
                for contour in contours:
                    contour = contour.squeeze(axis=1).astype(np.float32)
                    contour[:, 0] *= scale[1]
                    contour[:, 1] *= scale[0]

                    area = cv2.contourArea(contour)
                    if area < 5:
                        continue
                    rotated_rect = cv2.minAreaRect(contour)
                    # min_box[:, 0] *= scale[1]
                    # min_box[:, 1] *= scale[0]
                    box = cv2.boundingRect(contour)
                    # print(box, min_box.astype(np.int32).shape)

                    shapes.append(
                        dict(
                            cls_id=cls_id,
                            area=area,
                            rotated_rect=rotated_rect,
                            box=box,
                        ),
                    )
            results.append(shapes)
        cv2.imwrite("text_mask.png", mask * 255)
        return results


if __name__ == "__main__":
    engine = OnnxEngine("assets/models/ch_PP-OCRv4_det_infer.onnx")
    infer = SegInfer(engine, 736)
    import cv2

    im = cv2.imread("assets/text.jpg")
    print(im.shape)
    results = infer(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    # results = infer(im)
    # print(results, len(results), len(results[0]))
    # im = cv2.resize(im, (736, 672))
    for res in results[0]:
        min_box = res["min_box"]
        x, y, w, h = res["box"]
        print(min_box)
        cv2.drawContours(im, [np.array(min_box)], 0, (0, 0, 255), 2)
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imwrite("text.png", im)
