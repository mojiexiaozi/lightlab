from typing import List
import numpy as np
import cv2

from lightlab.inference.onnx_engine import OnnxEngine
from lightlab.inference.inference import Inference, Engine
from lightlab.inference.utils import (
    xywh2xyxy,
    LetterBox,
    Transform,
    ToCHWImage,
    Normalize,
)


class YoloDetInfer(Inference):
    conf_thres: float = 0.5
    iou_thres: float = 0.5

    def __init__(self, engine: Engine, input_size=640, trans=None) -> None:
        super().__init__(engine, input_size)
        if trans is None:
            trans = Transform(
                [
                    LetterBox(self.input_size),
                    Normalize(std=(1.0, 1.0, 1.0), mean=(0.0, 0.0, 0.0)),
                    ToCHWImage(),
                ]
            )
        self.trans = trans

    def postprocess(self, outputs) -> None:
        results = []
        outputs = np.transpose(outputs[0], (0, 2, 1))
        for batch, output in enumerate(outputs):
            scale = 1 / self.scales[batch]

            classes_scores = output[:, 4:]
            max_scores = np.amax(classes_scores, axis=1)
            mask = max_scores > self.conf_thres

            scores = max_scores[mask]
            class_ids = np.argmax(classes_scores[mask], axis=1)
            boxes = xywh2xyxy(output[mask, :4], scale).astype(np.int32)

            indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_thres, self.iou_thres)
            results.append(
                [
                    {
                        "box": boxes[i].tolist(),
                        "score": scores[i],
                        "class_id": class_ids[i],
                    }
                    for i in indices
                ]
            )

        return results


if __name__ == "__main__":
    engine = OnnxEngine("assets/models/yolov8n.onnx")
    infer = YoloDetInfer(engine, (640, 640))
    import cv2

    im = cv2.imread("assets/bus.jpg")
    results = infer(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))[0]
    # 绘制结果
    for res in results:
        print(res)
        x1, y1, x2, y2 = res["box"]
        cv2.rectangle(
            im,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2,
        )
    cv2.imwrite("bus_det.jpg", im)
    # 测速不同batch推理速度
    # import time

    # for batch_size in [1, 2, 4, 8, 16, 32]:
    #     # warmup 3 times
    #     for _ in range(3):
    #         infer([im] * batch_size)
    #     start = time.time()
    #     for _ in range(20):
    #         res = infer([im] * batch_size)
    #     print(
    #         f"total time: {time.time() - start}, avg time: {(time.time() - start) / batch_size/20}"
    #     )
