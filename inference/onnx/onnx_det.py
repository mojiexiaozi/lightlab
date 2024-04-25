from pathlib import Path
from typing import Tuple
import numpy as np
import cv2

from inference.onnx.onnx_inference import OnnxInference
from inference.utils import xywh2xyxy


class OnnxDetInfer(OnnxInference):
    def __init__(
        self,
        model_path: Path | str,
        input_size: int | Tuple[int],
        warmup: bool = True,
        conf_thres: float = 0.5,
        iou_thres: float = 0.5,
    ) -> None:
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        super().__init__(model_path, input_size, warmup)

    def preprocessing(self, inputs) -> None:
        inputs = super().preprocessing(inputs)
        h, w = self.input_size
        for i in range(len(inputs)):
            inputs[i] = cv2.resize(inputs[i], (w, h))
        inputs = np.array(inputs).astype(np.float32) / 255.0
        # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        inputs = inputs[..., ::-1].transpose((0, 3, 1, 2))
        return {self.input_names[0]: np.ascontiguousarray(inputs)}

    def postprocessing(self, outputs) -> None:
        results = []
        outputs = np.transpose(outputs, (0, 2, 1))
        for batch, output in enumerate(outputs):
            x_factory = self.ori_shapes[batch][1] / self.input_size[1]
            y_factory = self.ori_shapes[batch][0] / self.input_size[0]

            classes_scores = output[:, 4:]
            max_scores = np.amax(classes_scores, axis=1)
            mask = max_scores > self.conf_thres

            scores = max_scores[mask]
            class_ids = np.argmax(classes_scores[mask], axis=1)
            boxes = xywh2xyxy(output[mask, :4], (x_factory, y_factory)).astype(np.int32)

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
    infer = OnnxDetInfer("assets/models/yolov8n.onnx", (640, 480))
    import cv2

    im = cv2.imread("assets/bus.jpg")
    results = infer(im)[0]
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
    import time

    for batch_size in [1, 2, 4, 8, 16, 32]:
        # warmup 3 times
        for _ in range(3):
            infer([im] * batch_size)
        start = time.time()
        for _ in range(20):
            res = infer([im] * batch_size)
        print(
            f"total time: {time.time() - start}, avg time: {(time.time() - start) / batch_size/20}"
        )
