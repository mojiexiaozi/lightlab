import numpy as np
import cv2
from typing import List

from lightlab.inference.inference import Inference, Engine
from lightlab.inference.utils import (
    Transform,
    ToCHWImage,
    ResizeByShortEdge,
    CenterCrop,
    Normalize,
)


class YoloClsInfer(Inference):
    def __init__(self, engine: Engine, input_size=640, trans=None) -> None:
        super().__init__(engine, input_size)
        if trans is None:
            trans = Transform(
                [
                    ResizeByShortEdge(self.input_size),
                    CenterCrop(self.input_size),
                    Normalize(std=(1.0, 1.0, 1.0), mean=(0.0, 0.0, 0.0)),
                    ToCHWImage(),
                ]
            )
        self.trans = trans

    def postprocess(self, outputs) -> None:
        return np.argmax(outputs[0], axis=1).tolist()


if __name__ == "__main__":
    from lightlab.inference.onnx_engine import OnnxEngine

    engine = OnnxEngine("assets/models/yolov8n-cls.onnx")
    infer = YoloClsInfer(engine, 640)
    import cv2

    im = cv2.imread("assets/bus.jpg")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    print(infer([im]))
    # 测速不同batch推理速度
    # import time

    # 4060
    # total time: 0.15358281135559082, avg time: 0.007679140567779541
    # total time: 0.29361510276794434, avg time: 0.007340377569198609
    # total time: 0.6245980262756348, avg time: 0.007807475328445434
    # total time: 1.317793607711792, avg time: 0.0082362100481987
    # total time: 2.556508779525757, avg time: 0.00798908993601799
    # total time: 5.7365562915802, avg time: 0.008963369205594063
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
    #     print(res)
