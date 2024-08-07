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


class PPOcrRec(Inference):
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

    engine = OnnxEngine("assets/models/ch_PP-OCRv4_rec_infer.onnx")
    infer = PPOcrRec(engine, 640)
    import cv2

    im = cv2.imread("assets/text.jpg")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # print(infer([im]))
