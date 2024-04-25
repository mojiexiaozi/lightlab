import numpy as np
import cv2

from inference.onnx.onnx_inference import OnnxInference
from inference.utils import resize_by_shortest_edge, center_crop


class OnnxClsInfer(OnnxInference):
    def preprocessing(self, inputs) -> None:
        inputs = super().preprocessing(inputs)
        for i in range(len(inputs)):
            inputs[i] = resize_by_shortest_edge(inputs[i], self.input_size[0])
            inputs[i] = center_crop(inputs[i], self.input_size)
        inputs = np.array(inputs).astype(np.float32)
        inputs = inputs / 255.0
        # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        inputs = inputs[..., ::-1].transpose((0, 3, 1, 2))
        return {self.input_names[0]: np.ascontiguousarray(inputs)}

    def postprocessing(self, outputs) -> None:
        return np.argmax(outputs, axis=1).tolist()


if __name__ == "__main__":
    infer = OnnxClsInfer("assets/models/yolov8n-cls.onnx", 640)
    import cv2

    im = cv2.imread("assets/bus.jpg")
    # 测速不同batch推理速度
    import time

    # 4060
    # total time: 0.15358281135559082, avg time: 0.007679140567779541
    # total time: 0.29361510276794434, avg time: 0.007340377569198609
    # total time: 0.6245980262756348, avg time: 0.007807475328445434
    # total time: 1.317793607711792, avg time: 0.0082362100481987
    # total time: 2.556508779525757, avg time: 0.00798908993601799
    # total time: 5.7365562915802, avg time: 0.008963369205594063
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
        print(res)
