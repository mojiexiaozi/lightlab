import numpy as np

from inference.onnx.onnx_inference import OnnxInference
from inference.utils import padding, smallest_resize


class OnnxSamEncoder(OnnxInference):
    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([58.395, 57.12, 57.375])

    def postprocessing(self, outputs):
        return {self.output_names[0]: outputs, "orig_im_size": self.ori_shapes}

    def preprocessing(self, inputs) -> None:
        inputs = super().preprocessing(inputs)
        for i in range(len(inputs)):
            inputs[i] = smallest_resize(inputs[i], self.input_size)[0]
            inputs[i] = padding(inputs[i], self.input_size)
        inputs = np.array(inputs).astype(np.float32)
        inputs = (inputs - self.mean) / self.std
        inputs = np.transpose(inputs, (0, 3, 1, 2)).astype(np.float32)
        return {self.input_names[0]: inputs}


if __name__ == "__main__":
    infer = OnnxSamEncoder(
        "backend/assets/models/sam_mobile_encoder.onnx", 1024, warmup=True
    )
