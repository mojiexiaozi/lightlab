from pathlib import Path
import onnxruntime as ort
from typing import Any, Union, List
import numpy as np

from lightlab.inference.inference import Engine


class OnnxEngine(Engine):
    def __init__(self, model_path: Union[Path, str]) -> None:

        providers: list = ort.get_available_providers()
        if "TensorrtExecutionProvider" in providers:
            providers.remove("TensorrtExecutionProvider")
        if "CUDAExecutionProvider" in providers:
            providers.remove("CPUExecutionProvider")
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_names = [input.name for input in self.session.get_inputs()]
        self.intput_shapes = [input.shape for input in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        self.output_shapes = [out.shape for out in self.session.get_outputs()]
        print(
            [(name, shape) for name, shape in zip(self.input_names, self.intput_shapes)]
        )
        print(
            [
                (name, shape)
                for name, shape in zip(self.output_names, self.output_shapes)
            ]
        )
        super().__init__(model_path)

    def run_session(self, inputs: List[np.ndarray]) -> np.ndarray:
        input_dict = {}
        for i, name in enumerate(self.input_names):
            input_dict[name] = inputs[i]
        outputs = self.session.run(None, input_dict)
        return outputs


if __name__ == "__main__":
    engine = OnnxEngine("assets/models/yolov8n-cls.onnx")
