from pathlib import Path
from typing import Tuple
import onnxruntime as ort


from inference.inference import Inference


class OnnxInference(Inference):
    def __init__(
        self,
        model_path: Path | str,
        input_size: int | Tuple[int, int],
        warmup: bool = True,
    ) -> None:

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
        super().__init__(model_path, input_size, warmup)

    def run_inference(self, inputs):
        outputs = self.session.run(None, inputs)[0]
        return outputs
