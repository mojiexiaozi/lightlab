from pathlib import Path
from typing import Any, Union, Tuple, List
import numpy as np


class Engine:
    def __init__(self, model_path: Union[Path, str]) -> None:
        self._model_path = model_path

    def run_session(self, inputs: List[np.ndarray]) -> np.ndarray:
        pass


class Inference:
    def __init__(self, engine: Engine, input_size=640) -> None:
        self._engine = engine
        if isinstance(input_size, int):
            input_size = [input_size, input_size]
        self.input_size = input_size
        self.scales = []
        self.ori_shapes = []
        self.trans = None

    def preprocess(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        """前处理"""
        self.scales.clear()
        self.ori_shapes.clear()

        for i in range(len(inputs)):
            img = inputs[i]
            self.ori_shapes.append(img.shape[:2])
            out: dict = self.trans(dict(img=img))
            self.scales.append(out.get("scale", 1.0))
            inputs[i] = out["img"]
        return [np.ascontiguousarray(inputs, np.float32)]

    def postprocess(self, outputs: np.ndarray):
        raise NotImplementedError

    def infer(self, inputs: List[np.ndarray]):
        if isinstance(inputs, np.ndarray):
            inputs = [inputs]

        pre_inputs = self.preprocess(inputs)
        outputs = self._engine.run_session(pre_inputs)
        results = self.postprocess(outputs)
        return results

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.infer(*args, **kwds)


# class Inference:
#     def __init__(
#         self,
#         model_path: Union[Path, str],
#         input_size: Union[int, Tuple[int, int]],
#         warmup: bool = True,
#     ) -> None:
#         self.model_path = model_path
#         self.input_size = input_size
#         if isinstance(input_size, int):
#             self.input_size = (input_size, input_size)

#         if warmup:
#             self.warmup()

#     def warmup(self) -> None:
#         im = np.random.randint(0, 255, (self.input_size[0], self.input_size[1], 3))
#         print("start warmup!")
#         self.inference(im.astype(np.uint8))
#         print("warmup finish!")

#     def preprocessing(self, inputs) -> None:
#         """前处理"""
#         self.ori_shapes = []
#         if not isinstance(inputs, List):
#             inputs = [inputs]
#         for input in inputs:
#             self.ori_shapes.append(input.shape[:2])
#         return inputs

#     def postprocessing(self, outputs) -> None:
#         """后处理"""
#         return outputs

#     def run_inference(self, inputs) -> None:
#         """运行推理"""
#         return inputs

#     def inference(self, inputs):
#         """推理"""
#         inputs = self.preprocessing(inputs)
#         outputs = self.run_inference(inputs)
#         return self.postprocessing(outputs)

#     def __call__(self, *args: Any, **kwds: Any) -> Any:
#         return self.inference(*args, **kwds)


# if __name__ == "__main__":
#     infer = Inference("model_path", 256, batch_size=1, warmup=True)
