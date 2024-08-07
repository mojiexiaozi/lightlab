import numpy as np
from typing import List, Dict

from lightlab.inference.utils import (
    apply_coords,
    LetterBox,
    Transform,
    Normalize,
    ToCHWImage,
)
from lightlab.inference.inference import Inference, Engine


class SamEncoder(Inference):
    def __init__(self, engine: Engine, input_size=640, trans=None) -> None:
        super().__init__(engine, input_size)
        if trans is None:
            trans = Transform(
                [
                    LetterBox(self.input_size, pad_val=0),
                    Normalize(),
                    ToCHWImage(),
                ]
            )
        self.trans = trans

    def postprocess(self, outputs):
        return outputs[0]


class SamDecoder(Inference):
    def postprocess(self, outputs):
        return outputs[0]

    def preprocess(self, inputs) -> None:
        image_embeddings: np.ndarray = inputs[0]
        point_coords = inputs[1]
        point_labels = inputs[2]
        mask_input = inputs[3]
        ori_im_size = inputs[4]

        if len(image_embeddings.shape) == 3:
            image_embeddings = image_embeddings[None]

        batch_size = image_embeddings.shape[0]
        if not isinstance(mask_input, list):
            mask_input = [mask_input]

        has_mask_input = []
        if len(mask_input) < batch_size:
            mask_input.extend([None] * (batch_size - len(mask_input)))

        for i in range(len(mask_input)):
            if mask_input[i] is None:
                mask_input[i] = np.zeros((1, 256, 256))
                has_mask_input.append(0)
            else:
                has_mask_input.append(1)

        return [
            image_embeddings,
            np.array(point_coords, np.float32),
            np.array(point_labels, np.float32),
            np.array(mask_input, np.float32),
            np.array(has_mask_input, np.float32),
            np.array(ori_im_size, np.float32),
        ]


class Sam:
    def __init__(self, encode_engine, decoder_engine, target_size=1024) -> None:
        self.encoder = SamEncoder(encode_engine, target_size)
        self.decoder = SamDecoder(decoder_engine, target_size)

        self.features = None
        self.im_size = None
        self.target_size = target_size

    def register_image(self, img: np.ndarray):
        self.im_size = img.shape[:2]
        self.features = self.encoder(img)[0]

    def postprocess(self, outputs):
        mask = outputs[0][0]
        mask[mask > 0.0] = 255
        mask[mask < 0.0] = 0
        return mask.astype(np.uint8)

    def inference(self, prompt: List[Dict]):
        points = []
        labels = []
        for mark in prompt:
            if mark["type"] == "point":
                points.append(mark["points"][0])
                labels.append(mark["label"])
            elif mark["type"] == "rectangle":
                points.append(mark["points"][0])  # top left
                points.append(mark["points"][1])  # bottom right
                labels.append(2)
                labels.append(3)
        points = np.array(points, dtype=np.float32)[None]
        labels = np.array(labels, dtype=np.float32)[None]
        points = apply_coords(points, self.im_size, self.target_size)

        outputs = self.decoder(
            [
                self.features,  # image_embeddings
                points,  # point_coords
                labels,  # point_labels
                None,  # mask input
                self.im_size,  # orig_im_size
            ]
        )
        return self.postprocess(outputs)


if __name__ == "__main__":
    import cv2
    from lightlab.inference.utils import mask_to_shapes
    from lightlab.inference.onnx_engine import OnnxEngine

    onnx_sam = Sam(
        OnnxEngine("assets/models/sam_mobile_encoder.onnx"),
        OnnxEngine("assets/models/sam_mobile_decoder.onnx"),
    )
    img = cv2.imread("assets/test.png")
    print(img.shape)
    onnx_sam.register_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    mask = onnx_sam.inference(
        [
            {"type": "point", "points": [[382, 264]], "label": 1},
            # {"type": "point", "points": [[313, 282]], "label": 0},
        ]
    )
    cv2.circle(img, (382, 264), 5, (0, 0, 255), 2)
    cv2.circle(img, (282, 313), 5, (0, 0, 255), 2)
    # rotation, circle, rectangle, polygon
    shapes = mask_to_shapes(mask, "polygon")
    print(shapes)
    # 将shapes绘制到图片
    for shape in shapes:
        if shape["shape_type"] == "circle":
            center, radius = shape["points"]
            cv2.circle(img, tuple(center), radius, (255, 255, 255), 2)
        else:
            points = np.array(shape["points"], dtype=np.int32)
            cv2.polylines(
                img, [points], isClosed=True, color=(255, 255, 255), thickness=2
            )
    # 保存图片
    cv2.imwrite("shapes.jpg", img)
    # 保存mask
    cv2.imwrite("mask.jpg", mask)
    print(shapes)
