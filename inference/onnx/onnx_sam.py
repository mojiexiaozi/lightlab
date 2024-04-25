import numpy as np

from inference.onnx.onnx_sam_encoder import OnnxSamEncoder
from inference.onnx.onnx_sam_decoder import OnnxSamDecoder
from inference.utils import apply_coords


class OnnxSam:
    def __init__(self, encode_model_path, decoder_model_path) -> None:
        self.target_size = 1024
        self.encoder = OnnxSamEncoder(encode_model_path, self.target_size)
        self.decoder = OnnxSamDecoder(decoder_model_path, self.target_size)

        self.features = None

    def register_image(self, img: np.ndarray):
        self.features = self.encoder(img)

    def postprocess(self, outputs, original_size):
        mask = outputs["masks"][0][0]
        max_size = max(original_size)
        mask = cv2.resize(mask, (max_size, max_size))
        mask = mask[: original_size[0], : original_size[1]]
        mask[mask > 0.0] = 255
        mask[mask < 0.0] = 0

        return mask.astype(np.uint8)

    def inference(self, prompt):
        points = []
        labels = []
        original_size = self.features["orig_im_size"][0]
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
        points = apply_coords(points, original_size, self.target_size)

        outputs = self.decoder(
            {
                "image_embeddings": self.features["image_embeddings"],
                "point_coords": points,
                "point_labels": labels,
                "orig_im_size": original_size,
            }
        )
        return self.postprocess(outputs, original_size)


if __name__ == "__main__":
    import cv2
    from inference.utils import mask_to_shapes

    onnx_sam = OnnxSam(
        "backend/assets/models/sam_mobile_encoder.onnx",
        "backend/assets/models/sam_mobile_decoder.onnx",
    )
    img = cv2.imread("backend/assets/bus.jpg")
    print(img.shape)
    onnx_sam.register_image(img)
    mask = onnx_sam.inference(
        [
            {"type": "point", "points": [[252, 49]], "label": 1},
        ]
    )
    # rotation, circle, rectangle, polygon
    shapes = mask_to_shapes(mask, "circle")
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
    cv2.imwrite("bus_shapes.jpg", img)
    # 保存mask
    cv2.imwrite("bus_mask.jpg", mask)
    print(shapes)
