import numpy as np

from inference.onnx.onnx_inference import OnnxInference


class OnnxSamDecoder(OnnxInference):
    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([58.395, 57.12, 57.375])

    def warmup(self) -> None:
        x = {
            self.input_names[0]: np.random.random((1, 256, 64, 64)).astype(np.float32),
            self.input_names[1]: np.random.random((1, 1, 2)).astype(np.float32),
            self.input_names[2]: np.ones((1, 1), dtype=np.float32),
            self.input_names[3]: None,
            self.input_names[4]: None,
            self.input_names[5]: (1024, 1024),
        }
        print("start warmup!")
        self.inference(x)
        print("warmup finish!")

    def postprocessing(self, outputs):
        return {self.output_names[0]: outputs}

    def preprocessing(self, inputs: dict) -> None:
        image_embeddings = inputs.get("image_embeddings")
        point_coords = inputs.get("point_coords")
        point_labels = inputs.get("point_labels")
        mask_input = inputs.get("mask_input")
        orig_im_size = inputs.get("orig_im_size")

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

        return {
            self.input_names[0]: image_embeddings,
            self.input_names[1]: np.array(point_coords, np.float32),
            self.input_names[2]: np.array(point_labels, np.float32),
            self.input_names[3]: np.array(mask_input, np.float32),
            self.input_names[4]: np.array(has_mask_input, np.float32),
            self.input_names[5]: np.array(orig_im_size, np.float32),
        }


if __name__ == "__main__":
    import cv2
    from inference.onnx.onnx_sam_encoder import OnnxSamEncoder

    infer = OnnxSamEncoder(
        "backend/assets/models/sam_mobile_encoder.onnx", 1024, warmup=True
    )
    im = cv2.imread("backend/assets/bus.jpg")
    out = infer(im)
    decoder = OnnxSamDecoder("backend/assets/models/sam_mobile_decoder.onnx", 1024)
