import logging

import numpy as np
from pydantic import Field
from typing_extensions import Literal

from frigate.detectors.detection_api import DetectionApi
from frigate.detectors.detector_config import BaseDetectorConfig

try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ModuleNotFoundError:
    from tensorflow.lite.python.interpreter import Interpreter, load_delegate


logger = logging.getLogger(__name__)

DETECTOR_KEY = "edgetpu"


class EdgeTpuDetectorConfig(BaseDetectorConfig):
    type: Literal[DETECTOR_KEY]
    device: str = Field(default=None, title="Device Type")


class EdgeTpuTfl(DetectionApi):
    type_key = DETECTOR_KEY

    def __init__(self, detector_config: EdgeTpuDetectorConfig):
        device_config = {}
        if detector_config.device is not None:
            device_config = {"device": detector_config.device}

        edge_tpu_delegate = None

        try:
            device_type = (
                device_config["device"] if "device" in device_config else "auto"
            )
            logger.info(f"Attempting to load TPU as {device_type}")
            edge_tpu_delegate = load_delegate("libedgetpu.so.1.0", device_config)
            logger.info("TPU found")
            self.interpreter = Interpreter(
                model_path=detector_config.model.path,
                experimental_delegates=[edge_tpu_delegate],
            )
        except ValueError:
            logger.error(
                "No EdgeTPU was detected. If you do not have a Coral device yet, you must configure CPU detectors."
            )
            raise

        self.interpreter.allocate_tensors()

        self.tensor_input_details = self.interpreter.get_input_details()
        self.tensor_output_details = self.interpreter.get_output_details()
        self.model_type = detector_config.model.model_type

    def detect_raw(self, tensor_input):
        scale, zero_point = self.tensor_input_details[0]["quantization"]
        tensor_input = (
            (tensor_input - scale * zero_point * 255) * (1.0 / (scale * 255))
        ).astype(self.tensor_input_details[0]["dtype"])

        self.interpreter.set_tensor(self.tensor_input_details[0]["index"], tensor_input)
        self.interpreter.invoke()

        tensor_output = self.interpreter.get_tensor(
            self.tensor_output_details[0]["index"]
        )
        # Dequantize the output (if needed later)
        tensor_output = (tensor_output.astype(np.float32) - zero_point) * scale

        model_input_shape = self.tensor_input_details[0]["shape"]
        tensor_output[:, [0, 2]] *= model_input_shape[2]
        tensor_output[:, [1, 3]] *= model_input_shape[1]

        # Initialize detections array (20 detections with 6 parameters each)
        score_threshold = 0.5
        nms_threshold = 0.5
        box_count = 20
        model_box_count = tensor_output.shape[2]
        probs = tensor_output[0, 4:, :]
        all_ids = np.argmax(probs, axis=0)
        all_confidences = probs.T[np.arange(model_box_count), all_ids]
        all_boxes = tensor_output[0, 0:4, :].T
        mask = all_confidences > score_threshold
        class_ids = all_ids[mask]
        confidences = all_confidences[mask]
        cx, cy, w, h = all_boxes[mask].T
        if model_input_shape[3] == 3:
            scale_y, scale_x = 1 / model_input_shape[1], 1 / model_input_shape[2]
        else:
            scale_y, scale_x = 1 / model_input_shape[2], 1 / model_input_shape[3]

        detections = np.stack(
            (
                class_ids,
                confidences,
                scale_y * (cy - h / 2),
                scale_x * (cx - w / 2),
                scale_y * (cy + h / 2),
                scale_x * (cx + w / 2),
            ),
            axis=1,
        )
        if detections.shape[0] > box_count:
            # if too many detections, do nms filtering to suppress overlapping boxes
            boxes = np.stack((cx - w / 2, cy - h / 2, w, h), axis=1)
            indexes = cv2.dnn.NMSBoxes(
                boxes, confidences, score_threshold, nms_threshold
            )
            detections = detections[indexes]
            # if still too many, trim the rest by confidence
            if detections.shape[0] > box_count:
                detections = detections[
                    np.argpartition(detections[:, 1], -box_count)[-box_count:]
                ]
            detections = detections.copy()
        detections.resize((box_count, 6))

        for i in detections:
            if i[1] == 0:
                i[0] = -1

        return detections
