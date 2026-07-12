"""JSON-lines PaddleOCR worker executed inside its isolated virtualenv."""

import base64
from contextlib import redirect_stdout
import json
import os
import sys

import cv2
import numpy as np
from paddleocr import PaddleOCR


def main() -> None:
    with redirect_stdout(sys.stderr):
        pipeline = PaddleOCR(
            lang=os.getenv("PADDLE_OCR_LANGUAGE", "en"),
            text_detection_model_name=os.getenv(
                "PADDLE_OCR_DETECTION_MODEL", "PP-OCRv5_mobile_det"
            ),
            text_recognition_model_name=os.getenv(
                "PADDLE_OCR_RECOGNITION_MODEL", "PP-OCRv5_mobile_rec"
            ),
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    for line in sys.stdin:
        try:
            request = json.loads(line)
            encoded = np.frombuffer(base64.b64decode(request["image"]), dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("invalid encoded image")
            results = []
            with redirect_stdout(sys.stderr):
                for result in pipeline.predict(image):
                    payload = getattr(result, "json", result)
                    if callable(payload):
                        payload = payload()
                    results.append(payload)
            response = {"results": results}
        except Exception as exc:
            response = {"error": str(exc)}
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
