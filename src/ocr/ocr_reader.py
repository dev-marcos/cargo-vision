from typing import Optional, Tuple

import cv2
import easyocr
import numpy as np
import re


class CargoNumberOCR:
    """Simple OCR helper to extract a two-digit cargo number from a trailer bounding box."""

    def __init__(self) -> None:
        # English is enough for numeric recognition
        self.reader = easyocr.Reader(["en"])

    def _preprocess_crop(self, roi: np.ndarray) -> np.ndarray:
        """Basic preprocessing to improve OCR robustness."""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return binary

    def extract_number(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[str]:
        """Extracts a two-digit cargo number from a trailer bounding box.

        Args:
            frame: Original image (BGR).
            bbox: (x1, y1, x2, y2) from YOLO/ByteTrack.

        Returns:
            Two-digit string like '31', or None if nothing was found.
        """
        x1, y1, x2, y2 = bbox

        # Basic crop: use the entire cargo bounding box (simpler and more robust for now)
        h_frame, w_frame = frame.shape[:2]
        x1 = max(0, min(x1, w_frame))
        x2 = max(0, min(x2, w_frame))
        y1 = max(0, min(y1, h_frame))
        y2 = max(0, min(y2, h_frame))

        if x2 <= x1 or y2 <= y1:
            return None

        roi = frame[y1:y2, x1:x2]
        processed = self._preprocess_crop(roi)

        # detail=0 -> only text strings
        results = self.reader.readtext(processed, detail=0)

        for text in results:
            # Look for exactly two digits
            match = re.findall(r"\b\d{2}\b", text)
            if match:
                return match[0]

        return None
