from typing import Optional, Tuple, List

import cv2
import easyocr
import numpy as np
import re


class CargoNumberOCR:
    """Performs OCR to extract cargo numbers from trailer regions.

    This class assumes that the cargo number is located on the front canvas
    of the trailer and that a bounding box of the trailer has been provided.
    """

    def __init__(self, languages: Optional[List[str]] = None) -> None:
        if languages is None:
            languages = ["en"]
        self.reader = easyocr.Reader(languages)

    def _preprocess_crop(self, roi: np.ndarray) -> np.ndarray:
        """Applies preprocessing to improve OCR performance."""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return binary

    def extract_number(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[str]:
        """Extracts a two-digit cargo number from a trailer bounding box.

        Args:
            frame: Original image frame (BGR).
            bbox: Bounding box (x1, y1, x2, y2) of the trailer.

        Returns:
            Detected two-digit number as a string, or None if not found.
        """
        x1, y1, x2, y2 = bbox
        h = y2 - y1
        w = x2 - x1

        # Heuristic: focus on the upper front region of the trailer where the number is printed.
        y_start = y1
        y_end = y1 + int(h * 0.35)
        x_start = x1 + int(w * 0.20)
        x_end = x1 + int(w * 0.80)

        # Ensure boundaries are within frame
        h_frame, w_frame = frame.shape[:2]
        y_start = max(0, min(y_start, h_frame))
        y_end = max(0, min(y_end, h_frame))
        x_start = max(0, min(x_start, w_frame))
        x_end = max(0, min(x_end, w_frame))

        if y_end <= y_start or x_end <= x_start:
            return None

        roi = frame[y_start:y_end, x_start:x_end]
        processed = self._preprocess_crop(roi)

        results = self.reader.readtext(processed, detail=0)

        for text in results:
            matches = re.findall(r"\b\d{2}\b", text)
            if matches:
                return matches[0]

        return None
