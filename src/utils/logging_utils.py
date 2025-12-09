import csv
import os
from typing import Optional, Tuple


class TrackingLogger:
    """CSV logger for tracking information.

    Each row contains:
        frame_index, track_id, cargo_number, x1, y1, x2, y2
    """

    def __init__(self, csv_path: str) -> None:
        self.csv_path = csv_path
        self._ensure_header()

    def _ensure_header(self) -> None:
        if not os.path.exists(self.csv_path):
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["frame_index", "track_id", "cargo_number", "x1", "y1", "x2", "y2"]
                )

    def log(
        self,
        frame_index: int,
        track_id: int,
        cargo_number: Optional[str],
        bbox: Tuple[int, int, int, int],
    ) -> None:
        x1, y1, x2, y2 = bbox
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [frame_index, track_id, cargo_number if cargo_number is not None else "", x1, y1, x2, y2]
            )
