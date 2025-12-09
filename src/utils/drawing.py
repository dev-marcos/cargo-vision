from typing import Optional, Tuple
import cv2
import numpy as np


def draw_tracked_box(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    track_id: int,
    cargo_number: Optional[str] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> None:
    """Draws a bounding box and label for a tracked object on the frame."""
    x1, y1, x2, y2 = bbox

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"ID {track_id}"
    if cargo_number is not None:
        label += f" | Cargo {cargo_number}"

    cv2.putText(
        frame,
        label,
        (x1, max(y1 - 10, 0)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        lineType=cv2.LINE_AA,
    )
