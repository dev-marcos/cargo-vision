# =====================================================================
# Fix Python import path so "src/" becomes a valid root package
# =====================================================================
import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

print("DEBUG - SRC_DIR added to PYTHONPATH:", SRC_DIR)



# =====================================================================
# Imports
# =====================================================================
from typing import Dict, Optional, Tuple

import cv2
import torch
from ultralytics import YOLO

from ocr.ocr_reader import CargoNumberOCR
from utils.drawing import draw_tracked_box
from utils.logging_utils import TrackingLogger



# =====================================================================
# Video Processing Pipeline
# =====================================================================
class VideoProcessingPipeline:
    """
    End-to-end video processing pipeline:
        - Loads a trained YOLO model
        - Runs ByteTrack for multi-object tracking
        - Applies OCR to extract cargo numbers
        - Draws annotated frames
        - Outputs new video + CSV logs
    """

    def __init__(
        self,
        model_path: str,
        output_dir: str = "videos/output",
        log_dir: str = "videos/logs",
        tracker_config: str = "bytetrack.yaml",
        cargo_class_index: int = 0,
    ) -> None:

        self.model_path = model_path
        self.output_dir = output_dir
        self.log_dir = log_dir
        self.tracker_config = tracker_config
        self.cargo_class_index = cargo_class_index

        # Device selection
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        # Load YOLO model
        print("Loading YOLO model...")
        self.model = YOLO(model_path)
        self.model.to(device)

        # OCR helper
        self.ocr = CargoNumberOCR()

        # Ensure folders exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    # -----------------------------------------------------------------
    def process_folder(self, input_dir: str) -> None:
        """Process all videos inside a directory."""

        valid_ext = (".mp4", ".avi", ".mov", ".mkv")

        if not os.path.exists(input_dir):
            print(f"Input folder not found: {input_dir}")
            return

        videos = [
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.lower().endswith(valid_ext)
        ]

        if not videos:
            print(f"No video files found in folder: {input_dir}")
            return

        print(f"Found {len(videos)} video(s) in {input_dir}")

        for idx, video_path in enumerate(videos, start=1):
            filename = os.path.basename(video_path)
            name_no_ext = os.path.splitext(filename)[0]

            print(f"[{idx}/{len(videos)}] Processing: {filename}")
            self.process_video(video_path=video_path, output_name=name_no_ext)

    # -----------------------------------------------------------------
    def process_video(self, video_path: str, output_name: str) -> None:
        """Process a single video file frame-by-frame."""

        # Open video to get metadata
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        cap.release()

        # Output files
        output_path = os.path.join(self.output_dir, f"{output_name}_processed.mp4")
        log_path = os.path.join(self.log_dir, f"{output_name}_tracking.csv")

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        logger = TrackingLogger(log_path)

        # Stores: track_id -> cargo_number
        cargo_by_track: Dict[int, Optional[str]] = {}

        frame_index = 0

        # ByteTrack streaming
        results_generator = self.model.track(
            source=video_path,
            tracker=self.tracker_config,
            stream=True,
            verbose=False,
        )

        # ---------------------------------------------------------------
        # Process all frames
        # ---------------------------------------------------------------
        for result in results_generator:
            frame = result.orig_img
            boxes = result.boxes

            if boxes is None or boxes.xyxy is None:
                writer.write(frame)
                frame_index += 1
                continue

            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy() if boxes.cls is not None else None
            ids = boxes.id.cpu().numpy() if boxes.id is not None else None

            num_boxes = xyxy.shape[0]

            # -----------------------------------------------------------
            # Loop each detected/ tracked object
            # -----------------------------------------------------------
            for i in range(num_boxes):
                x1, y1, x2, y2 = map(int, xyxy[i])
                bbox = (x1, y1, x2, y2)

                track_id = int(ids[i]) if ids is not None else -1
                class_id = int(cls[i]) if cls is not None else -1

                # Only process cargo class
                if class_id != -1 and class_id != self.cargo_class_index:
                    continue

                # Check if we already know the cargo number
                cargo_number = cargo_by_track.get(track_id)

                # If unknown → try OCR
                if cargo_number is None:
                    detected_number = self.ocr.extract_number(frame, bbox)
                    if detected_number is not None:
                        cargo_by_track[track_id] = detected_number
                        cargo_number = detected_number

                # Draw box
                draw_tracked_box(
                    frame=frame,
                    bbox=bbox,
                    track_id=track_id,
                    cargo_number=cargo_number,
                )

                # Logging
                logger.log(
                    frame_index=frame_index,
                    track_id=track_id,
                    cargo_number=cargo_number,
                    bbox=bbox,
                )

            writer.write(frame)
            frame_index += 1

        writer.release()

        print(f"Video saved to: {output_path}")
        print(f"Tracking log saved to: {log_path}")



# =====================================================================
# Script entry point
# =====================================================================
def main() -> None:
    pipeline = VideoProcessingPipeline(
        model_path="models/best.pt",
        output_dir="videos/output",
        log_dir="videos/logs",
        tracker_config="bytetrack.yaml",
        cargo_class_index=0,
    )

    pipeline.process_folder("videos/input")


if __name__ == "__main__":
    main()
