# =====================================================================
# Fix Python import path so "src/" becomes a valid root package
# =====================================================================
import sys
import os

# Path to: D:\Visao\yolo\src\inference
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to: D:\Visao\yolo\src
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

# Add src/ to Python path
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

print("DEBUG - SRC_DIR added to PYTHONPATH:", SRC_DIR)





from typing import Dict, Optional, Tuple

import cv2
import torch
from ultralytics import YOLO

from ocr.ocr_reader import CargoNumberOCR
from utils.drawing import draw_tracked_box
from utils.logging_utils import TrackingLogger


class VideoProcessingPipeline:
    """End-to-end video processing pipeline using YOLO, ByteTrack and OCR.

    This pipeline:
        - loads a trained YOLO model,
        - runs ByteTrack-based multi-object tracking,
        - applies OCR for cargo number recognition on new track IDs,
        - draws annotations on each frame,
        - writes the result to a video file,
        - logs track and cargo information in CSV format.
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

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        print("Loading YOLO model...")
        self.model = YOLO(model_path)
        self.model.to(device)

        self.ocr = CargoNumberOCR()

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    def process_folder(self, input_dir: str) -> None:
        """Processes all video files in the specified input directory."""
        valid_ext = (".mp4", ".avi", ".mov", ".mkv")
        videos = [
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.lower().endswith(valid_ext)
        ]

        if not videos:
            print(f"No video files found in {input_dir}.")
            return

        print(f"Found {len(videos)} video(s) in {input_dir}.")

        for idx, video_path in enumerate(videos, start=1):
            filename = os.path.basename(video_path)
            name_no_ext = os.path.splitext(filename)[0]
            print(f"[{idx}/{len(videos)}] Processing: {filename}")
            self.process_video(
                video_path=video_path,
                output_name=name_no_ext,
            )

    def process_video(self, video_path: str, output_name: str) -> None:
        """Processes a single video file."""
        # Capture video metadata
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        cap.release()

        output_path = os.path.join(self.output_dir, f"{output_name}_processed.mp4")
        log_path = os.path.join(self.log_dir, f"{output_name}_tracking.csv")

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        logger = TrackingLogger(log_path)
        cargo_by_track: Dict[int, Optional[str]] = {}
        frame_index = 0

        # Use Ultralytics tracking in streaming mode to access each frame and its tracks.
        results_generator = self.model.track(
            source=video_path,
            tracker=self.tracker_config,
            stream=True,
            verbose=False,
        )

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

            for i in range(num_boxes):
                x1, y1, x2, y2 = map(int, xyxy[i])
                bbox = (x1, y1, x2, y2)

                track_id = int(ids[i]) if ids is not None else -1
                class_id = int(cls[i]) if cls is not None else -1

                # Filter only cargo class if class information is present
                if class_id != -1 and class_id != self.cargo_class_index:
                    continue

                # Perform OCR only once per track ID (or if unknown)
                if track_id not in cargo_by_track or cargo_by_track[track_id] is None:
                    number = self.ocr.extract_number(frame, bbox)
                    if number is not None:
                        cargo_by_track[track_id] = number
                    else:
                        # Ensure key exists, even if number is not yet known
                        cargo_by_track.setdefault(track_id, None)

                cargo_number = cargo_by_track.get(track_id)
                draw_tracked_box(
                    frame=frame,
                    bbox=bbox,
                    track_id=track_id,
                    cargo_number=cargo_number,
                )

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


def main() -> None:
    """Entry point for processing all videos in the input directory."""
    pipeline = VideoProcessingPipeline(
        model_path="models/best.pt",
        output_dir="videos/output",
        log_dir="videos/logs",
        tracker_config="bytetrack.yaml",  # Ultralytics built-in configuration
        cargo_class_index=0,  # Assuming class 0 is 'Cargo'
    )

    pipeline.process_folder("videos/input")


if __name__ == "__main__":
    main()
