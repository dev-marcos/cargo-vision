# CargoVision – Automated Trailer Detection, Tracking and Identification System

## 1. Overview

CargoVision is an end-to-end computer vision system designed to detect industrial cargo trailers, track their movement, and identify their unique ID numbers using OCR, leveraging only existing CCTV infrastructure. This solution eliminates the need for additional IoT hardware such as GPS, RFID, LoRaWAN or sensor tags.

The system is composed of a YOLOv8-based detector, ByteTrack multi-object tracking, and OCR-based number extraction. It has been developed as part of a research project in the context of manufacturing logistics, enabling real-time visibility of trailer positions and historical movement records.

---

## 2. Features

- Trailer (cargo) detection using YOLOv8  
- Multi-object tracking with ByteTrack  
- Automatic trailer ID (00–99) extraction using OCR  
- Dataset conversion pipeline from Label Studio → YOLO format  
- Separation of training, inference, OCR and dataset management modules  
- Configurable through YAML files  
- Clean, modular and production-ready codebase  
- Easily extendable for real-time deployment

---

## 3. Project Structure

```
cargo-vision/
│
├── configs/
│   ├── data_config.yaml
│   ├── train_config.yaml
│   ├── tracker_config.yaml
│
├── data/
│   ├── raw/
│   │   ├── images/
│   │   ├── labels/
│   ├── yolo_dataset/
│       ├── images/train/
│       ├── images/val/
│       ├── labels/train/
│       ├── labels/val/
│       ├── data.yaml
│
├── models/
│   ├── yolov8m.pt
│   ├── best.pt
│
├── notebooks/
│   ├── exploration.ipynb
│   ├── evaluation.ipynb
│
├── src/
│   ├── dataset/
│   │   ├── dataset_builder.py
│   │   ├── label_converter.py
│   │
│   ├── training/
│   │   ├── train_yolo.py
│   │
│   ├── inference/
│   │   ├── video_tracker.py
│   │
│   ├── ocr/
│   │   ├── ocr_reader.py
│   │
│   ├── utils/
│       ├── file_utils.py
│       ├── logging_utils.py
│
├── videos/
│   ├── input/
│   ├── output/
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 4. Installation

### 4.1 System Requirements

- Python 3.9+  
- CUDA-enabled GPU recommended (NVIDIA)  
- OS: Linux, Windows or macOS  

### 4.2 Install Dependencies

```bash
pip install -r requirements.txt
```

### 4.3 Additional Dependencies

Download YOLO weights (if not present):

```bash
mkdir -p models
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt -O models/yolov8m.pt
```

---

## 5. Dataset Workflow

### 5.1 Prepare Dataset From Label Studio

Place your exported images and labels here:

```
data/raw/images/
data/raw/labels/
```

### 5.2 Convert to YOLO Format

```bash
python src/dataset/dataset_builder.py
```

This will generate:

```
data/yolo_dataset/
├── images/train/
├── images/val/
├── labels/train/
├── labels/val/
└── data.yaml
```

---

## 6. Training the Detector

Edit configuration file:

```
configs/train_config.yaml
```

Start training:

```bash
python src/training/train_yolo.py
```

Output model will be stored in:

```
runs/detect/train/weights/best.pt
```

Move the trained model to:

```
models/best.pt
```

---

## 7. Video Tracking Inference

Place input videos in:

```
videos/input/
```

Run the inference:

```bash
python src/inference/video_tracker.py
```

Output results (annotated videos) will appear in:

```
videos/output/
```

---

## 8. OCR-Based Cargo Number Extraction

The OCR module is located in:

```
src/ocr/ocr_reader.py
```

It automatically:

- extracts the frontal region of the trailer  
- applies preprocessing  
- performs OCR using EasyOCR  
- filters valid identifiers (two-digit numbers)  

---

## 9. Technical Pipeline

### 9.1 Object Detection

- Model: YOLOv8m  
- Input resolution: 640×640  
- Classes:  
  - Cargo (trailer)  
  - Tractor  

### 9.2 Tracking

- Algorithm: ByteTrack  
- Responsibilities:  
  - Assign persistent track IDs  
  - Handle occlusion  
  - Maintain trajectory over time  

### 9.3 OCR

- Library: EasyOCR  
- Preprocessing:  
  - ROI extraction  
  - Grayscale conversion  
  - Gaussian blur  
  - Adaptive thresholding  
- Output: Cargo identifier (00–99)

---

## 10. Development Guidelines

- Modular architecture  
- Separation of concerns  
- Reproducible workflows  
- Configurable behavior via YAML files  
- Code documented in English  
- Suitable for research or production deployment  

---

## 11. Roadmap

| Feature | Status |
|--------|--------|
| YOLO detection | Complete |
| ByteTrack integration | Complete |
| OCR cargo ID extraction | Complete |
| Unified pipeline | In progress |
| 2D position mapping (homography) | Planned |
| Heatmaps of cargo movement | Planned |
| Real-time dashboard | Planned |
| Multi-camera integration | Planned |

---

## 12. License

This project is released under the MIT License.
