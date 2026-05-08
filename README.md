# Koi Health Computer Vision Prototype

This project is a prototype computer vision pipeline for analyzing koi fish behavior from video data. The system tracks movement and spatial positioning to identify potential distress indicators.

The current implementation uses classical image processing techniques (OpenCV) as a baseline before introducing machine learning models.

---

## Features

- Edge-based detection using Canny filtering  
- Contour extraction and filtering for fish candidates  
- Multi-object centroid tracking  
- Movement trajectory visualization  
- Heuristic behavior detection:
  - **Piping**: prolonged presence near the water surface with minimal movement  
  - **Diving**: rapid downward motion over a short time window  

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

```bash
python src/koi_video_edge_tracker.py \
    --input path/to/koi_video.mp4 \
    --output outputs/annotated_koi.mp4 \
    --show-edges
```

---

## Methodology

1. Frame preprocessing (grayscale, blur, histogram equalization)  
2. Edge detection (Canny)  
3. Morphological operations  
4. Contour detection and filtering  
5. Centroid tracking  
6. Rule-based behavioral classification  

---

## Limitations

- Sensitive to lighting and water reflections  
- Cannot reliably distinguish overlapping fish  
- Behavior detection is heuristic, not learned  
- Not suitable for clinical use  

---

## Roadmap

- Replace contour detection with segmentation (YOLO / Mask R-CNN)  
- Add learned tracking (Deep SORT)  
- Train behavior classifiers  
- Introduce temporal models (RNN / Transformer)  

---

## Disclaimer

This is an experimental prototype and not intended for real-world health decisions.
