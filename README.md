# Koi Health Computer Vision Prototype

Prototype computer-vision system for detecting koi movement and position from video.

The first version uses OpenCV edge detection and contour tracking to estimate koi positions and flag possible distress behaviors such as piping and diving.

## Install

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## Run

python src/koi_video_edge_tracker.py --input path/to/koi_video.mp4 --output outputs/annotated_koi.mp4 --show-edges

## Notes

This is a baseline prototype, not a clinical classifier.
# Koi-Edge-Detection
