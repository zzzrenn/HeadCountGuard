# People Counter

A Python application that uses object detection and tracking to count people entering and leaving a defined space. The system detects people crossing a virtual line in a video feed and maintains an accurate count of entries and exits.

## Features

- **Real-time Person Detection**: Uses YOLOv8 for accurate person detection
- **Multi-Object Tracking**: Implements ByteTrack for robust tracking across frames
- **Line Crossing Detection**: Configurable virtual line to define entry/exit boundaries
- **Visual Interface**: Real-time display with bounding boxes, track IDs, and count overlay
- **Video Processing**: Support for recorded video files
- **Configurable Settings**: YAML-based configuration for easy customization
- **Customization**: Easily extend with customise detection and tracker algorithms

## Use Cases

- **Retail Analytics**: Monitor customer traffic in stores
- **Security Systems**: Track people entering/leaving secure areas
- **Event Management**: Count attendees at venues or events
- **Public Spaces**: Monitor foot traffic in parks, libraries, or buildings
- **Research**: Analyze pedestrian behavior and movement patterns

## Demo
[![Demo Video](https://img.youtube.com/vi/WjjeQFLygXU/maxresdefault.jpg)](https://youtu.be/WjjeQFLygXU)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/zzzrenn/HeadCountGuard
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download YOLOv8 model:
```bash
# The yolov8s.pt model should be placed in the models/ folder
```

## Usage
### GUI Application

Launch the GUI interface:
```bash
make run-app
```

### Configuration

Edit `configs/yolo_config.yaml` to customize:

- **Video source**: Path to video file
- **Detection settings**: Confidence thresholds and model parameters
- **Tracking parameters**: Tracking sensitivity and buffer settings
