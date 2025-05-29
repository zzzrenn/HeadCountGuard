import argparse
import os
import time

import cv2
import yaml
from loguru import logger

from counter import LineCrossingCounter
from detectors import PersonDetector
from tracker import PersonTracker


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main(config):
    # Initialize components
    detector = PersonDetector(**config["detector"])

    tracker = PersonTracker(**config["tracking"])
    counter = LineCrossingCounter(config["line"]["position"])

    # Initialize video capture
    cap = cv2.VideoCapture(config["video"]["path"])
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {config['video']['path']}")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate separation line position
    line_x = int(width * config["line"]["position"])

    # Setup video writer if saving results
    if config["video"]["save_result"]:
        timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        save_folder = os.path.join("outputs", timestamp)
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, os.path.basename(config["video"]["path"]))
        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )

    frame_id = 0
    results = []

    while True:
        if frame_id % 20 == 0:
            logger.info(f"Processing frame {frame_id}")

        ret_val, frame = cap.read()
        if not ret_val:
            break

        # Detect people
        detections = detector.detect(frame)

        # Track people
        tracked_objects = tracker.update(
            detections,
            [height, width],
            (height, width),  # Using original image size
        )

        # Update counter
        count = counter.update(tracked_objects, width)

        # Draw separation line
        cv2.line(frame, (line_x, 0), (line_x, height), (0, 255, 0), 2)

        # Draw in/out labels
        cv2.putText(
            frame, "IN", (line_x + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        cv2.putText(
            frame, "OUT", (line_x - 80, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )

        # Draw count
        cv2.putText(
            frame,
            f"Count: {count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # Draw bounding boxes and track IDs
        for obj in tracked_objects:
            bbox = obj["bbox"]
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID: {obj['track_id']}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            # Save results in MOT format
            results.append(
                f"{frame_id},{obj['track_id']},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{obj['score']:.2f},-1,-1,-1\n"
            )

        if config["video"]["save_result"]:
            vid_writer.write(frame)

        # Resize frame to fit screen while maintaining aspect ratio
        max_width = config["display"]["max_width"]
        max_height = config["display"]["max_height"]

        # Calculate scaling factor
        scale = min(max_width / width, max_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)

        # Resize frame
        frame = cv2.resize(frame, (new_width, new_height))

        # Display frame
        cv2.imshow("People Counter", frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_id += 1

    # Clean up
    cap.release()
    if config["video"]["save_result"]:
        vid_writer.release()
    cv2.destroyAllWindows()

    # Save results to file
    if config["video"]["save_result"]:
        results_path = os.path.join(save_folder, "results.txt")
        with open(results_path, "w") as f:
            f.writelines(results)
        logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="People Counter")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    main(config)
