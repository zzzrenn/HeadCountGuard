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

    # Get line points from config
    line_points = (
        (config["line"]["x1"], config["line"]["y1"]),
        (config["line"]["x2"], config["line"]["y2"]),
    )
    counter = LineCrossingCounter(line_points, config["line"]["in_side"])

    # Initialize video capture
    cap = cv2.VideoCapture(config["video"]["path"])
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {config['video']['path']}")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate line endpoints in pixel coordinates
    p1 = (int(width * config["line"]["x1"]), int(height * config["line"]["y1"]))
    p2 = (int(width * config["line"]["x2"]), int(height * config["line"]["y2"]))
    if p1[1] > p2[1]:
        p1, p2 = p2, p1

    # find minimum scale factor to fit the line in the frame
    scale = min(width / (p2[0] - p1[0]), height / (p2[1] - p1[1]))
    p1 = (int(p1[0] * scale), int(p1[1] * scale))
    p2 = (int(p2[0] * scale), int(p2[1] * scale))

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
    previous_count = 0
    while True:
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
        count = counter.update(tracked_objects, width, height)
        if count != previous_count:
            logger.info(f"Frame {frame_id}: Count={count}")
            previous_count = count

        # Draw separation line
        cv2.line(frame, p1, p2, (0, 255, 0), 2)

        # Calculate line direction vector for label placement
        line_vector = (p2[0] - p1[0], p2[1] - p1[1])
        line_length = (line_vector[0] ** 2 + line_vector[1] ** 2) ** 0.5
        line_normal = (-line_vector[1] / line_length, line_vector[0] / line_length)

        # Draw in/out labels
        label_offset = 30
        in_label_pos = (
            int((p1[0] + p2[0]) / 2 + line_normal[0] * label_offset),
            int((p1[1] + p2[1]) / 2 + line_normal[1] * label_offset),
        )
        out_label_pos = (
            int((p1[0] + p2[0]) / 2 - line_normal[0] * label_offset),
            int((p1[1] + p2[1]) / 2 - line_normal[1] * label_offset),
        )

        cv2.putText(
            frame, "IN", in_label_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        cv2.putText(
            frame, "OUT", out_label_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
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
