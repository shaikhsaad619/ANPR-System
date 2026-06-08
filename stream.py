"""
Live camera / RTSP stream processor.
Runs independently of Flask – pipe results straight to the DB and optionally
forward webhook alerts for flagged plates.

Usage:
    python stream.py --source 0              # webcam index 0
    python stream.py --source rtsp://…      # RTSP camera
    python stream.py --source video.mp4     # test with file
    python stream.py --source 0 --webhook https://hooks.example.com/anpr
"""

import argparse
import logging
import time
import requests

import cv2

from detector import ANPRDetector
from database import db

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def send_webhook(url: str, payload: dict):
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception as exc:
        logger.warning("Webhook failed: %s", exc)


def run(source, interval: float = 1.0, webhook: str = None, show: bool = True):
    detector = ANPRDetector()

    cap_source = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    logger.info("Streaming from %s  (press Q to quit)", source)
    last_process = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.info("Stream ended or no frame.")
            break

        now = time.time()
        if now - last_process >= interval:
            last_process = now
            results = detector.detect(frame, source=str(source))
            for r in results:
                if not r.plate_text:
                    continue
                det = db.save_detection(
                    plate_text=r.plate_text,
                    confidence=r.confidence,
                    detection_score=r.detection_score,
                    image_path=r.crop_path,
                    source=str(source),
                )
                logger.info(
                    "Plate: %-12s  conf: %.2f  flagged: %s",
                    det.plate_text, det.confidence, det.is_flagged
                )
                if det.is_flagged and webhook:
                    send_webhook(webhook, det.to_dict())

                # Draw bounding box on frame
                if r.bbox and show:
                    x1, y1, x2, y2 = r.bbox
                    colour = (0, 0, 255) if det.is_flagged else (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                    cv2.putText(frame, r.plate_text, (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)

        if show:
            cv2.imshow("ANPR Live", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ANPR live stream processor")
    parser.add_argument("--source",   default="0",  help="Camera index, file path, or RTSP URL")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between detections")
    parser.add_argument("--webhook",  default=None, help="URL to POST flagged-plate alerts")
    parser.add_argument("--no-show",  action="store_true", help="Disable OpenCV window")
    args = parser.parse_args()
    run(args.source, args.interval, args.webhook, show=not args.no_show)
