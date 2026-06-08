"""
ANPR Detection Pipeline
-----------------------
Stage 1 – YOLOv8 detects the plate region in the frame.
Stage 2 – OpenCV preprocessing sharpens the crop.
Stage 3 – EasyOCR (or Tesseract fallback) reads the plate text.
"""

import os
import re
import uuid
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports – graceful degradation so the module loads even without GPU
# ---------------------------------------------------------------------------
try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False
    logger.warning("ultralytics not installed – YOLO detection disabled.")

try:
    import easyocr
    _EASYOCR_AVAILABLE = True
except ImportError:
    _EASYOCR_AVAILABLE = False
    logger.warning("easyocr not installed – falling back to Tesseract.")

try:
    import pytesseract
    _TESSERACT_AVAILABLE = True
except ImportError:
    _TESSERACT_AVAILABLE = False
    logger.warning("pytesseract not installed.")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CROPS_DIR       = Path("uploads/crops")
CROPS_DIR.mkdir(parents=True, exist_ok=True)

# Plate regex patterns (extendable per country)
PLATE_PATTERNS = [
    r"^[A-Z]{2}[0-9]{2}[A-Z]{3}$",          # UK new style  AB12 CDE
    r"^[A-Z]{1,3}[0-9]{1,4}[A-Z]{0,3}$",    # UK old style
    r"^[A-Z]{3}[0-9]{4}$",                   # PK format
    r"^[0-9]{1,4}[A-Z]{1,3}[0-9]{0,4}$",    # generic
]


def is_valid_plate(text: str) -> bool:
    """Heuristic validation – adjust patterns for your locale."""
    cleaned = re.sub(r"[\s\-\.]", "", text.upper())
    if len(cleaned) < 3 or len(cleaned) > 10:
        return False
    return any(re.fullmatch(p, cleaned) for p in PLATE_PATTERNS)


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------
@dataclass
class PlateResult:
    plate_text:      str
    confidence:      float          # OCR confidence
    detection_score: float = 0.0   # YOLO box confidence
    bbox:            tuple = field(default_factory=tuple)   # (x1, y1, x2, y2)
    crop_path:       Optional[str] = None
    valid:           bool = False


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_plate(crop: np.ndarray) -> np.ndarray:
    """
    Chain of OpenCV ops to maximise OCR accuracy:
      1. Upscale so characters are ≥ 30px tall
      2. Grayscale
      3. CLAHE for contrast normalisation
      4. Bilateral filter (removes noise, keeps edges)
      5. Adaptive threshold (handles uneven lighting)
    """
    h, w = crop.shape[:2]
    scale = max(1, 120 // h)          # ensure ≥ 120px height
    crop = cv2.resize(crop, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    denoised = cv2.bilateralFilter(gray, 11, 17, 17)

    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2,
    )
    return thresh


# ---------------------------------------------------------------------------
# OCR backends
# ---------------------------------------------------------------------------

class EasyOCRBackend:
    def __init__(self, langs=("en",)):
        if not _EASYOCR_AVAILABLE:
            raise RuntimeError("easyocr not installed")
        self._reader = easyocr.Reader(list(langs), gpu=False, verbose=False)

    def read(self, image: np.ndarray) -> tuple[str, float]:
        """Returns (plate_text, confidence)."""
        results = self._reader.readtext(image, detail=1, paragraph=False)
        if not results:
            return "", 0.0
        # pick the highest-confidence result
        best = max(results, key=lambda r: r[2])
        text = re.sub(r"[^A-Z0-9]", "", best[1].upper())
        return text, float(best[2])


class TesseractBackend:
    _CONFIG = "--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    def read(self, image: np.ndarray) -> tuple[str, float]:
        if not _TESSERACT_AVAILABLE:
            raise RuntimeError("pytesseract not installed")
        data = pytesseract.image_to_data(
            image, config=self._CONFIG,
            output_type=pytesseract.Output.DICT,
        )
        texts, confs = [], []
        for i, conf in enumerate(data["conf"]):
            if int(conf) > 0:
                texts.append(data["text"][i])
                confs.append(int(conf) / 100.0)
        text = re.sub(r"[^A-Z0-9]", "", " ".join(texts).upper())
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        return text, avg_conf


def _build_ocr_backend():
    if _EASYOCR_AVAILABLE:
        try:
            return EasyOCRBackend()
        except Exception as exc:
            logger.warning("EasyOCR init failed (%s); trying Tesseract.", exc)
    if _TESSERACT_AVAILABLE:
        return TesseractBackend()
    raise RuntimeError("No OCR backend available. Install easyocr or pytesseract.")


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------

class ANPRDetector:
    """
    Wraps YOLOv8 + preprocessing + OCR into one callable.

    Usage:
        detector = ANPRDetector()
        results  = detector.detect(frame)     # list[PlateResult]
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",   # swap for a fine-tuned plate model
        conf_threshold: float = 0.4,
        ocr_conf_threshold: float = 0.5,
        save_crops: bool = True,
    ):
        self.conf_threshold     = conf_threshold
        self.ocr_conf_threshold = ocr_conf_threshold
        self.save_crops         = save_crops

        # YOLO
        self._yolo = None
        if _YOLO_AVAILABLE:
            try:
                self._yolo = YOLO(model_path)
                logger.info("YOLOv8 loaded from %s", model_path)
            except Exception as exc:
                logger.error("YOLO load failed: %s", exc)

        # OCR
        self._ocr = _build_ocr_backend()
        logger.info("OCR backend: %s", type(self._ocr).__name__)

    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray, source: str = "unknown") -> list[PlateResult]:
        """
        Run the full pipeline on a single BGR frame.
        Returns a list of PlateResult (one per detected plate).
        """
        if self._yolo is None:
            # Fallback: treat entire frame as the plate region
            return [self._ocr_on_region(frame, (0, 0, frame.shape[1], frame.shape[0]), 1.0)]

        results = []
        yolo_out = self._yolo(frame, conf=self.conf_threshold, verbose=False)

        for det in yolo_out[0].boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
            score = float(det.conf[0])
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            result = self._ocr_on_region(crop, (x1, y1, x2, y2), score)
            if result.confidence >= self.ocr_conf_threshold and result.plate_text:
                if self.save_crops:
                    result.crop_path = self._save_crop(crop)
                result.valid = is_valid_plate(result.plate_text)
                results.append(result)

        return results

    def detect_from_path(self, image_path: str) -> list[PlateResult]:
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        return self.detect(frame, source=image_path)

    # ------------------------------------------------------------------

    def _ocr_on_region(self, crop, bbox, detection_score) -> PlateResult:
        processed = preprocess_plate(crop)
        text, confidence = self._ocr.read(processed)
        return PlateResult(
            plate_text=text,
            confidence=confidence,
            detection_score=detection_score,
            bbox=bbox,
        )

    @staticmethod
    def _save_crop(crop: np.ndarray) -> str:
        filename = CROPS_DIR / f"{uuid.uuid4().hex}.jpg"
        cv2.imwrite(str(filename), crop)
        return str(filename)
