# ANPR – Automated Number Plate Recognition

Full-stack Python system: **YOLOv8** detection → **OpenCV** preprocessing → **EasyOCR** text extraction → **SQLAlchemy** database → **Flask** REST API + dashboard.

```
anpr/
├── detector.py      # YOLOv8 + OpenCV + OCR pipeline
├── database.py      # SQLAlchemy models (SQLite / PostgreSQL)
├── app.py           # Flask API + web dashboard
├── stream.py        # Live camera / RTSP stream processor
├── requirements.txt
└── uploads/
    └── crops/       # Saved plate crop images
```

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Install Tesseract binary for fallback OCR
#    Ubuntu/Debian:  sudo apt install tesseract-ocr
#    macOS:          brew install tesseract

# 3. Run the API server
python app.py
# → http://localhost:5000  (dashboard + API)

# 4. Run the live stream processor (separate terminal)
python stream.py --source 0                      # webcam
python stream.py --source rtsp://192.168.1.1/…  # IP cam
python stream.py --source test_video.mp4         # file
```

## API reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/detect` | Upload image → detect plates |
| `GET`  | `/api/detections?limit=50` | Recent detections |
| `GET`  | `/api/search?q=ABC` | Search by plate text |
| `POST` | `/api/watchlist` | Add plate to watchlist |
| `GET`  | `/api/stats` | Aggregated statistics |

### POST /api/detect

```bash
curl -X POST http://localhost:5000/api/detect \
     -F "image=@plate.jpg"
```

```json
{
  "detected": 1,
  "plates": [{
    "id": 42,
    "plate_text": "AB12CDE",
    "confidence": 0.9312,
    "detection_score": 0.8754,
    "is_flagged": false,
    "timestamp": "2024-11-05T14:30:00"
  }]
}
```

### POST /api/watchlist

```bash
curl -X POST http://localhost:5000/api/watchlist \
     -H "Content-Type: application/json" \
     -d '{"plate": "XY99ZZZ", "reason": "stolen vehicle"}'
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_URL` | `sqlite:///anpr.db` | Any SQLAlchemy URL (swap to PostgreSQL) |
| `YOLO_MODEL` | `yolov8n.pt` | Path to YOLOv8 weights |
| `YOLO_CONF` | `0.4` | YOLO detection confidence threshold |
| `OCR_CONF` | `0.5` | OCR confidence threshold |
| `PORT` | `5000` | Flask port |

## Using a fine-tuned plate detection model

The default `yolov8n.pt` is the general COCO model. For best accuracy, use a
model fine-tuned on licence plates (e.g. from Roboflow Universe):

```bash
YOLO_MODEL=license_plate_detector.pt python app.py
```

## PostgreSQL switch

```bash
DB_URL=postgresql://user:password@localhost/anpr python app.py
```
