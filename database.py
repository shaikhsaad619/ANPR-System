"""
Database layer for ANPR system.
Now includes Vehicle Registration table with owner info.
"""

import os
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, String, Float,
    DateTime, Text, Boolean, Index
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
import logging

logger = logging.getLogger(__name__)
Base = declarative_base()


class Vehicle(Base):
    """Registered vehicle with owner details."""
    __tablename__ = "vehicles"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    plate_text   = Column(String(20), nullable=False, unique=True, index=True)
    owner_name   = Column(String(100), nullable=False)
    owner_cnic   = Column(String(20), nullable=True)
    owner_phone  = Column(String(20), nullable=True)
    brand        = Column(String(50), nullable=True)
    model        = Column(String(50), nullable=True)
    model_year   = Column(Integer, nullable=True)
    color        = Column(String(30), nullable=True)
    engine_cc    = Column(String(20), nullable=True)
    registered_at = Column(DateTime, default=datetime.utcnow)
    notes        = Column(Text, nullable=True)

    def to_dict(self):
        return {
            "id":           self.id,
            "plate_text":   self.plate_text,
            "owner_name":   self.owner_name,
            "owner_cnic":   self.owner_cnic,
            "owner_phone":  self.owner_phone,
            "brand":        self.brand,
            "model":        self.model,
            "model_year":   self.model_year,
            "color":        self.color,
            "engine_cc":    self.engine_cc,
            "registered_at": self.registered_at.isoformat() if self.registered_at else None,
            "notes":        self.notes,
        }


class Detection(Base):
    """Stores every plate detection event."""
    __tablename__ = "detections"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    plate_text      = Column(String(20), nullable=False, index=True)
    confidence      = Column(Float, nullable=False)
    detection_score = Column(Float, nullable=True)
    image_path      = Column(Text, nullable=True)
    source          = Column(String(255), nullable=True)
    timestamp       = Column(DateTime, default=datetime.utcnow, index=True)
    is_flagged      = Column(Boolean, default=False)

    __table_args__ = (
        Index("ix_plate_ts", "plate_text", "timestamp"),
    )

    def to_dict(self):
        return {
            "id":              self.id,
            "plate_text":      self.plate_text,
            "confidence":      round(self.confidence, 4),
            "detection_score": round(self.detection_score or 0, 4),
            "image_path":      self.image_path,
            "source":          self.source,
            "timestamp":       self.timestamp.isoformat() if self.timestamp else None,
            "is_flagged":      self.is_flagged,
        }


class Watchlist(Base):
    """Plates of interest (stolen vehicles, etc.)."""
    __tablename__ = "watchlist"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    plate_text = Column(String(20), nullable=False, unique=True, index=True)
    reason     = Column(String(255), nullable=True)
    added_at   = Column(DateTime, default=datetime.utcnow)
    active     = Column(Boolean, default=True)


class DatabaseManager:
    def __init__(self, db_url: str = "sqlite:///anpr.db"):
        self.engine = create_engine(
            db_url,
            connect_args={"check_same_thread": False} if db_url.startswith("sqlite") else {},
            echo=False,
        )
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False)
        logger.info("Database initialised at %s", db_url)

    def get_session(self) -> Session:
        return self.SessionLocal()

    # ------------------------------------------------------------------
    # Vehicle Registration
    # ------------------------------------------------------------------

    def register_vehicle(self, data: dict) -> Vehicle:
        with self.get_session() as session:
            plate = data.get("plate_text", "").upper().strip()
            existing = session.query(Vehicle).filter_by(plate_text=plate).first()
            if existing:
                for k, v in data.items():
                    if k != "plate_text" and v is not None:
                        setattr(existing, k, v)
                existing.plate_text = plate
                session.commit()
                session.refresh(existing)
                return existing
            v = Vehicle(
                plate_text=plate,
                owner_name=data.get("owner_name", ""),
                owner_cnic=data.get("owner_cnic"),
                owner_phone=data.get("owner_phone"),
                brand=data.get("brand"),
                model=data.get("model"),
                model_year=data.get("model_year"),
                color=data.get("color"),
                engine_cc=data.get("engine_cc"),
                notes=data.get("notes"),
            )
            session.add(v)
            session.commit()
            session.refresh(v)
            return v

    def get_vehicle(self, plate_text: str):
        with self.get_session() as session:
            v = session.query(Vehicle).filter_by(
                plate_text=plate_text.upper().strip()
            ).first()
            return v.to_dict() if v else None

    def get_all_vehicles(self, limit: int = 200):
        with self.get_session() as session:
            rows = session.query(Vehicle).order_by(Vehicle.registered_at.desc()).limit(limit).all()
            return [r.to_dict() for r in rows]

    def delete_vehicle(self, plate_text: str) -> bool:
        with self.get_session() as session:
            v = session.query(Vehicle).filter_by(plate_text=plate_text.upper().strip()).first()
            if v:
                session.delete(v)
                session.commit()
                return True
            return False

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------

    def save_detection(self, plate_text: str, confidence: float,
                       detection_score: float = 0.0, image_path: str = None,
                       source: str = None) -> Detection:
        with self.get_session() as session:
            flagged = self._is_flagged(session, plate_text)
            det = Detection(
                plate_text=plate_text.upper().strip(),
                confidence=confidence,
                detection_score=detection_score,
                image_path=image_path,
                source=source,
                is_flagged=flagged,
            )
            session.add(det)
            session.commit()
            session.refresh(det)
            return det

    def get_recent(self, limit: int = 50):
        with self.get_session() as session:
            rows = (
                session.query(Detection)
                .order_by(Detection.timestamp.desc())
                .limit(limit)
                .all()
            )
            return [r.to_dict() for r in rows]

    def search_plate(self, plate_text: str, limit: int = 100):
        with self.get_session() as session:
            rows = (
                session.query(Detection)
                .filter(Detection.plate_text.ilike(f"%{plate_text}%"))
                .order_by(Detection.timestamp.desc())
                .limit(limit)
                .all()
            )
            return [r.to_dict() for r in rows]

    # ------------------------------------------------------------------
    # Watchlist helpers
    # ------------------------------------------------------------------

    def add_to_watchlist(self, plate_text: str, reason: str = ""):
        with self.get_session() as session:
            entry = Watchlist(plate_text=plate_text.upper().strip(), reason=reason)
            session.merge(entry)
            session.commit()

    def _is_flagged(self, session: Session, plate_text: str) -> bool:
        return (
            session.query(Watchlist)
            .filter_by(plate_text=plate_text.upper().strip(), active=True)
            .first()
        ) is not None


_db_url = os.getenv("DB_URL", "sqlite:///anpr.db")
db = DatabaseManager(_db_url)
