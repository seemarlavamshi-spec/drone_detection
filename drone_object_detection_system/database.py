"""
SQLite Database Module for Drone Object Detection System.
Stores detection records, intrusion alerts, and drone flight telemetry.
"""

import sqlite3
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional

DB_NAME = "drone_surveillance.db"


def get_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            frame_id INTEGER NOT NULL,
            track_id INTEGER NOT NULL,
            class_name TEXT NOT NULL,
            confidence REAL NOT NULL,
            bbox_x1 REAL NOT NULL,
            bbox_y1 REAL NOT NULL,
            bbox_x2 REAL NOT NULL,
            bbox_y2 REAL NOT NULL,
            estimated_lat REAL,
            estimated_lon REAL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            track_id INTEGER NOT NULL,
            class_name TEXT NOT NULL,
            zone_name TEXT NOT NULL,
            severity TEXT NOT NULL,
            lat REAL,
            lon REAL,
            description TEXT,
            status TEXT DEFAULT 'ACTIVE'
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS flight_telemetry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            altitude_m REAL,
            speed_kmh REAL,
            heading_deg REAL,
            battery_pct INTEGER,
            lat REAL,
            lon REAL,
            mode TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def insert_detection(session_id: str, frame_id: int, track_id: int, class_name: str,
                     confidence: float, bbox: tuple, lat: Optional[float] = None, lon: Optional[float] = None):
    conn = get_connection()
    cursor = conn.cursor()
    x1, y1, x2, y2 = bbox
    now = datetime.now().isoformat()
    cursor.execute("""
        INSERT INTO detections 
        (session_id, timestamp, frame_id, track_id, class_name, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2, estimated_lat, estimated_lon)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (session_id, now, frame_id, track_id, class_name, round(confidence, 4), x1, y1, x2, y2, lat, lon))
    conn.commit()
    conn.close()


def insert_alert(session_id: str, track_id: int, class_name: str, zone_name: str,
                 severity: str, description: str, lat: Optional[float] = None, lon: Optional[float] = None):
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    cursor.execute("""
        INSERT INTO alerts (session_id, timestamp, track_id, class_name, zone_name, severity, lat, lon, description, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'ACTIVE')
    """, (session_id, now, track_id, class_name, zone_name, severity, lat, lon, description))
    conn.commit()
    conn.close()


def insert_telemetry(session_id: str, altitude: float, speed: float, heading: float,
                     battery: int, lat: float, lon: float, mode: str):
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    cursor.execute("""
        INSERT INTO flight_telemetry 
        (session_id, timestamp, altitude_m, speed_kmh, heading_deg, battery_pct, lat, lon, mode)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (session_id, now, altitude, speed, heading, battery, lat, lon, mode))
    conn.commit()
    conn.close()


def get_all_detections_df(session_id: Optional[str] = None) -> pd.DataFrame:
    conn = get_connection()
    query = "SELECT * FROM detections"
    params = []
    if session_id:
        query += " WHERE session_id = ?"
        params.append(session_id)
    query += " ORDER BY id DESC"
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def get_all_alerts_df(session_id: Optional[str] = None) -> pd.DataFrame:
    conn = get_connection()
    query = "SELECT * FROM alerts"
    params = []
    if session_id:
        query += " WHERE session_id = ?"
        params.append(session_id)
    query += " ORDER BY id DESC"
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def get_telemetry_df(session_id: Optional[str] = None) -> pd.DataFrame:
    conn = get_connection()
    query = "SELECT * FROM flight_telemetry"
    params = []
    if session_id:
        query += " WHERE session_id = ?"
        params.append(session_id)
    query += " ORDER BY id ASC"
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def get_summary_statistics(session_id: Optional[str] = None) -> Dict[str, Any]:
    conn = get_connection()
    cursor = conn.cursor()
    clause = "WHERE session_id = ?" if session_id else ""
    params = (session_id,) if session_id else ()

    cursor.execute(f"SELECT COUNT(DISTINCT frame_id) FROM detections {clause}", params)
    total_frames = cursor.fetchone()[0] or 0

    cursor.execute(f"SELECT COUNT(DISTINCT track_id) FROM detections {clause}", params)
    total_unique_objects = cursor.fetchone()[0] or 0

    cursor.execute(f"SELECT COUNT(*) FROM alerts {clause}", params)
    total_alerts = cursor.fetchone()[0] or 0

    cursor.execute(f"""
        SELECT class_name, COUNT(DISTINCT track_id) as count 
        FROM detections {clause} 
        GROUP BY class_name
        ORDER BY count DESC
    """, params)
    class_counts = {row["class_name"]: row["count"] for row in cursor.fetchall()}

    conn.close()
    return {
        "total_frames": total_frames,
        "total_unique_objects": total_unique_objects,
        "total_alerts": total_alerts,
        "class_counts": class_counts
    }
