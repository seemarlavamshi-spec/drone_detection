"""
DRONE OBJECT DETECTION SYSTEM USING AI
A Streamlit Web Application for Aerial Surveillance, YOLOv8/YOLO11 Detection,
ByteTrack Multi-Object Tracking, Folium GPS Mapping, Restricted-Area Intrusion Alerts,
and SQLite Intelligence Database.
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import folium
from streamlit_folium import st_folium
import tempfile
import os
import uuid
from datetime import datetime

# Import modular components
from detector import DroneYOLODetector
from tracker import DroneByteTracker
from geofence import GeofenceManager, RestrictedZone
from telemetry import DroneTelemetryHandler, TelemetryMode
import database as db

# Page Configuration - Dark Navy & Blue Dashboard
st.set_page_config(
    page_title="Drone Object Detection System Using AI",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Tactical Dark Navy & Blue CSS Styling
st.markdown("""
<style>
    .stApp {
        background-color: #080d1a;
        color: #e2e8f0;
    }
    .css-1d391kg, .css-12oz5g7 {
        background-color: #0b1329;
    }
    div[data-testid="stMetric"] {
        background-color: #0f1c3f;
        border: 1px solid #1e3a8a;
        border-radius: 8px;
        padding: 12px 16px;
    }
    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.8rem !important;
    }
    .stButton > button {
        background-color: #1d4ed8;
        color: white;
        border-radius: 6px;
        border: 1px solid #2563eb;
        font-weight: 600;
    }
    .stButton > button:hover {
        background-color: #2563eb;
        border-color: #3b82f6;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0b1329;
        padding: 6px;
        border-radius: 8px;
        border: 1px solid #1e293b;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        color: #94a3b8;
        padding: 8px 16px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e3a8a !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize SQLite database
db.init_db()

# Session State Initialization
if "session_id" not in st.session_state:
    st.session_state.session_id = f"MISSION-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

if "geofence_mgr" not in st.session_state:
    st.session_state.geofence_mgr = GeofenceManager()

if "telemetry_mgr" not in st.session_state:
    st.session_state.telemetry_mgr = DroneTelemetryHandler()

if "detector" not in st.session_state:
    st.session_state.detector = None

if "tracker" not in st.session_state:
    st.session_state.tracker = DroneByteTracker()

if "active_alerts" not in st.session_state:
    st.session_state.active_alerts = []

# Sidebar Controls
with st.sidebar:
    st.markdown("### 🚁 MISSION CONTROL")
    st.markdown(f"**Session ID:** `{st.session_state.session_id}`")
    st.divider()

    st.markdown("#### ⚙️ AI Detection Parameters")
    model_choice = st.selectbox(
        "YOLO Neural Model",
        ["yolov8n.pt", "yolov8s.pt", "yolo11n.pt"],
        index=0
    )
    conf_threshold = st.slider(
        "Confidence Threshold (%)",
        min_value=10,
        max_value=95,
        value=35,
        step=5
    ) / 100.0

    iou_threshold = st.slider(
        "NMS IoU Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.45,
        step=0.05
    )

    selected_classes = st.multiselect(
        "Target Detection Classes",
        options=["person", "car", "bus", "truck", "motorcycle", "bicycle", "boat"],
        default=["person", "car", "bus", "truck", "motorcycle"]
    )

    st.divider()
    st.markdown("#### 🛰️ GPS & Telemetry Engine")
    telemetry_selection = st.radio(
        "GPS Coordinates Mode",
        ["Automatic Detection", "Force Demonstration Mode", "Real Drone GPS"],
        index=0
    )
    if telemetry_selection == "Force Demonstration Mode":
        st.session_state.telemetry_mgr.mode = TelemetryMode.DEMONSTRATION
    elif telemetry_selection == "Real Drone GPS":
        st.session_state.telemetry_mgr.mode = TelemetryMode.REAL_GPS

    st.divider()
    if st.button("🔄 Reset Mission Session", use_container_width=True):
        st.session_state.session_id = f"MISSION-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        st.session_state.tracker.reset()
        st.session_state.active_alerts = []
        st.rerun()

# Instantiate detector when parameters change
if (st.session_state.detector is None or 
    st.session_state.detector.model_name != model_choice or 
    st.session_state.detector.confidence != conf_threshold):
    st.session_state.detector = DroneYOLODetector(
        model_name=model_choice,
        confidence=conf_threshold,
        iou_thresh=iou_threshold
    )
st.session_state.detector.set_target_classes(selected_classes)

st.title("DRONE OBJECT DETECTION SYSTEM USING AI")
st.caption("AI Aerial Reconnaissance, Multi-Object ByteTrack Tracking, GIS Folium Geofencing, and Intrusion Defense System")

if st.session_state.telemetry_mgr.mode == TelemetryMode.DEMONSTRATION:
    st.info("ℹ️ DEMONSTRATION MODE ACTIVE: Target and drone positions are projected using simulation reference coordinates.")
else:
    st.success("🛰️ AUTHENTIC GPS TELEMETRY ACTIVE: Reading real GPS coordinates from drone telemetry stream.")

tab_surveillance, tab_map, tab_analytics, tab_alerts, tab_database = st.tabs([
    "📹 Live Video Surveillance",
    "🗺️ Tactical GPS Map & Geofence",
    "📊 Object Analytics & Counts",
    "🚨 Restricted-Area Alerts",
    "💾 SQLite Database & Reports"
])

with tab_surveillance:
    col_upload, col_controls = st.columns([2, 1])
    with col_upload:
        uploaded_video = st.file_uploader(
            "Upload Drone Reconnaissance Video (MP4, AVI, MOV)",
            type=["mp4", "avi", "mov"]
        )
    with col_controls:
        process_every_n = st.slider("Process Every Nth Frame", min_value=1, max_value=5, value=2)
        save_processed_video = st.checkbox("Export Processed Video", value=True)
        max_frames_to_process = st.number_input("Max Frames", min_value=30, max_value=1500, value=300, step=50)

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_video.name)[1])
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        st.session_state.telemetry_mgr.inspect_video_metadata(video_path)
        cap = cv2.VideoCapture(video_path)
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        st.markdown(f"**Video Info:** {frame_w}x{frame_h} @ {video_fps:.1f} FPS | Total Frames: {total_video_frames}")

        if st.button("▶️ Start Drone Video Processing", use_container_width=True):
            st.session_state.tracker.reset()
            st.session_state.active_alerts = []

            video_frame_placeholder = st.empty()
            progress_bar = st.progress(0.0)

            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            kpi_unique = metric_col1.empty()
            kpi_active = metric_col2.empty()
            kpi_alerts = metric_col3.empty()
            kpi_altitude = metric_col4.empty()

            out_video_path = None
            writer = None
            if save_processed_video:
                out_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(out_video_path, fourcc, video_fps / process_every_n, (frame_w, frame_h))

            frame_idx = 0
            processed_count = 0

            while cap.isOpened() and processed_count < max_frames_to_process:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                if frame_idx % process_every_n != 0:
                    continue
                processed_count += 1

                detections = st.session_state.detector.detect(frame)
                tracks = st.session_state.tracker.update(detections, (frame_w, frame_h))
                telemetry = st.session_state.telemetry_mgr.get_telemetry_at_frame(frame_idx, total_video_frames, video_fps)

                db.insert_telemetry(
                    st.session_state.session_id, telemetry["altitude_m"], telemetry["speed_kmh"],
                    telemetry["heading_deg"], telemetry["battery_pct"], telemetry["latitude"],
                    telemetry["longitude"], telemetry["mode"]
                )

                canvas = frame.copy()
                for trk in tracks:
                    t_id = trk["track_id"]
                    x1, y1, x2, y2 = [int(v) for v in trk["bbox"]]
                    c_name = trk["class_name"]
                    conf = trk["confidence"]
                    color = (0, 255, 127) if c_name == "person" else (255, 191, 0)

                    target_lat, target_lon = st.session_state.telemetry_mgr.project_pixel_to_gps(
                        (x1 + x2) / 2.0, (y1 + y2) / 2.0, frame_w, frame_h,
                        telemetry["latitude"], telemetry["longitude"], telemetry["altitude_m"]
                    )

                    db.insert_detection(st.session_state.session_id, frame_idx, t_id, c_name, conf, (x1, y1, x2, y2), target_lat, target_lon)

                    alert = st.session_state.geofence_mgr.check_intrusion(t_id, c_name, target_lat, target_lon)
                    if alert:
                        db.insert_alert(st.session_state.session_id, t_id, c_name, alert["zone_name"], alert["severity"], alert["description"], target_lat, target_lon)
                        st.session_state.active_alerts.append(alert)
                        color = (0, 0, 255)

                    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
                    label_text = f"ID #{t_id} | {c_name.upper()} {int(conf * 100)}%"
                    cv2.putText(canvas, label_text, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if writer is not None:
                    writer.write(canvas)

                canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
                video_frame_placeholder.image(canvas_rgb, channels="RGB", use_container_width=True)

                kpi_unique.metric("Unique Objects", st.session_state.tracker.get_total_unique())
                kpi_active.metric("Active Tracks", len(tracks))
                kpi_alerts.metric("Alerts", len(st.session_state.active_alerts))
                kpi_altitude.metric("Altitude", f"{telemetry['altitude_m']}m")

                progress = min(1.0, processed_count / max_frames_to_process)
                progress_bar.progress(progress)

            cap.release()
            if writer is not None:
                writer.release()
            st.success("✅ Video Processing Completed!")

            if out_video_path and os.path.exists(out_video_path):
                with open(out_video_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Processed Video (.mp4)",
                        data=f.read(),
                        file_name=f"processed_drone_{st.session_state.session_id}.mp4",
                        mime="video/mp4",
                        use_container_width=True
                    )

with tab_map:
    st.markdown("### 🗺️ Tactical GIS Map & Restricted Area Geofencing")
    drone_lat = st.session_state.telemetry_mgr.base_lat
    drone_lon = st.session_state.telemetry_mgr.base_lon

    m = folium.Map(location=[drone_lat, drone_lon], zoom_start=16, tiles="CartoDB dark_matter")
    for zone in st.session_state.geofence_mgr.get_all_zones():
        folium.Polygon(locations=zone.coordinates, color=zone.color, fill=True, fill_color=zone.color, fill_opacity=0.35, tooltip=zone.name).add_to(m)

    folium.Marker(location=[drone_lat, drone_lon], icon=folium.Icon(color="blue", icon="plane", prefix="fa"), tooltip="UAV Drone Alpha").add_to(m)
    for alert in st.session_state.active_alerts:
        folium.CircleMarker(location=[alert["lat"], alert["lon"]], radius=8, color="#DC2626", fill=True, fill_color="#EF4444", tooltip=f"Alert: {alert['class_name']}").add_to(m)
    st_folium(m, width="100%", height=500)

with tab_analytics:
    st.markdown("### 📊 Unique Object Statistics & Class Analytics")
    stats = db.get_summary_statistics(st.session_state.session_id)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Unique Tracked", stats["total_unique_objects"])
    c2.metric("Frames Analyzed", stats["total_frames"])
    c3.metric("Alerts Recorded", stats["total_alerts"])
    if stats["class_counts"]:
        df_classes = pd.DataFrame(list(stats["class_counts"].items()), columns=["Class", "Count"])
        st.bar_chart(df_classes.set_index("Class"))

with tab_alerts:
    st.markdown("### 🚨 Perimeter Intrusion & Security Alert Log")
    alerts_df = db.get_all_alerts_df(st.session_state.session_id)
    if not alerts_df.empty:
        st.dataframe(alerts_df, use_container_width=True)
        st.download_button("📥 Download Alerts (CSV)", alerts_df.to_csv(index=False).encode('utf-8'), f"alerts_{st.session_state.session_id}.csv", "text/csv")
    else:
        st.success("✅ Secure Perimeter: No active intrusions detected.")

with tab_database:
    st.markdown("### 💾 SQLite Surveillance Database & Export Center")
    det_df = db.get_all_detections_df(st.session_state.session_id)
    st.dataframe(det_df.head(100), use_container_width=True)
    if not det_df.empty:
        st.download_button("📥 Export Detections (CSV)", det_df.to_csv(index=False).encode('utf-8'), f"detections_{st.session_state.session_id}.csv", "text/csv")
