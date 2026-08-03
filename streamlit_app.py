"""Streamlit deployment prototype for the hawker-centre YOLO11n model."""

from __future__ import annotations

import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from ultralytics import YOLO


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models_yolo11n_hp" / "run1_baseline_50epochs_best.pt"
CLASS_NAMES = ("empty_seat", "tray", "uncleared_tableware", "pest_bird")
CLASS_LABELS = {
    "empty_seat": "Empty seats",
    "tray": "Trays",
    "uncleared_tableware": "Uncleared tableware",
    "pest_bird": "Pest birds",
}


@dataclass
class DetectionOutput:
    annotated_bgr: np.ndarray
    counts: dict[str, int]
    rows: list[dict[str, object]]
    inference_ms: float


def status_from_counts(counts: dict[str, int]) -> dict[str, object]:
    """Convert independent object counts into operational monitoring states."""
    cleaning_items = counts.get("tray", 0) + counts.get("uncleared_tableware", 0)
    pest_found = counts.get("pest_bird", 0) > 0
    return {
        "available_seats_detected": counts.get("empty_seat", 0),
        "cleaning_attention_required": cleaning_items > 0,
        "cleaning_items_detected": cleaning_items,
        "pest_alert": pest_found,
        "priority": "HIGH" if pest_found else ("MEDIUM" if cleaning_items else "NORMAL"),
    }


@st.cache_resource(show_spinner="Loading the final YOLO11n checkpoint...")
def load_model(weights: str) -> YOLO:
    return YOLO(weights)


def run_detection(
    model: YOLO,
    image_bgr: np.ndarray,
    confidence: float,
    iou: float,
    image_size: int,
    device: str | int,
) -> DetectionOutput:
    """Run YOLO on one BGR frame and return annotated output and structured data."""
    started = time.perf_counter()
    result = model.predict(
        source=image_bgr,
        conf=confidence,
        iou=iou,
        imgsz=image_size,
        device=device,
        verbose=False,
    )[0]
    inference_ms = (time.perf_counter() - started) * 1000

    counts = {name: 0 for name in CLASS_NAMES}
    rows: list[dict[str, object]] = []
    boxes = result.boxes
    if boxes is not None:
        class_ids = boxes.cls.int().cpu().tolist()
        scores = boxes.conf.cpu().tolist()
        coordinates = boxes.xyxy.cpu().tolist()
        for class_id, score, xyxy in zip(class_ids, scores, coordinates):
            class_name = str(result.names[int(class_id)])
            counts.setdefault(class_name, 0)
            counts[class_name] += 1
            rows.append(
                {
                    "Class": class_name,
                    "Confidence": round(float(score), 3),
                    "x1": round(float(xyxy[0]), 1),
                    "y1": round(float(xyxy[1]), 1),
                    "x2": round(float(xyxy[2]), 1),
                    "y2": round(float(xyxy[3]), 1),
                }
            )

    return DetectionOutput(result.plot(), counts, rows, inference_ms)


def render_counts(counts: dict[str, int], inference_ms: float | None = None) -> None:
    columns = st.columns(5)
    for column, class_name in zip(columns[:4], CLASS_NAMES):
        column.metric(CLASS_LABELS[class_name], counts.get(class_name, 0))
    speed_text = f"{inference_ms:.1f} ms" if inference_ms is not None else "Video"
    columns[4].metric("Latest inference", speed_text)


def render_alerts(
    counts: dict[str, int],
    pest_confirmed: bool | None = None,
    cleaning_confirmed: bool | None = None,
) -> None:
    status = status_from_counts(counts)
    pest_active = status["pest_alert"] if pest_confirmed is None else pest_confirmed
    cleaning_active = (
        status["cleaning_attention_required"]
        if cleaning_confirmed is None
        else cleaning_confirmed
    )

    if pest_active:
        st.error(
            "HIGH PRIORITY — Pest bird detected. Send a pest-removal alert; "
            "do not describe it as a table-clearing alert."
        )
    if cleaning_active:
        st.warning(
            f"CLEANING REQUIRED — {status['cleaning_items_detected']} tray/tableware "
            "item(s) detected."
        )
    if not pest_active and not cleaning_active:
        st.success("No pest or cleaning alert is currently active.")

    st.info(f"Available empty seats detected: {status['available_seats_detected']}")


def image_to_jpeg(image_rgb: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(
        ".jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92]
    )
    if not success:
        raise RuntimeError("Unable to encode the annotated image.")
    return encoded.tobytes()


def render_still_result(output: DetectionOutput, source_name: str) -> None:
    annotated_rgb = cv2.cvtColor(output.annotated_bgr, cv2.COLOR_BGR2RGB)
    st.image(annotated_rgb, caption=f"Detections — {source_name}", use_container_width=True)
    render_counts(output.counts, output.inference_ms)
    render_alerts(output.counts)

    if output.rows:
        st.subheader("Detection details")
        detections = pd.DataFrame(output.rows)
        st.dataframe(detections, use_container_width=True, hide_index=True)
        st.download_button(
            "Download detections as CSV",
            detections.to_csv(index=False).encode("utf-8"),
            file_name="hawker_detections.csv",
            mime="text/csv",
        )
    else:
        st.caption("No objects were detected above the selected confidence threshold.")

    st.download_button(
        "Download annotated image",
        image_to_jpeg(annotated_rgb),
        file_name="hawker_detection.jpg",
        mime="image/jpeg",
    )


def add_video_banner(
    frame: np.ndarray,
    pest_confirmed: bool,
    cleaning_confirmed: bool,
) -> np.ndarray:
    if pest_confirmed:
        text, colour = "HIGH: PEST BIRD ALERT", (0, 0, 255)
    elif cleaning_confirmed:
        text, colour = "MEDIUM: CLEANING REQUIRED", (0, 165, 255)
    else:
        text, colour = "NORMAL: NO CONFIRMED ALERT", (0, 170, 0)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 42), (25, 25, 25), -1)
    cv2.putText(frame, text, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.72, colour, 2)
    return frame


def process_video(
    model: YOLO,
    uploaded_file,
    confidence: float,
    iou: float,
    image_size: int,
    device: str | int,
    persistence: int,
    frame_step: int,
    maximum_seconds: int,
) -> tuple[bytes, dict[str, object]]:
    input_suffix = Path(uploaded_file.name).suffix.lower() or ".mp4"
    input_path: Path | None = None
    output_path: Path | None = None
    capture = None
    writer = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=input_suffix) as temporary_input:
            temporary_input.write(uploaded_file.getbuffer())
            input_path = Path(temporary_input.name)

        output_handle = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        output_path = Path(output_handle.name)
        output_handle.close()

        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise ValueError("The uploaded video could not be opened by OpenCV.")

        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 25.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        max_source_frames = min(total_frames, int(maximum_seconds * fps)) if total_frames else int(maximum_seconds * fps)

        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError("The annotated video writer could not be created.")

        progress = st.progress(0, text="Starting video analysis...")
        preview = st.empty()
        latest_counts = {name: 0 for name in CLASS_NAMES}
        total_counts: Counter[str] = Counter()
        processed_frames = 0
        written_frames = 0
        pest_streak = cleaning_streak = 0
        pest_confirmed_ever = cleaning_confirmed_ever = False
        alert_frames = 0
        current_annotated: np.ndarray | None = None

        while written_frames < max_source_frames:
            ok, frame = capture.read()
            if not ok:
                break

            if written_frames % frame_step == 0 or current_annotated is None:
                output = run_detection(model, frame, confidence, iou, image_size, device)
                current_annotated = output.annotated_bgr
                latest_counts = output.counts
                total_counts.update(output.counts)
                processed_frames += 1

                pest_now = latest_counts.get("pest_bird", 0) > 0
                cleaning_now = (
                    latest_counts.get("tray", 0) + latest_counts.get("uncleared_tableware", 0)
                ) > 0
                pest_streak = pest_streak + 1 if pest_now else 0
                cleaning_streak = cleaning_streak + 1 if cleaning_now else 0
                pest_confirmed = pest_streak >= persistence
                cleaning_confirmed = cleaning_streak >= persistence
                pest_confirmed_ever |= pest_confirmed
                cleaning_confirmed_ever |= cleaning_confirmed
                if pest_confirmed or cleaning_confirmed:
                    alert_frames += 1
            else:
                pest_confirmed = pest_streak >= persistence
                cleaning_confirmed = cleaning_streak >= persistence
                current_annotated = frame.copy()

            display_frame = add_video_banner(
                current_annotated.copy(), pest_confirmed, cleaning_confirmed
            )
            writer.write(display_frame)
            written_frames += 1

            if written_frames % max(1, int(fps)) == 0:
                progress_value = min(written_frames / max(max_source_frames, 1), 1.0)
                progress.progress(progress_value, text=f"Processed {written_frames / fps:.1f} seconds")
                preview.image(
                    cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB),
                    caption="Latest processed frame",
                    use_container_width=True,
                )

        progress.progress(1.0, text="Video analysis complete")
        capture.release()
        capture = None
        writer.release()
        writer = None
        video_bytes = output_path.read_bytes()
        summary = {
            "latest_counts": latest_counts,
            "total_detections_in_sampled_frames": dict(total_counts),
            "processed_inference_frames": processed_frames,
            "written_video_frames": written_frames,
            "source_fps": round(float(fps), 2),
            "pest_alert_confirmed": pest_confirmed_ever,
            "cleaning_alert_confirmed": cleaning_confirmed_ever,
            "alert_frames": alert_frames,
        }
        return video_bytes, summary
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        if input_path is not None:
            input_path.unlink(missing_ok=True)
        if output_path is not None:
            output_path.unlink(missing_ok=True)


def main() -> None:
    st.set_page_config(
        page_title="Smart Hawker Centre Monitor",
        page_icon="🍽️",
        layout="wide",
    )
    st.title("Smart Hawker Centre Monitor")
    st.caption(
        "YOLO11n detection prototype for empty seats, trays, uncleared tableware and pest birds"
    )

    if not MODEL_PATH.exists():
        st.error(f"Model checkpoint not found: {MODEL_PATH}")
        st.stop()

    device: str | int = 0 if torch.cuda.is_available() else "cpu"
    model = load_model(str(MODEL_PATH))

    with st.sidebar:
        st.header("Detection settings")
        confidence = st.slider("Confidence threshold", 0.05, 0.90, 0.30, 0.05)
        iou = st.slider("NMS IoU threshold", 0.10, 0.90, 0.70, 0.05)
        image_size = st.select_slider("Inference image size", [320, 480, 640, 800], value=640)
        st.divider()
        st.write(f"**Device:** {'GPU' if device == 0 else 'CPU'}")
        st.write("**Model:** YOLO11n, 50-epoch best checkpoint")
        st.caption("Alerts are displayed in the dashboard only; no external message is sent.")

    image_tab, camera_tab, video_tab = st.tabs(["Upload image", "Camera snapshot", "Upload video"])

    with image_tab:
        uploaded_image = st.file_uploader(
            "Choose an image", type=["jpg", "jpeg", "png", "bmp", "webp"], key="image_upload"
        )
        if uploaded_image is not None:
            image_rgb = np.array(Image.open(uploaded_image).convert("RGB"))
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            with st.spinner("Running object detection..."):
                output = run_detection(model, image_bgr, confidence, iou, image_size, device)
            render_still_result(output, uploaded_image.name)

    with camera_tab:
        camera_image = st.camera_input("Take a camera photograph")
        if camera_image is not None:
            image_rgb = np.array(Image.open(camera_image).convert("RGB"))
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            with st.spinner("Running object detection..."):
                output = run_detection(model, image_bgr, confidence, iou, image_size, device)
            render_still_result(output, "camera snapshot")

    with video_tab:
        st.caption(
            "Video alerts require detections in consecutive sampled frames. This reduces alarms "
            "caused by a single incorrect frame."
        )
        video_file = st.file_uploader(
            "Choose a video", type=["mp4", "avi", "mov", "mkv"], key="video_upload"
        )
        video_col1, video_col2, video_col3 = st.columns(3)
        persistence = video_col1.number_input(
            "Consecutive detections required", min_value=1, max_value=30, value=5
        )
        frame_step = video_col2.number_input(
            "Run inference every N frames", min_value=1, max_value=10, value=1
        )
        maximum_seconds = video_col3.number_input(
            "Maximum video seconds", min_value=5, max_value=300, value=60, step=5
        )

        if video_file is not None:
            st.video(video_file)
            if st.button("Analyse video", type="primary"):
                try:
                    video_bytes, summary = process_video(
                        model,
                        video_file,
                        confidence,
                        iou,
                        image_size,
                        device,
                        int(persistence),
                        int(frame_step),
                        int(maximum_seconds),
                    )
                except Exception as error:
                    st.exception(error)
                else:
                    st.subheader("Annotated video")
                    st.video(video_bytes)
                    st.download_button(
                        "Download annotated video",
                        video_bytes,
                        file_name="hawker_detection_video.mp4",
                        mime="video/mp4",
                    )
                    latest_counts = summary["latest_counts"]
                    render_counts(latest_counts)
                    render_alerts(
                        latest_counts,
                        pest_confirmed=bool(summary["pest_alert_confirmed"]),
                        cleaning_confirmed=bool(summary["cleaning_alert_confirmed"]),
                    )
                    st.subheader("Video summary")
                    st.json(summary)

    with st.expander("How the alert logic works"):
        st.markdown(
            """
            - `pest_bird > 0` activates a **high-priority pest alert**.
            - `tray + uncleared_tableware > 0` activates a separate **cleaning alert**.
            - `empty_seat` contributes only to the availability count.
            - A bird without trays therefore creates a pest alert but not a cleaning alert.
            - Video alerts use consecutive-frame confirmation; image alerts describe one image only.

            The detector supplies object predictions. These operational rules are a separate layer and
            can be changed without retraining YOLO11n.
            """
        )


if __name__ == "__main__":
    main()
