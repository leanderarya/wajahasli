import os
import cv2
import tempfile
import numpy as np
import streamlit as st
from ultralytics import YOLO

# =========================================================
# 1. KONFIGURASI SISTEM
# =========================================================
APP_NAME = "WajahAsli"
MODEL_PATH = "best.pt"
SYSTEM_CONFIDENCE_THRESHOLD = 0.40

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# 2. LOAD MODEL
# =========================================================
@st.cache_resource
def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Gagal memuat model '{MODEL_PATH}': {e}")
    st.stop()

# =========================================================
# 3. UTILITAS
# =========================================================
def format_label(raw_label: str) -> str:
    mapping = {
        "wajah_asli": "WAJAH ASLI",
        "spoof_screen": "SPOOF SCREEN",
        "spoof_print": "SPOOF PRINT",
    }
    return mapping.get(raw_label, raw_label.replace("_", " ").upper())


def get_status_from_label(label: str):
    if label == "wajah_asli":
        return "safe", "WAJAH ASLI TERDETEKSI", (0, 255, 0)
    if label in {"spoof_screen", "spoof_print"}:
        return "danger", "SPOOFING TERDETEKSI", (0, 0, 255)
    return "idle", "MENUNGGU...", (255, 255, 0)


def draw_box_with_label(img, x1, y1, x2, y2, label_text, color):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    (text_w, text_h), _ = cv2.getTextSize(
        label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
    )
    box_y1 = max(0, y1 - 28)
    box_y2 = y1
    box_x2 = x1 + text_w + 8

    cv2.rectangle(img, (x1, box_y1), (box_x2, box_y2), color, -1)
    cv2.putText(
        img,
        label_text,
        (x1 + 4, y1 - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
    )


def process_frame(model: YOLO, frame: np.ndarray, conf_thresh: float):
    img_display = frame.copy()

    results = model(frame, verbose=False, conf=conf_thresh)

    final_status = "idle"
    final_message = "MENUNGGU..."
    max_conf = 0.0

    if results:
        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls]

                if conf > max_conf:
                    max_conf = conf
                    final_status, final_message, _ = get_status_from_label(label)

                label_text = f"{format_label(label)} ({conf:.2f})"
                _, _, color = get_status_from_label(label)
                draw_box_with_label(img_display, x1, y1, x2, y2, label_text, color)

    return img_display, final_status, final_message, max_conf


def show_result_image(result_frame, status_type, status_msg, max_conf, download_name):
    col1, col2 = st.columns([2, 1.2])

    with col1:
        st.image(
            result_frame,
            channels="BGR",
            width=500,
            clamp=True,
        )

    with col2:
        st.markdown("### Hasil Keputusan")

        if status_type == "safe":
            st.success(status_msg)
        elif status_type == "danger":
            st.error(status_msg)
        else:
            st.info(status_msg)

        st.metric("Confidence Maksimum", f"{max_conf:.2%}")

        success, buffer = cv2.imencode(".jpg", result_frame)
        if success:
            st.download_button(
                "📥 Unduh Hasil",
                data=buffer.tobytes(),
                file_name=download_name,
                mime="image/jpeg",
                use_container_width=True,
            )

# =========================================================
# 4. SIDEBAR
# =========================================================
with st.sidebar:
    st.title("Status Sistem")

    st.markdown("### Spesifikasi")
    st.markdown("**Model Engine:** YOLOv8 Nano")
    st.markdown(f"**Confidence Threshold:** {SYSTEM_CONFIDENCE_THRESHOLD}")
    st.caption("Threshold dikunci pada backend sesuai hasil analisis penelitian.")

    st.divider()

    st.markdown("### Kategori Deteksi")
    st.markdown(
        """
        - 🟢 **Wajah Asli**
        - 🔴 **Spoof Screen**
        - 🔴 **Spoof Print**
        """
    )

    st.divider()

    st.markdown("### Panduan")
    st.markdown(
        """
        - **Kamera**: ambil satu foto langsung dari perangkat
        - **Upload Foto**: unggah citra JPG, JPEG, atau PNG
        - **Upload Video**: unggah rekaman MP4 atau AVI
        """
    )

# =========================================================
# 5. HEADER
# =========================================================
st.title("🛡️ WajahAsli")
st.caption("Deteksi Keaslian Wajah Berbasis Deep Learning (YOLOv8)")
st.divider()

# =========================================================
# 6. TABS
# =========================================================
tab_cam, tab_img, tab_vid = st.tabs(
    ["📷 Kamera", "🖼️ Upload Foto", "📹 Upload Video"]
)

# =========================================================
# 7. TAB KAMERA
# =========================================================
with tab_cam:
    st.markdown("### Ambil Foto dari Kamera")
    st.info(
        "Ambil satu foto lalu sistem akan mendeteksi apakah wajah termasuk wajah asli atau spoof."
    )

    col_left, col_center, col_right = st.columns([1, 2, 1])

    with col_center:
        camera_photo = st.camera_input("Ambil gambar")

    if camera_photo is not None:
        file_bytes = np.asarray(bytearray(camera_photo.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if frame is None:
            st.error("Gambar dari kamera tidak valid atau tidak dapat dibaca.")
            st.stop()

        with st.spinner("Sedang menganalisis gambar dari kamera..."):
            result_frame, status_type, status_msg, max_conf = process_frame(
                model, frame, SYSTEM_CONFIDENCE_THRESHOLD
            )

        show_result_image(
            result_frame=result_frame,
            status_type=status_type,
            status_msg=status_msg,
            max_conf=max_conf,
            download_name="hasil_kamera.jpg",
        )

# =========================================================
# 8. TAB FOTO
# =========================================================
with tab_img:
    st.markdown("### Analisis Citra Tunggal")
    uploaded_img = st.file_uploader(
        "Unggah citra wajah", type=["jpg", "jpeg", "png"]
    )

    if uploaded_img is not None:
        file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if frame is None:
            st.error("File gambar tidak valid atau tidak dapat dibaca.")
            st.stop()

        with st.spinner("Sedang menganalisis citra..."):
            result_frame, status_type, status_msg, max_conf = process_frame(
                model, frame, SYSTEM_CONFIDENCE_THRESHOLD
            )

        show_result_image(
            result_frame=result_frame,
            status_type=status_type,
            status_msg=status_msg,
            max_conf=max_conf,
            download_name=f"hasil_{uploaded_img.name}",
        )

# =========================================================
# 9. TAB VIDEO
# =========================================================
with tab_vid:
    st.markdown("### Analisis Rekaman Video")
    uploaded_vid = st.file_uploader("Unggah video", type=["mp4", "avi"])

    if uploaded_vid is not None:
        input_suffix = os.path.splitext(uploaded_vid.name)[1].lower()
        if input_suffix not in [".mp4", ".avi"]:
            input_suffix = ".mp4"

        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=input_suffix)
        temp_input.write(uploaded_vid.read())
        temp_input.close()

        cap = cv2.VideoCapture(temp_input.name)

        if not cap.isOpened():
            os.unlink(temp_input.name)
            st.error("Video tidak dapat dibaca.")
            st.stop()

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_input = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps_input <= 0:
            fps_input = 25

        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(temp_output, fourcc, fps_input, (width, height))

        st.markdown("**Progres Pemrosesan:**")
        progress_bar = st.progress(0)
        status_placeholder = st.empty()

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            result_frame, _, _, _ = process_frame(
                model, frame, SYSTEM_CONFIDENCE_THRESHOLD
            )
            out.write(result_frame)

            frame_idx += 1
            if total_frames > 0 and frame_idx % 5 == 0:
                progress = min(frame_idx / total_frames, 1.0)
                progress_bar.progress(progress)
                status_placeholder.text(
                    f"Menganalisis frame {frame_idx} dari {total_frames}..."
                )

        cap.release()
        out.release()

        progress_bar.progress(1.0)
        status_placeholder.success("✅ Pemrosesan video selesai.")

        with open(temp_output, "rb") as f:
            video_bytes = f.read()

        st.video(video_bytes)

        st.download_button(
            label="📥 Unduh Video Hasil",
            data=video_bytes,
            file_name=f"hasil_{uploaded_vid.name}",
            mime="video/mp4",
            use_container_width=True,
        )

        os.unlink(temp_input.name)
        os.unlink(temp_output)

st.divider()
st.caption("© 2026 Arya Ajisadda Haryanto | Universitas Diponegoro")