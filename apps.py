import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile

# ====== Page Config ======
st.set_page_config(page_title="🎨 Stylized Transformer", page_icon="🖌️", layout="wide")
st.markdown("<h1 style='text-align: center; color: #4B0082;'>🎨 Stylized Transformer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#555;'>Rotasi, Skala, Brightness/Contrast & Efek Cartoon/Sketch dengan kontrol penuh.</p>", unsafe_allow_html=True)

# ====== Fungsi bantu ======
def rotate_and_scale(image, angle, scale):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_scaled = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
    return rotated_scaled

def adjust_brightness_contrast(image, brightness=0, contrast=0):
    beta = brightness
    alpha = 1 + (contrast / 100)
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

def cartoonize(image, d=9, sigma_color=250, sigma_space=250, block_size=9, C=9):
    color = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, C
    )
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

def sketchify(image, blur_k=21, scale=256):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (blur_k, blur_k), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=scale)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)

# ====== Upload gambar ======
uploaded_file = st.file_uploader("📤 Upload Gambar (JPG/PNG)", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    image = cv2.imread(tfile.name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # ====== Sidebar dengan container & expander ======
    st.sidebar.header("⚙️ Kontrol Gambar")
    
    # Rotasi & Skala
    with st.sidebar.expander("🔄 Rotasi & Skala", expanded=True):
        angle_slider = st.slider("Rotasi (°)", -180, 180, 0)
        angle_input = st.number_input("Rotasi manual", -180, 180, value=angle_slider, step=1)
        angle = angle_input
        scale_slider = st.slider("Skala", 0.1, 2.0, 1.05, step=0.01)
        scale_input = st.number_input("Skala manual", 0.1, 2.0, value=scale_slider, step=0.01, format="%.2f")
        scale = scale_input

    # Brightness & Contrast
    with st.sidebar.expander("💡 Brightness & Contrast", expanded=True):
        brightness_slider = st.slider("Brightness", -100, 100, 0)
        brightness_input = st.number_input("Brightness manual", -100, 100, value=brightness_slider)
        brightness = brightness_input
        contrast_slider = st.slider("Contrast", -100, 100, 0)
        contrast_input = st.number_input("Contrast manual", -100, 100, value=contrast_slider)
        contrast = contrast_input

    # Efek
    with st.sidebar.expander("🎭 Efek Gaya", expanded=True):
        mode = st.radio("Pilih efek:", ["Cartoon", "Sketch"])
        if mode == "Cartoon":
            st.write("🔧 Parameter Cartoon")
            d = st.slider("Diameter (d)", 1, 20, 9)
            sigma_color = st.slider("Sigma Color", 10, 500, 250)
            sigma_space = st.slider("Sigma Space", 10, 500, 250)
            block_size = st.slider("Block Size (odd)", 3, 31, 9, step=2)
            C = st.slider("C (bias)", 1, 20, 9)
        else:
            st.write("🔧 Parameter Sketch")
            blur_k = st.slider("Gaussian Blur Kernel (odd)", 5, 51, 21, step=2)
            scale_val = st.slider("Scale", 128, 512, 256)

    # ====== Pemrosesan Gambar ======
    transformed = rotate_and_scale(image, angle, scale)
    adjusted = adjust_brightness_contrast(transformed, brightness, contrast)

    if mode == "Cartoon":
        result = cartoonize(adjusted, d, sigma_color, sigma_space, block_size, C)
    else:
        result = sketchify(adjusted, blur_k, scale_val)

    # ====== Preview Lebih Interaktif ======
    st.subheader("📸 Preview Gambar")
    col1, col2 = st.columns(2)
    with col1:
        st.image(adjusted, caption="Setelah Rotasi, Skala, Brightness & Contrast", use_container_width=True)
    with col2:
        st.image(result, caption=f"Efek {mode}", use_container_width=True)

    # ====== Download Button ======
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    st.download_button(
        "💾 Unduh Hasil",
        data=cv2.imencode('.jpg', result_bgr)[1].tobytes(),
        file_name=f"Stylized_{mode.lower()}_{angle}_{scale:.2f}.jpg",
        mime="image/jpeg"
    )
else:
    st.info("Silakan unggah gambar untuk mulai memproses.")
