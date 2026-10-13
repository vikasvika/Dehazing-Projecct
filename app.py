import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import os
import urllib.request
import sys
import types
import pandas as pd
from datetime import datetime
import time
import zipfile
import tempfile
from fpdf import FPDF

# --- 1. AI Architecture (DehazeNet) ---
class DehazeNet(nn.Module):
    def __init__(self):
        super(DehazeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 16, 5, padding=2)
        self.conv4 = nn.Conv2d(16, 16, 7, padding=3)
        self.conv5 = nn.Conv2d(64, 16, 3, padding=1)
        self.conv6 = nn.Conv2d(16, 1, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(7, stride=1, padding=3)
        self.brelu = nn.Hardtanh(0.0, 1.0)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2_1 = self.conv2(x1)
        x2_2 = self.conv3(x1)
        x2_3 = self.conv4(x1)
        x2 = torch.cat((x2_1, x2_2, x2_3, x1), 1) 
        x3 = self.relu(self.conv5(x2))
        x4 = self.pool(x3)
        transmission_map = self.brelu(self.conv6(x4))
        return transmission_map

fake_model = types.ModuleType('model')
fake_model.DehazeNet = DehazeNet
sys.modules['model'] = fake_model

@st.cache_resource
def load_ai_model(weights_path):
    model = DehazeNet()
    if not os.path.exists(weights_path):
        url = "https://github.com/thuBingo/DehazeNet_Pytorch/raw/master/defog4_noaug.pth"
        try:
            urllib.request.urlretrieve(url, weights_path)
        except Exception:
            return model 
    try:
        loaded_data = torch.load(weights_path, map_location=torch.device('cpu'), weights_only=False)
        if hasattr(loaded_data, 'state_dict'):
            model.load_state_dict(loaded_data.state_dict())
        else:
            model.load_state_dict(loaded_data)
    except Exception:
        pass
    model.eval()
    return model

# --- 2. Mathematical Core ---
def guided_filter(I, p, r, eps):
    mean_I = cv2.boxFilter(I, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = cv2.boxFilter(I * I, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))
    return mean_a * I + mean_b

def get_dark_channel(image, patch_size=15):
    b, g, r = cv2.split(image)
    min_img = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    return cv2.erode(min_img, kernel)

def recover_image(img_array, engine, omega=0.95, model=None):
    img_float = img_array.astype(np.float64)
    dark_channel = get_dark_channel(img_float)
    
    h, w = img_float.shape[:2]
    image_size = h * w
    num_pixels = int(max(image_size * 0.001, 1))
    indices = np.argsort(dark_channel.reshape(image_size))[-num_pixels:]
    atmospheric_light = np.mean(img_float.reshape(image_size, 3)[indices], axis=0)
    
    if engine == "AI Deep Learning (DehazeNet)" and model is not None:
        transform = transforms.Compose([transforms.ToTensor()])
        hazy_tensor = transform(img_array).unsqueeze(0).float()
        with torch.no_grad():
            t_tensor = model(hazy_tensor)
        t_map = t_tensor.squeeze().numpy()
        if np.std(t_map) < 0.01: 
            norm_image = img_float / atmospheric_light
            raw_t = 1 - omega * get_dark_channel(norm_image, 15)
            gray_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
            transmission = guided_filter(gray_image, raw_t, r=40, eps=1e-3)
        else:
            transmission = cv2.resize(t_map, (w, h))
    else:
        norm_image = img_float / atmospheric_light
        raw_transmission = 1 - omega * get_dark_channel(norm_image, 15)
        gray_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
        transmission = guided_filter(gray_image, raw_transmission, r=40, eps=1e-3)
        
    transmission = np.maximum(transmission, 0.1)
    transmission_map = np.empty(img_float.shape)
    for i in range(3):
        transmission_map[:, :, i] = transmission
        
    recovered_image = np.clip((img_float - atmospheric_light) / transmission_map + atmospheric_light, 0, 255).astype(np.uint8)
    gaussian = cv2.GaussianBlur(recovered_image, (0, 0), 2.0)
    final_img = cv2.addWeighted(recovered_image, 1.5, gaussian, -0.5, 0)
    
    return final_img

# --- 3. Database, Metrics & PDF Generation ---
DB_FILE = "processing_history.csv"

def save_to_database(filename, method, exec_time, clarity):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_data = pd.DataFrame([{"Date": now, "File": filename, "Method": method, "Time (s)": round(exec_time, 2), "Clarity Gain": f"+{clarity}%"}])
    if os.path.exists(DB_FILE):
        df = pd.read_csv(DB_FILE)
        df = pd.concat([df, new_data], ignore_index=True)
    else:
        df = new_data
    df.to_csv(DB_FILE, index=False)
    
    if "session_data" not in st.session_state:
        st.session_state.session_data = []
    st.session_state.session_data.append({
        "File": filename,
        "Method": method,
        "Time (s)": round(exec_time, 2),
        "Clarity Numeric": float(clarity)
    })

def calculate_clarity_gain(hazy_img, clean_img):
    hazy_std = np.std(cv2.cvtColor(hazy_img, cv2.COLOR_RGB2GRAY))
    clean_std = np.std(cv2.cvtColor(clean_img, cv2.COLOR_RGB2GRAY))
    if hazy_std == 0: return 0
    gain = ((clean_std - hazy_std) / hazy_std) * 100
    return round(max(0, gain), 1)

def generate_pdf_report(orig_pil, clean_pil, filename, method, exec_time, clarity):
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "VisionClear AI - Project Case Report", ln=True, align='C')
    pdf.ln(5)
    
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 8, f"File Name: {filename}", ln=True)
    pdf.cell(0, 8, f"Processing Engine: {method}", ln=True)
    pdf.cell(0, 8, f"Execution Time: {exec_time:.2f} seconds", ln=True)
    pdf.cell(0, 8, f"Clarity Improvement: +{clarity}%", ln=True)
    pdf.cell(0, 8, f"Date Processed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(10)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as f_orig, \
         tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as f_clean:
         
        orig_pil.save(f_orig.name, format="JPEG")
        clean_pil.save(f_clean.name, format="JPEG")
        
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Visual Analysis:", ln=True)
        
        y_pos = pdf.get_y()
        pdf.image(f_orig.name, x=10, y=y_pos, w=90)
        pdf.image(f_clean.name, x=110, y=y_pos, w=90)
        
        pdf.set_y(y_pos + 70)
        pdf.set_font("Arial", 'I', 10)
        pdf.cell(90, 10, "Original Hazy Image", align='C')
        pdf.cell(90, 10, "De-hazed Output", align='C', ln=True)
        
    pdf_out = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    pdf.output(pdf_out.name)
    
    with open(pdf_out.name, "rb") as f:
        pdf_bytes = f.read()
        
    return pdf_bytes

# --- 4. Dashboard UI ---
st.set_page_config(layout="wide", page_title="VisionClear AI Dashboard")

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = "initial_key"
if "session_count" not in st.session_state:
    st.session_state.session_count = 0
if "session_data" not in st.session_state:
    st.session_state.session_data = []

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103322.png", width=60)
    st.title("Control Panel")
    st.markdown("---")
    
    method = st.radio("⚙️ Processing Engine:", ["Advanced Baseline (Guided DCP)", "AI Deep Learning (DehazeNet)"])
    omega = 0.95
    if method == "Advanced Baseline (Guided DCP)":
        omega = st.slider("🎚️ De-hazing Strength", 0.0, 1.0, 0.95, 0.05)
    st.markdown("---")
    
    if st.button("🔄 Reset Workspace", use_container_width=True):
        st.session_state.uploader_key = str(time.time())
        st.session_state.session_count = 0
        st.session_state.session_data = []
        st.rerun()
        
    st.markdown("---")
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "rb") as file:
            st.download_button(label="📥 Export Report (CSV)", data=file, file_name="Dehaze_Report.csv", mime="text/csv")

st.title("☁️ AI Image & Video De-hazing System")
st.markdown("**Center for Emerging Computing Technologies - Major Project**")

col_a, col_b, col_c = st.columns(3)
col_a.metric("Images Processed (Current Session)", st.session_state.session_count)
col_b.metric("Active Engine", "DehazeNet CNN" if "AI" in method else "Guided DCP Math")
col_c.metric("System Status", "Online & Ready")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["📸 Single Image & Webcam", "🗂️ Batch Processing", "🎥 Video Processing", "📊 Live Analytics"])

# TAB 1: Single Image & Webcam
with tab1:
    input_source = st.radio("Select Input Source:", ["Upload File", "Use Webcam"], horizontal=True)
    uploaded_file = None
    if input_source == "Upload File":
        uploaded_file = st.file_uploader("📂 Upload a hazy image...", type=["jpg", "jpeg", "png"], key=st.session_state.uploader_key)
    else:
        uploaded_file = st.camera_input("📷 Take a picture")

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)
        start_time = time.time()
        with st.spinner(f"Executing {method} algorithms..."):
            ai_model = load_ai_model("dehazenet.pth") if "AI" in method else None
            recovered_img = recover_image(img_array, method, omega, ai_model)
            exec_time = time.time() - start_time
            clarity_gain = calculate_clarity_gain(img_array, recovered_img)
            
            save_to_database(uploaded_file.name if hasattr(uploaded_file, 'name') else "webcam_capture", method, exec_time, clarity_gain)
            st.session_state.session_count += 1
        
        st.success(f"✅ Processing Complete in {exec_time:.2f} seconds | Visibility Improved by {clarity_gain}%")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Input")
            st.image(image, use_container_width=True)
        with col2:
            st.subheader("De-hazed Output")
            st.image(recovered_img, use_container_width=True)
                
        result_pil = Image.fromarray(recovered_img)
        buf = io.BytesIO()
        result_pil.save(buf, format="JPEG")
        
        st.markdown("---")
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            st.download_button(label="💾 Save Clean Image", data=buf.getvalue(), file_name="clean_image.jpg", mime="image/jpeg", use_container_width=True)
        with col_btn2:
            filename = uploaded_file.name if hasattr(uploaded_file, 'name') else "webcam_capture"
            pdf_bytes = generate_pdf_report(image, result_pil, filename, method, exec_time, clarity_gain)
            st.download_button(label="📄 Download PDF Case Report", data=pdf_bytes, file_name=f"Report_{filename}.pdf", mime="application/pdf", use_container_width=True)

# TAB 2: Batch Processing
with tab2:
    st.subheader("Process multiple files at once")
    batch_files = st.file_uploader("Upload multiple hazy images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if batch_files and st.button("Start Batch Processing"):
        progress_bar = st.progress(0)
        zip_buffer = io.BytesIO()
        ai_model = load_ai_model("dehazenet.pth") if "AI" in method else None
        
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            for i, file in enumerate(batch_files):
                img_start_time = time.time()
                img = Image.open(file).convert("RGB")
                img_arr = np.array(img)
                rec_img = recover_image(img_arr, method, omega, ai_model)
                
                img_exec = time.time() - img_start_time
                clarity = calculate_clarity_gain(img_arr, rec_img)
                save_to_database(file.name, method, img_exec, clarity)
                
                rec_pil = Image.fromarray(rec_img)
                img_byte_arr = io.BytesIO()
                rec_pil.save(img_byte_arr, format='JPEG')
                zf.writestr(f"clean_{file.name}", img_byte_arr.getvalue())
                
                progress_bar.progress((i + 1) / len(batch_files))
                st.session_state.session_count += 1
                
        st.success("✅ Batch processing complete! View the Analytics tab to see the breakdown.")
        st.download_button(label="📥 Download All Clean Images (ZIP)", data=zip_buffer.getvalue(), file_name="Dehazed_Batch.zip", mime="application/zip")

# TAB 3: Video Processing
with tab3:
    st.subheader("Process Hazy Video (Frame-by-Frame)")
    st.warning("⚠️ Video processing takes significant computational power.")
    video_file = st.file_uploader("Upload a hazy video", type=["mp4", "mov", "avi"])
    
    if video_file and st.button("Start Video Processing"):
        ai_model = load_ai_model("dehazenet.pth") if "AI" in method else None
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_file.name, fourcc, fps, (width, height))
        
        progress_text = st.empty()
        video_progress = st.progress(0)
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            clean_frame_rgb = recover_image(frame_rgb, method, omega, ai_model)
            clean_frame_bgr = cv2.cvtColor(clean_frame_rgb, cv2.COLOR_RGB2BGR)
            out.write(clean_frame_bgr)
            frame_count += 1
            video_progress.progress(min(frame_count / total_frames, 1.0))
            progress_text.text(f"Processing frame {frame_count} of {total_frames}...")
            
        cap.release()
        out.release()
        st.success("✅ Video processing complete!")
        with open(out_file.name, 'rb') as vf:
            video_bytes = vf.read()
            st.video(video_bytes)
            st.download_button(label="💾 Download Clean Video", data=video_bytes, file_name="dehazed_video.mp4", mime="video/mp4")

# TAB 4: Live Analytics Dashboard
with tab4:
    st.subheader("📊 Session Analytics (Per Image)")
    if "session_data" in st.session_state and len(st.session_state.session_data) > 0:
        df_analytics = pd.DataFrame(st.session_state.session_data)
        
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown("**Clarity Gain per Image (%)**")
            st.bar_chart(df_analytics.set_index("File")["Clarity Numeric"])
            
        with col_g2:
            st.markdown("**Processing Time per Image (Seconds)**")
            st.bar_chart(df_analytics.set_index("File")["Time (s)"])
            
        st.markdown("**Overall Session Trend (%)**")
        st.line_chart(df_analytics["Clarity Numeric"])
    else:
        st.info("Process an image to see live analytics! (Data will clear when you hit Reset Workspace)")

st.markdown("---")

with st.expander("🕒 Watch History (Persistent Database)"):
    if os.path.exists(DB_FILE):
        col_db1, col_db2 = st.columns([4, 1])
        with col_db2:
            if st.button("🗑️ Clear All History", use_container_width=True):
                os.remove(DB_FILE)
                st.rerun()
        with col_db1:
            st.info("💡 To delete a single entry: Click the checkbox next to a row and press the 'Delete' icon at the top right of the table.")
            history_df = pd.read_csv(DB_FILE).sort_values(by="Date", ascending=False)
            edited_df = st.data_editor(
                history_df, 
                use_container_width=True, 
                num_rows="dynamic",
                hide_index=False
            )
            if len(edited_df) != len(history_df):
                edited_df.to_csv(DB_FILE, index=False)
                st.rerun()
    else:
        st.write("No images processed yet in the database.")