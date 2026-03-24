import streamlit as st
import pandas as pd
import joblib
import json  

# 1. ตั้งค่าหน้าเพจ
st.set_page_config(page_title="Music Genre Predictor", page_icon="🎵", layout="wide")

# --- 🎨 เวทมนตร์ CSS สำหรับตกแต่งธีมเว็บให้เหมือนโปสเตอร์ ---
st.markdown("""
<style>
/* 1. เปลี่ยนสีพื้นหลังหลักเป็นการไล่สี ม่วง-ชมพู แบบโปสเตอร์ */
.stApp {
    background: linear-gradient(135deg, #a4508b 0%, #5f0a87 100%);
}

/* 2. เปลี่ยนสีแถบด้านข้าง (Sidebar) ให้โปร่งแสงนิดๆ ดูกลืนไปกับภาพรวม */
[data-testid="stSidebar"] {
    background-color: rgba(0, 0, 0, 0.25);
}

/* 3. บังคับให้ตัวหนังสือ หัวข้อ และข้อความส่วนใหญ่เป็นสีขาว จะได้ตัดกับพื้นหลัง */
h1, h2, h3, p, label, .stMarkdown {
    color: white !important;
}

/* 4. ปรับสีเส้นคั่น (Divider) */
hr {
    border-color: rgba(255, 255, 255, 0.3) !important;
}

/* 5. ตกแต่งกรอบแสดงผลลัพธ์ (Success/Info) ให้ดูพรีเมียมขึ้น */
div[data-testid="stAlert"] {
    background-color: rgba(255, 255, 255, 0.1) !important;
    border: 1px solid rgba(255, 255, 255, 0.3);
    color: white !important;
}
</style>
""", unsafe_allow_html=True)
# ---------------------------------------------------------

# 2. ฟังก์ชันโหลดโมเดลและ Metadata
@st.cache_resource
def load_models():
    pipeline = joblib.load("music_model_artifacts/music_pipeline.pkl")
    encoder = joblib.load("music_model_artifacts/label_encoder.pkl")

    with open("music_model_artifacts/metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return pipeline, encoder, metadata

pipeline, le, metadata = load_models()

# --- 🌟 แถบด้านข้าง (Sidebar) โชว์ข้อมูล Metadata ---
st.sidebar.title("ℹ️ ข้อมูลโมเดล AI")
st.sidebar.info(f"**โปรเจกต์:** {metadata.get('project', 'Music Genre')}")
st.sidebar.success(f"**ความแม่นยำ (Accuracy):** {metadata.get('accuracy', 0) * 100:.2f}%")
st.sidebar.caption("โมเดลที่ใช้: Gradient Boosting Classifier")
st.sidebar.divider()
st.sidebar.write("📌 **ปัจจัยที่ใช้ตัดสิน:**")
for feature in metadata.get('features', []):
    st.sidebar.write(f"- {feature}")

# 3. ส่วนหัวของเว็บไซต์
st.title("🎧 Music Genre Predictor")
st.markdown("ระบบ AI ทำนายแนวเพลงจากคุณลักษณะของเสียงดนตรี")
st.divider()

# 4. ส่วนรับข้อมูลจากผู้ใช้งาน
st.subheader("🎛️ ปรับแต่งคุณลักษณะของเพลง")

col1, col2 = st.columns(2)

with col1:
    popularity = st.slider("ความนิยม (Popularity)", 0, 100, 50)
    acousticness = st.slider("ความอะคูสติก (Acousticness)", 0.0, 1.0, 0.5)
    danceability = st.slider("จังหวะชวนเต้น (Danceability)", 0.0, 1.0, 0.5)
    energy = st.slider("พลังงาน (Energy)", 0.0, 1.0, 0.5)
    instrumentalness = st.slider("เสียงเครื่องดนตรี (Instrumentalness)", 0.0, 1.0, 0.1)

with col2:
    liveness = st.slider("ความรู้สึกเหมือนเล่นสด (Liveness)", 0.0, 1.0, 0.1)
    loudness = st.slider("ความดัง (Loudness dB)", -60.0, 0.0, -5.0)
    speechiness = st.slider("เสียงร้อง/เสียงพูด (Speechiness)", 0.0, 1.0, 0.05)
    tempo = st.slider("ความเร็วจังหวะ (Tempo BPM)", 50, 200, 120)
    valence = st.slider("ความอารมณ์ดี (Valence)", 0.0, 1.0, 0.5)

# 5. ปุ่มทำนายผล
if st.button("🚀 วิเคราะห์แนวเพลง", use_container_width=True):
    input_data = pd.DataFrame([{
        'popularity': popularity,
        'acousticness': acousticness,
        'danceability': danceability,
        'energy': energy,
        'instrumentalness': instrumentalness,
        'liveness': liveness,
        'loudness': loudness,
        'speechiness': speechiness,
        'tempo': tempo,
        'valence': valence
    }])

    pred_encoded = pipeline.predict(input_data)[0]
    pred_genre = le.inverse_transform([pred_encoded])[0]
    confidence = pipeline.predict_proba(input_data).max() * 100

    st.divider()
    st.success(f"### 🎵 AI ทำนายว่าเพลงนี้คือแนว: **{pred_genre.upper()}**")
    st.info(f"⭐ ความมั่นใจ: **{confidence:.2f}%**")
