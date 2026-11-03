
from pathlib import Path
import sys
import numpy as np
import streamlit as st
from PIL import Image, ImageStat

# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent
APP_DIR = PROJECT_DIR / "app"

sys.path.insert(0, str(APP_DIR))

from models.model_loader import (
    APP_CONFIG,
    load_crop_model,
    get_crop_classes
)

from models.prediction import predict_disease

from knowledge_base.knowledge_loader import (
    get_disease_information
)


# ==========================================
# 2. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AgriDoctor AI | Smart Farming",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# 3. CUSTOM CSS
# ==========================================

st.markdown("""
<style>

@import url(
'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap'
);

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #f5f8f3;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}

.hero {
    background: linear-gradient(
        120deg,
        #123b29 0%,
        #176b45 55%,
        #36966a 100%
    );
    padding: 38px;
    border-radius: 24px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 10px 30px rgba(25, 75, 48, 0.13);
}

.hero h1 {
    font-size: 42px;
    font-weight: 800;
    color: white;
    margin-bottom: 8px;
}

.hero p {
    font-size: 16px;
    color: #e2f3e7;
}

.eyebrow {
    color: #c4f0d0;
    font-size: 12px;
    letter-spacing: 2px;
    font-weight: 700;
}

.section-title {
    font-size: 23px;
    font-weight: 800;
    color: #183c2b;
    margin-top: 22px;
    margin-bottom: 12px;
}

.info-card {
    background: white;
    border: 1px solid #e4eee5;
    border-radius: 18px;
    padding: 20px;
    margin-bottom: 12px;
    box-shadow: 0 4px 15px rgba(30, 65, 40, 0.035);
}

.info-card h3 {
    color: #24623d;
    font-size: 17px;
}

.info-card p {
    color: #5b6d60;
    font-size: 14px;
}

.metric-card {
    background: white;
    padding: 20px;
    border: 1px solid #e1ebdf;
    border-radius: 17px;
    text-align: center;
}

.metric-label {
    color: #758477;
    font-size: 13px;
}

.metric-value {
    color: #176b45;
    font-size: 26px;
    font-weight: 800;
}

.result-card {
    background: linear-gradient(135deg, #e8f5e9, #f8fcf6);
    border: 1px solid #cde7d0;
    padding: 25px;
    border-radius: 20px;
    margin-top: 18px;
}

.result-card h2 {
    color: #175b36;
}

.stButton > button {
    background: linear-gradient(90deg, #176b45, #278b5b);
    color: white;
    border: none;
    border-radius: 12px;
    height: 48px;
    font-weight: 700;
    transition: 0.2s ease;
}

.stButton > button:hover {
    background: #104b30;
    color: white;
    border: none;
    transform: translateY(-2px);
}

[data-testid="stFileUploader"] {
    background: white;
    border: 2px dashed #9ac9a4;
    border-radius: 17px;
    padding: 15px;
}

section[data-testid="stSidebar"] {
    background: #edf5ed;
    border-right: 1px solid #dceadd;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 4. LANGUAGE TEXT
# ==========================================

TEXT = {
    "en": {
        "subtitle": "Your intelligent companion for healthier crops.",
        "select_crop": "Select your crop",
        "select_language": "Language",
        "upload": "Upload a plant leaf image",
        "diagnose": "Diagnose My Plant",
        "preview": "Image Preview",
        "result": "AI Diagnosis Result",
        "symptoms": "Symptoms",
        "causes": "Causes",
        "prevention": "Prevention",
        "medicine": "Medicine Guidance",
        "recovery": "Recovery",
        "uncertain": "Uncertain prediction",
        "ambiguous": "Ambiguous prediction",
        "expert": "Please consult a qualified agricultural expert.",
        "score": "Model score",
        "healthy": "Continue monitoring your crop.",
        "footer": "AI-assisted information. Not a substitute for expert diagnosis."
    },
    "ta": {
        "subtitle": "ஆரோக்கியமான பயிர்களுக்கான உங்கள் அறிவார்ந்த துணை.",
        "select_crop": "உங்கள் பயிரைத் தேர்ந்தெடுக்கவும்",
        "select_language": "மொழி",
        "upload": "தாவர இலைப் படத்தைப் பதிவேற்றவும்",
        "diagnose": "பயிரைக் கண்டறியவும்",
        "preview": "படத்தின் முன்னோட்டம்",
        "result": "AI நோய் கண்டறிதல் முடிவு",
        "symptoms": "அறிகுறிகள்",
        "causes": "காரணங்கள்",
        "prevention": "தடுப்பு முறைகள்",
        "medicine": "மருந்து வழிகாட்டுதல்",
        "recovery": "மீட்பு வழிகாட்டுதல்",
        "uncertain": "உறுதியற்ற கணிப்பு",
        "ambiguous": "தெளிவற்ற கணிப்பு",
        "expert": "தகுதியான வேளாண் நிபுணரை அணுகவும்.",
        "score": "மாடல் மதிப்பெண்",
        "healthy": "உங்கள் பயிரைத் தொடர்ந்து கண்காணிக்கவும்.",
        "footer": "AI உதவியுடன் வழங்கப்படும் தகவல். நிபுணர் ஆலோசனைக்கு மாற்றாகாது."
    }
}


# ==========================================
# 5. CROP CONFIGURATION
# ==========================================

CROP_OPTIONS = {
    "Tomato, Potato & Pepper": "plantvillage",
    "Rice / நெல்": "rice",
    "Sugarcane / கரும்பு": "sugarcane"
}


# ==========================================
# 6. IMAGE QUALITY CHECK
# ==========================================

def check_image_quality(image):

    image = image.convert("RGB")

    width, height = image.size

    if width < 100 or height < 100:
        return False, "Image resolution is too low."

    brightness = float(np.mean(np.array(image)))

    contrast = float(
        ImageStat.Stat(image.convert("L")).stddev[0]
    )

    if brightness < 30:
        return False, "Image is too dark."

    if brightness > 245:
        return False, "Image is too bright."

    if contrast < 10:
        return False, "Image contrast is too low."

    return True, "Image quality accepted."


# ==========================================
# 7. MODEL LOADING
# ==========================================

@st.cache_resource
def get_cached_model(crop_id):
    return load_crop_model(crop_id)


# ==========================================
# 8. MAIN DIAGNOSIS FUNCTION
# ==========================================

def diagnose(image, crop_id, language):

    quality_ok, quality_message = check_image_quality(image)

    if not quality_ok:
        return {
            "status": "image_rejected",
            "message": quality_message
        }

    model = get_cached_model(crop_id)

    class_names = get_crop_classes(crop_id)

    prediction = predict_disease(
        image=image,
        model=model,
        class_names=class_names
    )

    scores = np.asarray(prediction["all_scores"])

    top_indices = np.argsort(scores)[::-1][:3]

    top_score = float(scores[top_indices[0]])

    second_score = (
        float(scores[top_indices[1]])
        if len(top_indices) > 1 else 0.0
    )

    score_gap = top_score - second_score

    if top_score < 0.70:
        return {
            "status": "uncertain",
            "prediction": prediction,
            "top_indices": top_indices
        }

    if score_gap < 0.10:
        return {
            "status": "ambiguous",
            "prediction": prediction,
            "top_indices": top_indices
        }

    information = get_disease_information(
        crop_id=crop_id,
        disease_name=prediction["disease"],
        language=language
    )

    return {
        "status": "candidate_prediction",
        "prediction": prediction,
        "information": information,
        "top_indices": top_indices
    }


# ==========================================
# 9. SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown("# 🌱 AgriDoctor AI")
    st.caption("SMART FARMING PLATFORM")

    st.divider()

    language_choice = st.radio(
        "🌐 Language / மொழி",
        ["English", "தமிழ்"],
        horizontal=True
    )

    language = (
        "en" if language_choice == "English" else "ta"
    )

    t = TEXT[language]

    st.markdown("### 🌾 Crop Selection")

    selected_crop = st.selectbox(
        t["select_crop"],
        list(CROP_OPTIONS.keys())
    )

    crop_id = CROP_OPTIONS[selected_crop]

    st.divider()

    st.markdown("### 🧠 AI Models")

    st.success("PlantVillage · 15 classes")
    st.success("Rice · 4 classes")
    st.success("Sugarcane · 11 classes")

    st.divider()

    st.caption("AgriDoctor AI | Version 1.0")


# ==========================================
# 10. HERO SECTION
# ==========================================

st.markdown("")

with st.container():
    st.markdown("### 🌱 AgriDoctor AI")
    st.caption("INTELLIGENT AGRICULTURE • AI POWERED")
    st.write(
        "Your intelligent companion for plant disease diagnosis "
        "and smarter farming decisions."
    )


st.markdown(
    f'<div class="section-title">{t["upload"]}</div>',
    unsafe_allow_html=True
)


# ==========================================
# 11. IMAGE UPLOAD SECTION
# ==========================================

left, right = st.columns([1.1, 0.9], gap="large")

with left:

    uploaded_file = st.file_uploader(
        "Choose your plant image",
        type=["jpg", "jpeg", "png"],
        key="plant_upload"
    )

    st.caption(
        "For best results, photograph a clear leaf "
        "in natural lighting."
    )

with right:

    if uploaded_file is not None:

        uploaded_image = Image.open(uploaded_file).convert("RGB")

        st.image(
            uploaded_image,
            caption=t["preview"],
            use_container_width=True
        )

    else:

        st.markdown("""
        <div class="info-card">
            <h3>📸 Clear leaf image</h3>
            <p>
                Upload a well-lit image showing the affected
                plant leaf clearly.
            </p>
            <p>Supported formats: JPG, JPEG, PNG</p>
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# 12. DIAGNOSIS BUTTON
# ==========================================

st.markdown("")

if uploaded_file is not None:

    if st.button(
        f"🌿 {t['diagnose']}",
        use_container_width=True
    ):

        try:

            with st.spinner("Analyzing your plant image..."):

                final_result = diagnose(
                    uploaded_image,
                    crop_id,
                    language
                )

            if final_result["status"] == "image_rejected":

                st.error(
                    f"Image rejected: {final_result['message']}"
                )

            elif final_result["status"] in [
                "uncertain", "ambiguous"
            ]:

                message = (
                    t["uncertain"]
                    if final_result["status"] == "uncertain"
                    else t["ambiguous"]
                )

                st.warning(message)

                st.info(t["expert"])

                st.write(
                    "Please upload another clear image "
                    "or seek expert confirmation."
                )

            else:

                prediction = final_result["prediction"]

                information = final_result["information"]

                score = prediction["model_score"]

                st.markdown(
                    f'<div class="section-title">'
                    f'🌿 {t["result"]}</div>',
                    unsafe_allow_html=True
                )

                st.markdown(f"""
                <div class="result-card">
                    <h2>{prediction["disease"]}</h2>
                    <p>{t["score"]}: {score * 100:.2f}%</p>
                    <p>
                        Candidate AI prediction — confirm with
                        a qualified agricultural expert.
                    </p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("### 📊 Top Predictions")

                for index in final_result["top_indices"]:

                    class_name = get_crop_classes(crop_id)[index]

                    class_score = prediction["all_scores"][index]

                    st.write(
                        f"**{class_name}** — "
                        f"{class_score * 100:.2f}%"
                    )

                    st.progress(
                        min(max(float(class_score), 0.0), 1.0)
                    )

                st.divider()

                # Agricultural information tabs

                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    f"🔍 {t['symptoms']}",
                    f"🧬 {t['causes']}",
                    f"🛡️ {t['prevention']}",
                    f"💊 {t['medicine']}",
                    f"🌱 {t['recovery']}"
                ])

                tab_sections = [
                    (tab1, "symptoms"),
                    (tab2, "causes"),
                    (tab3, "prevention"),
                    (tab4, "medicine"),
                    (tab5, "recovery")
                ]

                for tab, section in tab_sections:

                    with tab:

                        for item in information[section]:
                            st.markdown(f"- {item}")

        except Exception as error:

            st.error(f"Diagnosis error: {error}")


# ==========================================
# 13. FOOTER
# ==========================================

st.divider()


with st.container():
    st.markdown("### 🌱 AgriDoctor AI")
    st.write("Smart Plant Diagnosis • Bilingual Farming Guidance")
    st.caption(
        "AI-assisted information. Always confirm important "
        "crop-management decisions with qualified local experts."
    )
