import streamlit as st
import pandas as pd
import pickle
import base64

# -------------------- BASE64 BACKGROUND --------------------
def get_base64_bg_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

bg_image = get_base64_bg_image("images.jpg")

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Canberra Weather AI",
    page_icon="🌤️",
    layout="wide"
)

# -------------------- LOAD MODEL --------------------
with open("canberra_temp_model.pkl", "rb") as f:
    model = pickle.load(f)

# -------------------- CUSTOM CSS --------------------
st.markdown(
    f"""
    <style>
    /* Background */
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/jpg;base64,{bg_image}");
        background-size: cover;
        background-attachment: fixed;
    }}

    /* Glass Panel */
    .glass {{
        background: rgba(255, 255, 255, 0.18);
        backdrop-filter: blur(100px);
        border-radius: 16px;
        padding: 25px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.25);
        color: #000;
    }}

    /* Headings */
    .title {{
        text-align: center;
        font-size: 38px;
        font-weight: bold;
        color: white;
        margin-bottom: 5px;
    }}

    .subtitle {{
        text-align: center;
        font-size: 20px;
        color: white;
        margin-bottom: 20px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------- HEADER --------------------
st.markdown('<div class="title">🌡 Canberra Weather Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-Powered Minimum Temperature Prediction & Climate Analysis</div>', unsafe_allow_html=True)

# -------------------- MAIN LAYOUT --------------------
col1, col2 = st.columns([1.2, 1])

# -------------------- INPUT PANEL --------------------
with col1:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("📥 Input Climate Parameters")

    year = st.slider("📅 Year", 2010, 2030, 2019)
    day_of_year = st.slider("🗓 Day of Year", 1, 366, 180)

    rolling7 = st.number_input("📈 7-Day Rolling Mean (°C)", value=5.0)
    rolling30 = st.number_input("📊 30-Day Rolling Mean (°C)", value=7.0)

    predict_btn = st.button("🔮 Predict Temperature")
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- OUTPUT PANEL --------------------
with col2:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("🤖 AI Prediction Output")

    if predict_btn:
        input_df = pd.DataFrame([{
            "Year": year,
            "DayOfYear": day_of_year,
            "Rolling7": rolling7,
            "Rolling30": rolling30
        }])

        prediction = model.predict(input_df)[0]

        st.metric(
            label="Predicted Minimum Temperature",
            value=f"{prediction:.2f} °C"
        )

        if prediction < 0:
            st.warning("❄ Freezing night expected — stay warm!")
        elif prediction < 5:
            st.info("🧥 Cold conditions — jacket recommended.")
        else:
            st.success("🌤 Mild weather conditions.")

    else:
        st.info("👈 Enter values and click **Predict Temperature**")

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- FOOTER --------------------
st.markdown(
    """
    <div style="text-align:center; color:white; margin-top:40px;">
    Built with ❤️ using Machine Learning & Streamlit<br>
    <b>Canberra Weather Analysis Project</b>
    </div>
    """,
    unsafe_allow_html=True
)
