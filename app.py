# app.py
import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="DSS ML Mini Project", layout="centered")
st.title("ระบบทำนายผลลัพธ์วิชาเลือก")

# Ensure the pickle file is in same folder when running streamlit
PICKLE = "best_model_joblib.pkl"
if not os.path.exists(PICKLE):
    st.error(f"ไม่พบไฟล์โมเดล '{PICKLE}'. โปรดวางไฟล์ในโฟลเดอร์เดียวกับ app.py")
    st.stop()

model_data = joblib.load(PICKLE)
pipe = model_data['pipeline']
le_target = model_data['label_encoder']
features = model_data.get('features', [])

st.markdown("### กรอกข้อมูลตัวแปร (features)")
inputs = {}
for feat in features:
    # พยายามเดาชนิด: ถ้าชื่อมีคำว่า 'difficulty' หรือ 'workload' ให้เป็นตัวเลข
    if any(k in feat.lower() for k in ['diff', 'work', 'score', 'rating', 'level']):
        inputs[feat] = st.number_input(f"{feat}", value=3, step=1)
    else:
        # สร้าง dropdown จากค่าที่เคยมีใน data ถ้าเป็นไปได้ (fallback เป็น text_input)
        # ถ้า dataset มีค่าให้ดึงมา
        try:
            # try to extract unique values from pipeline? fallback: text input
            inputs[feat] = st.text_input(f"{feat}", value="")
        except Exception:
            inputs[feat] = st.text_input(f"{feat}", value="")

if st.button("Submit / Predict"):
    input_df = pd.DataFrame({k: [v] for k, v in inputs.items()})
    try:
        pred = pipe.predict(input_df)
        proba = pipe.predict_proba(input_df) if hasattr(pipe, "predict_proba") else None
        label = le_target.inverse_transform(pred)[0]
        st.success(f"Predicted: **{label}**")
        if proba is not None:
            st.write("Probabilities:")
            for cls, p in zip(le_target.classes_, proba[0]):
                st.write(f"- {cls}: {p:.3f}")
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการทำนาย: {e}")
