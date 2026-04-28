import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
import plotly.express as px

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="ระบบทำนายอัตรารอดชีวิตจากโรควัณโรคปอด",
    page_icon="🏥",
    layout="wide"
)

# CSS สำหรับตกแต่ง
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #e3f2fd 0%, #bbdefb 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 2rem 0;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .death-box {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    .info-card {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ฟังก์ชันสร้างโมเดลตัวอย่าง (ในกรณีที่ยังไม่มีโมเดลที่ train แล้ว)
@st.cache_resource
def create_dummy_model():
    """สร้างโมเดลตัวอย่างสำหรับ demo"""
    model = xgb.XGBClassifier(
        colsample_bytree=0.7,
        learning_rate=0.05,
        max_depth=4,
        n_estimators=300,
        scale_pos_weight=2.9815837937384897,
        subsample=0.7,
        objective='binary:logistic',
        random_state=42,
        tree_method='hist',
        eval_metric='aucpr'
    )
    
    # สร้างข้อมูลตัวอย่างสำหรับ fit โมเดล
    np.random.seed(42)
    n_samples = 1000
    X_dummy = pd.DataFrame({
        'age': np.random.randint(18, 80, n_samples),
        'ICD_10_0': np.random.choice([0, 1], n_samples),
        'ICD_10_3': np.random.choice([0, 1], n_samples),
        'ICD_10_4': np.random.choice([0, 1], n_samples),
        'ICD_10_5': np.random.choice([0, 1], n_samples),
        'pos_disease_0': np.random.choice([0, 1], n_samples),
        'HIV_0': np.random.choice([0, 1], n_samples)
    })
    
    # สร้าง target แบบมีความสัมพันธ์กับ features
    y_dummy = ((X_dummy['age'] > 60).astype(int) + 
               X_dummy['HIV_0'] + 
               X_dummy['ICD_10_4'] + 
               np.random.choice([0, 1], n_samples, p=[0.7, 0.3])) > 1
    y_dummy = y_dummy.astype(int)
    
    model.fit(X_dummy, y_dummy)
    return model

# โหลดโมเดล
@st.cache_resource
def load_model():
    try:
        # พยายามโหลดโมเดลที่ save ไว้
        model = joblib.load('tb_model.pkl')
        st.sidebar.success("✅ โหลดโมเดลสำเร็จ")
    except:
        # ถ้าไม่มี ให้สร้างโมเดลตัวอย่าง
        model = create_dummy_model()
        st.sidebar.info("ℹ️ กำลังใช้โมเดลตัวอย่างสำหรับ demo")
    return model

# ฟังก์ชันทำนาย
def predict_survival(model, input_data, threshold=0.6):
    """
    ทำนายผลลัพธ์โดยใช้ threshold ที่กำหนด
    
    Parameters:
    - model: โมเดล XGBoost ที่ train แล้ว
    - input_data: DataFrame ของข้อมูลที่ต้องการทำนาย
    - threshold: ค่า threshold สำหรับการจำแนกประเภท (default 0.6)
    
    Returns:
    - prediction: 0 (Death) หรือ 1 (Success)
    - probability: ความน่าจะเป็นของการเสียชีวิต (class 0)
    """
    # ทำนายความน่าจะเป็น
    proba = model.predict_proba(input_data)[0]
    
    # proba[0] = ความน่าจะเป็นของการรอดชีวิต (Success)
    # proba[1] = ความน่าจะเป็นของการเสียชีวิต (Death)
    
    # ใช้ threshold ในการตัดสินใจ
    # ถ้าความน่าจะเป็นของการเสียชีวิต (proba[1]) >= threshold แสดงว่าเสียชีวิต
    prediction = 1 if proba[1] >= threshold else 0
    
    return prediction, proba

# Header
st.markdown('<div class="main-header">🏥 ระบบทำนายอัตรารอดชีวิตจากโรควัณโรคปอด</div>', 
            unsafe_allow_html=True)
st.markdown("### 📊 Tuberculosis Survival Prediction System (Threshold = 0.6)")

# โหลดโมเดล
model = load_model()

# สร้าง 2 columns
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("## 📝 กรอกข้อมูลผู้ป่วย")
    
    # อายุ
    age = st.slider(
        "🎂 อายุ (Age)",
        min_value=0,
        max_value=100,
        value=45,
        help="อายุของผู้ป่วย"
    )
    
    st.markdown("### 🏷️ รหัสโรค ICD-10")
    
    col_icd1, col_icd2 = st.columns(2)
    with col_icd1:
        icd_10_0 = st.checkbox("ICD-10 Code 0", help="รหัสโรคประเภทที่ 0")
        icd_10_3 = st.checkbox("ICD-10 Code 3", help="รหัสโรคประเภทที่ 3")
    
    with col_icd2:
        icd_10_4 = st.checkbox("ICD-10 Code 4", help="รหัสโรคประเภทที่ 4")
        icd_10_5 = st.checkbox("ICD-10 Code 5", help="รหัสโรคประเภทที่ 5")
    
    st.markdown("### 🦠 โรคประจำตัว")
    
    col_disease1, col_disease2 = st.columns(2)
    with col_disease1:
        pos_disease_0 = st.checkbox(
            "โรคประจำตัว (Positive Disease)",
            help="มีโรคประจำตัวหรือไม่"
        )
    
    with col_disease2:
        hiv_0 = st.checkbox(
            "ติดเชื้อ HIV (HIV Infection)",
            help="ติดเชื้อ HIV หรือไม่"
        )
    
    # ปุ่มทำนาย
    predict_button = st.button("🔮 ทำนายผล", type="primary", use_container_width=True)

with col2:
    st.markdown("## 📈 ผลการทำนาย")
    
    if predict_button:
        # เตรียมข้อมูลสำหรับทำนาย
        input_data = pd.DataFrame({
            'age': [age],
            'ICD_10_0': [int(icd_10_0)],
            'ICD_10_3': [int(icd_10_3)],
            'ICD_10_4': [int(icd_10_4)],
            'ICD_10_5': [int(icd_10_5)],
            'pos_disease_0': [int(pos_disease_0)],
            'HIV_0': [int(hiv_0)]
        })
        
        # ทำนายด้วย threshold = 0.6
        prediction, probabilities = predict_survival(model, input_data, threshold=0.6)
        
        # แสดงผลการทำนาย
        if prediction == 0:
            st.markdown(
                '<div class="prediction-box success-box">✅ คาดการณ์: รอดชีวิต (Success)</div>',
                unsafe_allow_html=True
            )
            result_icon = "✅"
            result_text = "รอดชีวิต"
            result_color = "#28a745"
        else:
            st.markdown(
                '<div class="prediction-box death-box">⚠️ คาดการณ์: เสียชีวิต (Death)</div>',
                unsafe_allow_html=True
            )
            result_icon = "⚠️"
            result_text = "เสียชีวิต"
            result_color = "#dc3545"
        
        # แสดงความน่าจะเป็น
        st.markdown("### 📊 ความน่าจะเป็น (Probability)")
        
        prob_df = pd.DataFrame({
            'สถานะ': ['รอดชีวิต (Success)', 'เสียชีวิต (Death)'],
            'ความน่าจะเป็น': [probabilities[0], probabilities[1]],
            'เปอร์เซ็นต์': [f"{probabilities[0]*100:.2f}%", f"{probabilities[1]*100:.2f}%"]
        })
        
        st.dataframe(prob_df, use_container_width=True, hide_index=True)
        
        # สร้างกราฟแท่ง
        fig = go.Figure(data=[
            go.Bar(
                x=['รอดชีวิต', 'เสียชีวิต'],
                y=[probabilities[0]*100, probabilities[1]*100],
                marker_color=['#28a745', '#dc3545'],
                text=[f"{probabilities[0]*100:.1f}%", f"{probabilities[1]*100:.1f}%"],
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title="ความน่าจะเป็นของแต่ละผลลัพธ์",
            yaxis_title="เปอร์เซ็นต์ (%)",
            xaxis_title="สถานะ",
            height=400,
            showlegend=False
        )
        
        # เพิ่มเส้น threshold
        fig.add_hline(
            y=60, 
            line_dash="dash", 
            line_color="red",
            annotation_text="Threshold = 60%",
            annotation_position="right"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # คำอธิบายเพิ่มเติม
        st.markdown("### 💡 คำอธิบาย")
        st.info(f"""
        **วิธีการตัดสินใจ:**
        - ระบบใช้ **Threshold = 0.6 (60%)**
        - ถ้าความน่าจะเป็นของการเสียชีวิต ≥ 60% → คาดการณ์ว่า **เสียชีวิต**
        - ถ้าความน่าจะเป็นของการเสียชีวิต < 60% → คาดการณ์ว่า **รอดชีวิต**
        
        **ผลการทำนาย:**
        - ความน่าจะเป็นของการเสียชีวิต: **{probabilities[1]*100:.2f}%**
        - ผลการทำนาย: **{result_text}** {result_icon}
        """)
        
        # แสดงข้อมูลที่ใช้ทำนาย
        with st.expander("🔍 ดูข้อมูลที่ใช้ในการทำนาย"):
            st.dataframe(input_data.T, use_container_width=True)
    
    else:
        st.info("👈 กรุณากรอกข้อมูลทางด้านซ้าย แล้วกดปุ่ม 'ทำนายผล'")
        
        # แสดงตัวอย่างการทำงาน
        st.markdown("### 📖 วิธีการใช้งาน")
        st.markdown("""
        1. **กรอกอายุ** ของผู้ป่วย
        2. **เลือกรหัสโรค ICD-10** ที่เกี่ยวข้อง
        3. **ระบุโรคประจำตัว** และสถานะการติดเชื้อ HIV
        4. **กดปุ่มทำนายผล** เพื่อดูผลการคาดการณ์
        
        **หมายเหตุ:**
        - ระบบใช้โมเดล XGBoost Classifier
        - Threshold = 0.6 สำหรับการตัดสินใจ
        - ผลลัพธ์แบ่งเป็น 2 กลุ่ม: รอดชีวิต (0) และเสียชีวิต (1)
        """)

# Sidebar - ข้อมูลเพิ่มเติม
st.sidebar.markdown("## 📚 ข้อมูลโมเดล")
st.sidebar.markdown("""
**Model:** XGBoost Classifier

**Selected Features:**
- Age (อายุ)
- ICD-10 Codes (0, 3, 4, 5)
- Positive Disease Status
- HIV Status

**Hyperparameters:**
- colsample_bytree: 0.7
- learning_rate: 0.05
- max_depth: 4
- n_estimators: 300
- scale_pos_weight: 2.98
- **threshold: 0.6**

**Target Classes:**
- 0: รอดชีวิต (Success)
- 1: เสียชีวิต (Death)
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔬 เกี่ยวกับ Threshold")
st.sidebar.info("""
**Threshold = 0.6** หมายความว่า:

ระบบจะคาดการณ์ว่าผู้ป่วยจะ
เสียชีวิต เมื่อความน่าจะเป็น
ของการเสียชีวิต ≥ 60%

การตั้งค่า threshold สูงขึ้น
จะทำให้ระบบระมัดระวังมากขึ้น
ในการพยากรณ์ผลลัพธ์ที่เลวร้าย
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🏥 ระบบทำนายอัตรารอดชีวิตจากโรควัณโรคปอด | Tuberculosis Survival Prediction System</p>
    <p>⚠️ ระบบนี้เป็นเครื่องมือช่วยตัดสินใจเท่านั้น ควรใช้ร่วมกับการวินิจฉัยของแพทย์</p>
</div>
""", unsafe_allow_html=True)
