import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import plotly.graph_objects as go

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="ระบบทำนายอัตรารอดชีวิตจากโรควัณโรคปอด",
    page_icon="🏥",
    layout="wide"
)

# CSS สำหรับธีมสีและตกแต่ง
st.markdown("""
<style>
    /* Solid Background */
    .stApp {
        background: #282a4e;
    }
    
    /* Main content styling */
    .main-header {
        font-size: 2.5rem;
        color: #ffffff;
        text-align: center;
        padding: 2rem;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    /* Card styling */
    .input-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        margin-bottom: 1rem;
    }
    
    /* Prediction boxes */
    .prediction-box {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: bold;
        margin: 2rem 0;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    .success-box {
        background: #6aef4f;
        color: white;
    }
    
    .death-box {
        background: #ff352e;
        color: white;
    }
    
    /* Info card */
    .info-card {
        background: rgba(255, 255, 255, 0.1);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }
    
    /* Override Streamlit default colors */
    .stSelectbox label, .stMultiSelect label, .stNumberInput label, .stRadio label {
        color: #1c2396 !important;
        font-weight: 600;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px 0 rgba(116, 79, 168, 0.75);
    }
    
    .stButton > button:hover {
        box-shadow: 0 6px 20px 0 rgba(116, 79, 168, 0.95);
        transform: translateY(-2px);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: #282a4e;
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: white;
    }
    
    /* Text color for main area */
    h1, h2, h3, p {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# รายการ ICD-10
ICD10_OPTIONS = [
    'วัณโรคปอด ยืนยันด้วยผลการตรวจเสมหะโดยใช้กล้องจุลทรรศน์ อาจมีหรือไม่มีการเพาะเชื้อ',
    'วัณโรคปอด ยืนยันด้วยผลการตรวจชิ้นเนื้อ',
    'วัณโรคปอด ยืนยันด้วยผลการเพาะเชื้อเท่านั้น',
    'วัณโรคระบบหายใจ ไม่ระบุรายละเอียด ยืนยันด้วยผลการตรวจหาเชื้อและการตรวจชิ้นเนื้อ',
    'วัณโรคปอด ผลการตรวจหาเชื้อและการตรวจชิ้นเนื้อเป็นลบ',
    'วัณโรคนอกปอด ต่อมน้ำเหลืองส่วนปลาย',
    'Acute miliary tuberculosis of multiple sites',
    'วัณโรคปอด ไม่ระบุการยืนยันด้วยผลการตรวจหาเชื้อหรือการตรวจชิ้นเนื้อ',
    'เยื่อหุ้มปอดอักเสบจากเชื้อวัณโรค ยืนยันด้วยผลการตรวจหาเชื้อและการตรวจชิ้นเนื้อ',
    'วัณโรคระบบหายใจส่วนอื่น ไม่ระบุการยืนยันด้วยผลการตรวจหาเชื้อหรือการตรวจชิ้นเนื้อ',
    'Acute miliary tuberculosis, Unspecified',
    'วัณโรคระบบหายใจส่วนอื่น ยืนยันด้วยผลการตรวจหาเชื้อและการตรวจชิ้นเนื้อ',
    'วัณโรคปอด ไม่ได้ตรวจหาเชื้อและไม่ได้ตรวจชิ้นเนื้อ',
    'วัณโรคนอกปอด ลำไส้ เยื่อบุช่องท้อง',
    'วัณโรคนอกปอด กระดูกและข้อ',
    'วัณโรคระบบหายใจ ไม่ระบุรายละเอียด ไม่ระบุการยืนยันด้วยผลการตรวจหาเชื้อหรือการตรวจชิ้นเนื้อ',
    'เยื่อหุ้มสมองอักเสบจากเชื้อวัณโรค (G01*)',
    'เยื่อหุ้มปอดอักเสบจากเชื้อวัณโรค ไม่ระบุการยืนยันด้วยผลการตรวจหาเชื้อหรือการตรวจชิ้นเนื้อ',
    'วัณโรคระบบหายใจชนิดปฐมภูมิ ยืนยันด้วยผลการตรวจหาเชื้อและการตรวจชิ้นเนื้อ',
    'วัณโรคปอด ยืนยันโดยไม่ระบุวิธี',
    'วัณโรคต่อมน้ำเหลืองในช่องอก ยืนยันด้วยผลการตรวจหาเชื้อและการตรวจชิ้นเนื้อ',
    'วัณโรคนอกปอด ระบบสืบพันธุ์และระบบทางเดินปัสสาวะ',
    'วัณโรคชนิดแพร่เชื้อกระจายเป็นจุดเล็กๆ',
    'วัณโรคนอกปอด ผิวหนัง',
    'Tuberculosis of other specified',
    'Miliary tuberculosis, unspecified',
    'วัณโรคต่อมน้ำเหลืองในช่องอก ไม่ระบุการยืนยันด้วยผลการตรวจหาเชื้อหรือการตรวจชิ้นเนื้อ',
    'ฝีวัณโรคเยื่อหุ้มสมอง (G07*)',
    'วัณโรคที่ส่วนอื่นของระบบประสาท',
    'วัณโรคนอกปอด ตา',
    'Other miliary tuberculosis',
    'วัณโรคนอกปอด หู',
    'วัณโรคกล่องเสียง ท่อลม และหลอดลม ไม่ระบุการยืนยันด้วยผลการตรวจหาเชื้อหรือการตรวจชิ้นเนื้อ',
    'Tuberculosis of adrenal glands',
    'วัณโรคระบบประสาท ไม่ระบุรายละเอียด (G99.9*)',
    'วัณโรคระบบหายใจชนิดปฐมภูมิ ไม่ระบุการยืนยันด้วยผลการตรวจหาเชื้อหรือการตรวจชิ้นเนื้อ',
    'วัณโรคกล่องเสียง ท่อลม และหลอดลม ยืนยันด้วยผลการตรวจหาเชื้อและการตรวจชิ้นเนื้อ',
    'โรคปอดอุดกั้นเรื้อรัง ไม่ระบุรายละเอียด',
    'TUBERCULOSIS OF JAWS AND/OR TM',
    'ไตวายเรื้อรัง ไม่ระบุรายละเอียด',
    'เบาหวานชนิดที่ไม่ต้องพึ่งอินซูลิน ไม่มีภาวะแทรกซ้อน'
]

# ฟังก์ชันสร้างโมเดลตัวอย่าง
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
        'gender': np.random.choice([0, 1], n_samples),
        'age': np.random.randint(18, 80, n_samples),
        'weight': np.random.uniform(40, 100, n_samples),
        'site_pulmonary': np.random.choice([0, 1], n_samples),
        'site_extrapulmonary': np.random.choice([0, 1], n_samples),
        'hiv_status': np.random.choice([0, 1, 2, 3], n_samples),
        'comorbid_ckd': np.random.choice([0, 1], n_samples),
        'comorbid_copd': np.random.choice([0, 1], n_samples),
        'comorbid_dm': np.random.choice([0, 1], n_samples),
        'comorbid_liver': np.random.choice([0, 1], n_samples),
        'icd10_selected': np.random.choice([0, 1], n_samples)
    })
    
    # สร้าง target แบบมีความสัมพันธ์กับ features
    y_dummy = ((X_dummy['age'] > 60).astype(int) + 
               (X_dummy['hiv_status'] == 1).astype(int) + 
               X_dummy['comorbid_ckd'] + 
               X_dummy['comorbid_copd'] +
               np.random.choice([0, 1], n_samples, p=[0.7, 0.3])) > 1
    y_dummy = y_dummy.astype(int)
    
    model.fit(X_dummy, y_dummy)
    return model

# โหลดโมเดล
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/tb_model.pkl')
        st.sidebar.success("✅ โหลดโมเดลสำเร็จ")
    except:
        model = create_dummy_model()
        st.sidebar.info("ℹ️ กำลังใช้โมเดลตัวอย่างสำหรับ demo")
    return model

# ฟังก์ชันทำนาย
def predict_survival(model, input_data, threshold=0.6):
    """ทำนายผลลัพธ์โดยใช้ threshold ที่กำหนด"""
    proba = model.predict_proba(input_data)[0]
    prediction = 1 if proba[1] >= threshold else 0
    return prediction, proba

# Header
st.markdown('<div class="main-header">🏥 ระบบทำนายอัตรารอดชีวิตจากโรควัณโรคปอด<br>Tuberculosis Survival Prediction System</div>', 
            unsafe_allow_html=True)

# โหลดโมเดล
model = load_model()

# สร้าง 2 columns
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("## 📝 กรอกข้อมูลผู้ป่วย")
    
    # 1. Gender
    gender = st.radio(
        "👤 เพศ (Gender)",
        options=["ชาย", "หญิง"],
        horizontal=True
    )
    gender_value = 1 if gender == "ชาย" else 0
    
    # 2. Age
    age = st.number_input(
        "🎂 อายุ (Age)",
        min_value=0,
        max_value=120,
        value=45,
        step=1,
        help="กรุณาระบุอายุเป็นตัวเลข"
    )
    
    # 3. Weight
    weight = st.number_input(
        "⚖️ น้ำหนัก (Weight) กิโลกรัม",
        min_value=0.0,
        max_value=300.0,
        value=60.0,
        step=0.1,
        format="%.1f",
        help="กรุณาระบุน้ำหนักเป็นตัวเลข (ทศนิยมได้)"
    )
    
    # 4. Site of disease
    site_disease = st.selectbox(
        "🫁 ตำแหน่งของโรค (Site of Disease)",
        options=["ในปอด", "นอกปอด", "ในและนอกปอด"],
        help="เลือกตำแหน่งของโรควัณโรค"
    )
    site_pulmonary = 1 if site_disease in ["ในปอด", "ในและนอกปอด"] else 0
    site_extrapulmonary = 1 if site_disease in ["นอกปอด", "ในและนอกปอด"] else 0
    
    # 5. HIV status
    hiv_status = st.radio(
        "🦠 สถานะ HIV (HIV Status)",
        options=["Negative", "Positive"],
        horizontal=True,
        help="เลือกสถานะการติดเชื้อ HIV"
    )
    hiv_status_value = 0 if hiv_status == "Negative" else 1
    
    # 6. Comorbidities
    st.markdown("### 🏥 โรคประจำตัว (Comorbidities)")
    comorbidities = st.multiselect(
        "เลือกโรคประจำตัว (เลือกได้หลายรายการ)",
        options=[
            "โรคไตเรื้อรัง (Chronic Kidney Disease)",
            "โรคปอดอุดกั้นเรื้อรัง (Chronic Obstructive Pulmonary Disease)",
            "โรคเบาหวาน (Diabetes Mellitus)",
            "โรคตับ (Liver Disease)"
        ],
        help="เลือกโรคประจำตัวที่มี"
    )
    
    comorbid_ckd = 1 if "โรคไตเรื้อรัง (Chronic Kidney Disease)" in comorbidities else 0
    comorbid_copd = 1 if "โรคปอดอุดกั้นเรื้อรัง (Chronic Obstructive Pulmonary Disease)" in comorbidities else 0
    comorbid_dm = 1 if "โรคเบาหวาน (Diabetes Mellitus)" in comorbidities else 0
    comorbid_liver = 1 if "โรคตับ (Liver Disease)" in comorbidities else 0
    
    # 7. ICD-10
    st.markdown("### 🏷️ รหัสโรค ICD-10")
    icd10_code = st.selectbox(
        "เลือกรหัส ICD-10 (เลือกได้ 1 รายการ)",
        options=["ไม่ระบุ"] + ICD10_OPTIONS,
        help="เลือกรหัสโรค ICD-10 ที่เกี่ยวข้อง"
    )
    icd10_selected = 0 if icd10_code == "ไม่ระบุ" else 1
    
    st.markdown("---")
    
    # ปุ่มทำนาย
    predict_button = st.button("🔮 ทำนายผล", type="primary", use_container_width=True)

with col2:
    st.markdown("## 📈 ผลการทำนาย")
    
    if predict_button:
        # เตรียมข้อมูลสำหรับทำนาย
        input_data = pd.DataFrame({
            'gender': [gender_value],
            'age': [age],
            'weight': [weight],
            'site_pulmonary': [site_pulmonary],
            'site_extrapulmonary': [site_extrapulmonary],
            'hiv_status': [hiv_status_value],
            'comorbid_ckd': [comorbid_ckd],
            'comorbid_copd': [comorbid_copd],
            'comorbid_dm': [comorbid_dm],
            'comorbid_liver': [comorbid_liver],
            'icd10_selected': [icd10_selected]
        })
        
        # ทำนายด้วย threshold = 0.6
        prediction, probabilities = predict_survival(model, input_data, threshold=0.6)
        
        # แสดงผลการทำนาย
        if prediction == 0:
            st.markdown(
                '<div class="prediction-box success-box">✅ คาดการณ์: รอดชีวิต (Success)</div>',
                unsafe_allow_html=True
            )
            result_text = "รอดชีวิต"
            result_color = "#6aef4f"
        else:
            st.markdown(
                '<div class="prediction-box death-box">⚠️ คาดการณ์: เสียชีวิต (Death)</div>',
                unsafe_allow_html=True
            )
            result_text = "เสียชีวิต"
            result_color = "#ff352e"
        
        # แสดงความน่าจะเป็น
        st.markdown("### 📊 ความน่าจะเป็น (Probability)")
        
        prob_col1, prob_col2 = st.columns(2)
        with prob_col1:
            st.metric(
                label="🟢 รอดชีวิต",
                value=f"{probabilities[0]*100:.2f}%"
            )
        with prob_col2:
            st.metric(
                label="🔴 เสียชีวิต",
                value=f"{probabilities[1]*100:.2f}%"
            )
        
        # สร้างกราฟแท่ง
        fig = go.Figure(data=[
            go.Bar(
                x=['รอดชีวิต', 'เสียชีวิต'],
                y=[probabilities[0]*100, probabilities[1]*100],
                marker_color=['#6aef4f', '#ff352e'],
                text=[f"{probabilities[0]*100:.1f}%", f"{probabilities[1]*100:.1f}%"],
                textposition='outside',
                textfont=dict(size=16, color='white')
            )
        ])
        
        fig.update_layout(
            title=dict(
                text="ความน่าจะเป็นของแต่ละผลลัพธ์",
                font=dict(color='white', size=18)
            ),
            yaxis_title="เปอร์เซ็นต์ (%)",
            xaxis_title="สถานะ",
            height=400,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.2)'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.2)')
        )
        
        # เพิ่มเส้น threshold
        fig.add_hline(
            y=60, 
            line_dash="dash", 
            line_color="yellow",
            line_width=2,
            annotation_text="Threshold = 60%",
            annotation_position="right",
            annotation_font_color="yellow"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # คำอธิบายเพิ่มเติม
        st.markdown(f"""
        <div class="info-card">
        <h3>💡 คำอธิบาย</h3>
        <p><strong>วิธีการตัดสินใจ:</strong></p>
        <ul>
        <li>ระบบใช้ <strong>Threshold = 0.6 (60%)</strong></li>
        <li>ถ้าความน่าจะเป็นของการเสียชีวิต ≥ 60% → คาดการณ์ว่า <strong>เสียชีวิต</strong></li>
        <li>ถ้าความน่าจะเป็นของการเสียชีวิต < 60% → คาดการณ์ว่า <strong>รอดชีวิต</strong></li>
        </ul>
        <p><strong>ผลการทำนาย:</strong></p>
        <ul>
        <li>ความน่าจะเป็นของการเสียชีวิต: <strong>{probabilities[1]*100:.2f}%</strong></li>
        <li>ผลการทำนาย: <strong>{result_text}</strong></li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # แสดงข้อมูลที่ใช้ทำนาย
        with st.expander("🔍 ดูข้อมูลที่ใช้ในการทำนาย"):
            summary_data = {
                'ตัวแปร': [
                    'เพศ', 'อายุ', 'น้ำหนัก', 'ตำแหน่งโรค', 
                    'สถานะ HIV', 'โรคประจำตัว', 'รหัส ICD-10'
                ],
                'ค่า': [
                    gender,
                    f"{age} ปี",
                    f"{weight} kg",
                    site_disease,
                    hiv_status,
                    ", ".join(comorbidities) if comorbidities else "ไม่มี",
                    icd10_code
                ]
            }
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
    
    else:
        st.markdown("""
        <div class="info-card">
        <p>👈 กรุณากรอกข้อมูลทางด้านซ้าย แล้วกดปุ่ม 'ทำนายผล'</p>
        <br>
        <h3>📖 วิธีการใช้งาน</h3>
        <ol>
        <li><strong>เพศ</strong> - เลือกเพศของผู้ป่วย</li>
        <li><strong>อายุ</strong> - ระบุอายุเป็นตัวเลข</li>
        <li><strong>น้ำหนัก</strong> - ระบุน้ำหนักเป็นกิโลกรัม</li>
        <li><strong>ตำแหน่งโรค</strong> - เลือกว่าเป็นวัณโรคในปอดหรือนอกปอด</li>
        <li><strong>สถานะ HIV</strong> - ระบุสถานะการติดเชื้อ HIV</li>
        <li><strong>โรคประจำตัว</strong> - เลือกโรคประจำตัวที่มี (เลือกได้หลายรายการ)</li>
        <li><strong>รหัส ICD-10</strong> - เลือกรหัสโรคที่เกี่ยวข้อง (เลือกได้ 1 รายการ)</li>
        <li><strong>กดปุ่มทำนายผล</strong> เพื่อดูผลการคาดการณ์</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

# Sidebar - ข้อมูลเพิ่มเติม
st.sidebar.markdown("## 📚 ข้อมูลโมเดล")
st.sidebar.markdown("""
**Model:** XGBoost Classifier

**Features:**
- Gender (เพศ)
- Age (อายุ)
- Weight (น้ำหนัก)
- Site of Disease (ตำแหน่งโรค)
- HIV Status (สถานะ HIV)
- Comorbidities (โรคประจำตัว)
- ICD-10 Codes (รหัสโรค)

**Target Classes:**
- 0: รอดชีวิต (Success)
- 1: เสียชีวิต (Death)
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔬 เกี่ยวกับ Threshold")
st.sidebar.markdown("""
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
<div style='text-align: center; color: white;'>
    <p>🏥 ระบบทำนายอัตรารอดชีวิตจากโรควัณโรคปอด | Tuberculosis Survival Prediction System</p>
    <p>⚠️ ระบบนี้เป็นเครื่องมือช่วยตัดสินใจเท่านั้น ควรใช้ร่วมกับการวินิจฉัยของแพทย์</p>
</div>
""", unsafe_allow_html=True)
