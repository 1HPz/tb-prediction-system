# 🏥 ระบบทำนายอัตรารอดชีวิตจากโรควัณโรคปอด
## Tuberculosis Survival Prediction System

ระบบทำนายอัตรารอดชีวิตของผู้ป่วยโรควัณโรคปอดโดยใช้ Machine Learning (XGBoost) พร้อม Web Application ที่สร้างด้วย Streamlit

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.0+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40.0+-red.svg)

---

## 📋 คุณสมบัติหลัก (Features)

- ✅ ทำนายอัตรารอดชีวิตจากโรควัณโรคปอด (Binary Classification: Success/Death)
- ✅ ใช้โมเดล XGBoost Classifier ที่ปรับแต่งแล้ว
- ✅ ตั้งค่า Threshold = 0.6 สำหรับการตัดสินใจ
- ✅ อินเทอร์เฟซสวยงามด้วยธีมสี #282a4e
- ✅ แสดงความน่าจะเป็นแบบ Real-time พร้อมกราฟ Interactive
- ✅ รองรับการใช้งานผ่าน Web Browser (Desktop & Mobile)
- ✅ ใช้งานฟรีผ่าน Streamlit Community Cloud

---

## 🗂️ โครงสร้างโปรเจค (Project Structure)

```
tb-prediction-system/
│
├── tb_prediction_app.py          # Streamlit Web Application
├── train_model.py                 # สคริปต์สำหรับ Train โมเดล
├── requirements.txt               # Python dependencies
├── README.md                      # เอกสารนี้
├── LICENSE                        # MIT License
├── .gitignore                     # Git ignore file
│
├── data/                          # โฟลเดอร์สำหรับข้อมูล
│   └── README.md
│
├── models/                        # โฟลเดอร์สำหรับเก็บโมเดล
│   └── README.md
│
└── notebooks/                     # Jupyter Notebooks
    └── XGBoost_Classification_Undersampling.ipynb
```

---

## 🚀 การติดตั้งและใช้งาน (Installation)

### ขั้นตอนที่ 1: Clone Repository

```bash
git clone https://github.com/your-username/tb-prediction-system.git
cd tb-prediction-system
```

### ขั้นตอนที่ 2: สร้าง Virtual Environment (แนะนำ)

```bash
# สร้าง virtual environment
python -m venv venv

# เปิดใช้งาน virtual environment
# สำหรับ Windows:
venv\Scripts\activate

# สำหรับ macOS/Linux:
source venv/bin/activate
```

### ขั้นตอนที่ 3: ติดตั้ง Dependencies

```bash
pip install -r requirements.txt
```

### ขั้นตอนที่ 4: รัน Streamlit App

```bash
streamlit run tb_prediction_app.py
```

เว็บแอปจะเปิดที่ `http://localhost:8501`

---

## 📊 ตัวแปรที่ใช้ในการทำนาย (Features)

โมเดลใช้ตัวแปรทั้งหมด **7 ตัว** ในการทำนาย:

| ตัวแปร | ประเภท | คำอธิบาย | ตัวเลือก/ช่วง |
|--------|--------|----------|---------------|
| **Gender** | Categorical | เพศของผู้ป่วย | ชาย / หญิง |
| **Age** | Continuous | อายุของผู้ป่วย | 0-120 ปี |
| **Weight** | Continuous | น้ำหนักของผู้ป่วย | 0-300 กิโลกรัม |
| **Site of Disease** | Categorical | ตำแหน่งของโรควัณโรค | ในปอด / นอกปอด / ในและนอกปอด |
| **HIV Status** | Binary | สถานะการติดเชื้อ HIV | Negative (0) / Positive (1) |
| **Comorbidities** | Multiple Selection | โรคประจำตัว | CKD, COPD, DM, Liver Disease |
| **ICD-10 Code** | Categorical | รหัสโรค ICD-10 | ไม่ระบุ + 41 รหัสโรค |

### รายละเอียดตัวแปร:

#### 1. Gender (เพศ)
- **ชาย** (Male) = 1
- **หญิง** (Female) = 0

#### 2. Age (อายุ)
- ระบุเป็นตัวเลข (0-120)
- ตัวอย่าง: 45 ปี

#### 3. Weight (น้ำหนัก)
- ระบุเป็นกิโลกรัม (รองรับทศนิยม)
- ตัวอย่าง: 65.5 kg

#### 4. Site of Disease (ตำแหน่งโรค)
- **ในปอด** (Pulmonary TB)
- **นอกปอด** (Extrapulmonary TB)
- **ในและนอกปอด** (Both)

#### 5. HIV Status (สถานะ HIV)
- **Negative** = 0 (ไม่ติดเชื้อ)
- **Positive** = 1 (ติดเชื้อ)

#### 6. Comorbidities (โรคประจำตัว)
เลือกได้หลายรายการ:
- โรคไตเรื้อรัง (Chronic Kidney Disease - CKD)
- โรคปอดอุดกั้นเรื้อรัง (Chronic Obstructive Pulmonary Disease - COPD)
- โรคเบาหวาน (Diabetes Mellitus - DM)
- โรคตับ (Liver Disease)

#### 7. ICD-10 Code (รหัสโรค)
เลือกได้ 1 รายการจาก 42 ตัวเลือก:
- ไม่ระบุ
- วัณโรคปอด ยืนยันด้วยผลการตรวจเสมหะ...
- วัณโรคปอด ยืนยันด้วยผลการตรวจชิ้นเนื้อ
- (และอีก 39 รหัส)

---

## 🎯 ผลลัพธ์ (Output)

โมเดลจะทำนายผลลัพธ์เป็น **2 กลุ่ม**:

### ผลการทำนาย:
- **รอดชีวิต (Success)** = 0 
  - กล่องสีเขียว (#6aef4f) ✅
  
- **เสียชีวิต (Death)** = 1
  - กล่องสีแดง (#ff352e) ⚠️

### การตัดสินใจ (Decision Rule):

```
ถ้า P(Death) ≥ 0.6 → ทำนายว่า "เสียชีวิต"
ถ้า P(Death) < 0.6 → ทำนายว่า "รอดชีวิต"
```

โดย **Threshold = 0.6 (60%)** ถูกเลือกเพื่อความสมดุลระหว่าง Sensitivity และ Specificity

---

## ⚙️ การตั้งค่าโมเดล (Model Configuration)

```python
XGBClassifier(
    colsample_bytree=0.7,
    learning_rate=0.05,
    max_depth=4,
    n_estimators=300,
    scale_pos_weight=2.98,
    subsample=0.7,
    objective='binary:logistic',
    random_state=42,
    tree_method='hist',
    eval_metric='aucpr'
)
```

---

## 🎨 การออกแบบ UI/UX

### ธีมสี:
- **พื้นหลัง**: #282a4e (น้ำเงินเข้ม)
- **กล่องรอดชีวิต**: #6aef4f (เขียว)
- **กล่องเสียชีวิต**: #ff352e (แดง)
- **ข้อความ**: สีขาว (#ffffff)

### คุณสมบัติพิเศษ:
- Glass Morphism Design
- Responsive Layout (รองรับมือถือ)
- Interactive Plotly Charts
- Real-time Prediction

---

## 📱 วิธีใช้งาน Web App

### ขั้นตอนการใช้งาน:

1. **กรอกข้อมูลผู้ป่วย** (ฝั่งซ้าย)
   - เลือก**เพศ** (ชาย/หญิง)
   - ใส่**อายุ** (ตัวเลข)
   - ใส่**น้ำหนัก** (กิโลกรัม)
   - เลือก**ตำแหน่งโรค** (ในปอด/นอกปอด/ทั้งสอง)
   - เลือก**สถานะ HIV** (Negative/Positive)
   - เลือก**โรคประจำตัว** (เลือกได้หลายรายการ)
   - เลือก**รหัส ICD-10** (เลือก 1 รายการ)

2. **กดปุ่ม "🔮 ทำนายผล"**

3. **ดูผลลัพธ์** (ฝั่งขวา)
   - กล่องแสดงผลการทำนาย (สีเขียว/แดง)
   - ความน่าจะเป็นแบบตัวเลข
   - กราฟแท่งแสดงความน่าจะเป็น
   - คำอธิบายการตัดสินใจ

---

## 🔬 การ Train โมเดลใหม่

หากต้องการ train โมเดลใหม่ด้วยข้อมูลของคุณเอง:

```python
# ใน train_model.py
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import joblib

# โหลดข้อมูล
# Features: gender, age, weight, site_pulmonary, site_extrapulmonary, 
#           hiv_status, comorbid_ckd, comorbid_copd, comorbid_dm, 
#           comorbid_liver, icd10_selected
X_train = pd.read_csv('your_X_train.csv')
y_train = pd.read_csv('your_y_train.csv')

# สร้างและ train โมเดล
model = xgb.XGBClassifier(
    colsample_bytree=0.7,
    learning_rate=0.05,
    max_depth=4,
    n_estimators=300,
    scale_pos_weight=2.98,
    subsample=0.7,
    objective='binary:logistic',
    random_state=42
)

model.fit(X_train, y_train)

# บันทึกโมเดล
joblib.dump(model, 'models/tb_model.pkl')
```

---

## 📦 Dependencies

### Python Packages:
```txt
streamlit>=1.40.0
pandas>=2.2.0
numpy>=1.26.0
xgboost>=2.0.0
scikit-learn>=1.4.0
matplotlib>=3.8.0
seaborn>=0.13.0
plotly>=5.18.0
joblib>=1.3.0
```

### เวอร์ชันที่แนะนำ:
- **Python**: 3.11+
- **Streamlit**: 1.40.0+
- **XGBoost**: 2.0.0+
- **Pandas**: 2.2.0+

---

## 🌐 Deploy บน Streamlit Community Cloud

### ขั้นตอนการ Deploy:

1. **Push โค้ดขึ้น GitHub**
   ```bash
   git add .
   git commit -m "Deploy TB Prediction System"
   git push origin main
   ```

2. **ไปที่ Streamlit Cloud**
   - เข้า https://share.streamlit.io
   - Login ด้วย GitHub
   - กด "New app"

3. **เลือก Repository**
   - Repository: `your-username/tb-prediction-system`
   - Branch: `main`
   - Main file: `tb_prediction_app.py`

4. **กด Deploy!**
   - รอ 2-3 นาที
   - เว็บแอปพร้อมใช้งาน! 🎉

### URL ตัวอย่าง:
```
https://your-app-name.streamlit.app
```

---

## ⚠️ ข้อควรระวัง (Disclaimer)

> **⚠️ ระบบนี้เป็นเครื่องมือช่วยตัดสินใจเท่านั้น**
> 
> ผลการทำนายจากระบบควรใช้ร่วมกับ:
> - ✅ การวินิจฉัยของแพทย์ผู้เชี่ยวชาญ
> - ✅ ผลการตรวจทางห้องปฏิบัติการ
> - ✅ ประวัติการรักษาของผู้ป่วย
> - ✅ ปัจจัยทางคลินิกอื่นๆ
>
> **❌ ห้ามใช้เป็นเครื่องมือเดียวในการตัดสินใจทางการแพทย์**

---

## 📊 ตัวอย่างการใช้งาน

### ตัวอย่างที่ 1: ผู้ป่วยความเสี่ยงต่ำ
```
เพศ: หญิง
อายุ: 35 ปี
น้ำหนัก: 55 kg
ตำแหน่งโรค: ในปอด
HIV: Negative
โรคประจำตัว: ไม่มี
ICD-10: วัณโรคปอด ยืนยันด้วยผลการตรวจเสมหะ

→ ผลลัพธ์: ✅ รอดชีวิต (Success)
→ ความน่าจะเป็น: 85% รอดชีวิต, 15% เสียชีวิต
```

### ตัวอย่างที่ 2: ผู้ป่วยความเสี่ยงสูง
```
เพศ: ชาย
อายุ: 68 ปี
น้ำหนัก: 45 kg
ตำแหน่งโรค: ในและนอกปอด
HIV: Positive
โรคประจำตัว: CKD, DM, COPD
ICD-10: Acute miliary tuberculosis of multiple sites

→ ผลลัพธ์: ⚠️ เสียชีวิต (Death)
→ ความน่าจะเป็น: 25% รอดชีวิต, 75% เสียชีวิต
```

---

## 🤝 การมีส่วนร่วม (Contributing)

ยินดีรับ Pull Requests! สำหรับการเปลี่ยนแปลงที่สำคัญ กรุณาเปิด Issue ก่อน

### ขั้นตอนการ Contribute:
1. Fork the Project
2. Create Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit Changes (`git commit -m 'Add AmazingFeature'`)
4. Push to Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

**Medical Disclaimer:** This software is for educational and research purposes only. Not intended for medical diagnosis or treatment.

---

## 👨‍💻 ผู้พัฒนา (Developer)

- **Project**: TB Survival Prediction System
- **Model**: XGBoost Binary Classifier
- **Framework**: Streamlit
- **Year**: 2024-2026

---

## 📞 ติดต่อ (Contact)

หากมีคำถามหรือข้อเสนอแนะ:
- 📧 Email: mochi23102548@gmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/1HPz/tb-prediction-system/issues)
- 🌐 Live Demo: [Streamlit App](https://tb-prediction-system-4kscmuxfey5ngeqzijqbui.streamlit.app/)

---

## 🙏 Acknowledgments

- ขอบคุณข้อมูลและแนวคิดจากงานวิจัยด้านโรควัณโรคปอด
- ขอบคุณ XGBoost และ Streamlit communities
- ขอบคุณผู้ใช้งานทุกท่านสำหรับ feedback

---

## 📈 Version History

- **v2.0** (2026-04-28)
  - เพิ่มตัวแปร 7 ตัวแปรใหม่
  - ปรับ UI/UX ด้วยธีมสี #282a4e
  - เปลี่ยน ICD-10 เป็นเลือกตัวเดียว
  - แก้ HIV Status เป็น Negative/Positive เท่านั้น
  - อัพเดท dependencies สำหรับ Python 3.14

- **v1.0** (2024)
  - เวอร์ชันแรก
  - โมเดล XGBoost พื้นฐาน

---

**Made with ❤️ for Medical Data Science**

*ระบบนี้พัฒนาขึ้นเพื่อช่วยเหลือแพทย์และบุคลากรทางการแพทย์ในการประเมินความเสี่ยงของผู้ป่วยโรควัณโรคปอด*
