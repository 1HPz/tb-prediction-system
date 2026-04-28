# 🏥 ระบบทำนายอัตรารอดชีวิตจากโรควัณโรคปอด
## Tuberculosis Survival Prediction System

ระบบทำนายอัตรารอดชีวิตของผู้ป่วยโรควัณโรคปอดโดยใช้ Machine Learning (XGBoost) พร้อม Web Application ที่สร้างด้วย Streamlit

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-red.svg)

---

## 📋 คุณสมบัติหลัก (Features)

- ✅ ทำนายอัตรารอดชีวิตจากโรควัณโรคปอด (Binary Classification: Success/Death)
- ✅ ใช้โมเดล XGBoost Classifier ที่ปรับแต่งแล้ว
- ✅ ตั้งค่า Threshold = 0.6 สำหรับการตัดสินใจ
- ✅ อินเทอร์เฟซที่เป็นมิตรกับผู้ใช้ (Streamlit Web App)
- ✅ แสดงความน่าจะเป็นแบบ Real-time
- ✅ กราฟและ Visualization ที่เข้าใจง่าย
- ✅ รองรับการใช้งานผ่าน Web Browser

---

## 🗂️ โครงสร้างโปรเจค (Project Structure)

```
tb-prediction-system/
│
├── tb_prediction_app.py          # Streamlit Web Application
├── train_model.py                 # สคริปต์สำหรับ Train โมเดล
├── requirements.txt               # Python dependencies
├── README.md                      # เอกสารนี้
├── .gitignore                     # Git ignore file
│
├── data/                          # โฟลเดอร์สำหรับข้อมูล (optional)
│   └── README.md
│
├── models/                        # โฟลเดอร์สำหรับเก็บโมเดล
│   └── tb_model.pkl              # โมเดลที่ train แล้ว (จะถูกสร้างขึ้นอัตโนมัติ)
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

### ขั้นตอนที่ 4: Train โมเดล (ถ้ายังไม่มีไฟล์โมเดล)

```bash
python train_model.py
```

### ขั้นตอนที่ 5: รัน Streamlit App

```bash
streamlit run tb_prediction_app.py
```

เว็บแอปจะเปิดที่ `http://localhost:8501`

---

## 📊 ตัวแปรที่ใช้ในการทำนาย (Features)

โมเดลใช้ตัวแปรทั้งหมด **7 ตัว** ในการทำนาย:

| ตัวแปร | ประเภท | คำอธิบาย |
|--------|--------|----------|
| `age` | ต่อเนื่อง | อายุของผู้ป่วย (0-100 ปี) |
| `ICD_10_0` | Binary | รหัสโรค ICD-10 ประเภท 0 (0 = ไม่มี, 1 = มี) |
| `ICD_10_3` | Binary | รหัสโรค ICD-10 ประเภท 3 (0 = ไม่มี, 1 = มี) |
| `ICD_10_4` | Binary | รหัสโรค ICD-10 ประเภท 4 (0 = ไม่มี, 1 = มี) |
| `ICD_10_5` | Binary | รหัสโรค ICD-10 ประเภท 5 (0 = ไม่มี, 1 = มี) |
| `pos_disease_0` | Binary | สถานะโรคประจำตัว (0 = ไม่มี, 1 = มี) |
| `HIV_0` | Binary | สถานะการติดเชื้อ HIV (0 = ไม่มี, 1 = มี) |

---

## 🎯 ผลลัพธ์ (Output)

โมเดลจะทำนายผลลัพธ์เป็น **2 กลุ่ม**:

- **0 (Success)** = รอดชีวิต ✅
- **1 (Death)** = เสียชีวิต ⚠️

### การตัดสินใจ (Decision Rule)

```
ถ้า P(Death) ≥ 0.6 → ทำนายว่า "เสียชีวิต"
ถ้า P(Death) < 0.6 → ทำนายว่า "รอดชีวิต"
```

โดย **Threshold = 0.6** ถูกเลือกเพื่อความสมดุลระหว่าง Sensitivity และ Specificity

---

## ⚙️ การตั้งค่าโมเดล (Model Configuration)

```python
XGBClassifier(
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
```

### Hyperparameters สำคัญ:
- **learning_rate**: 0.05 (ค่อยๆ เรียนรู้เพื่อความแม่นยำ)
- **max_depth**: 4 (ป้องกัน overfitting)
- **n_estimators**: 300 (จำนวน boosting rounds)
- **scale_pos_weight**: 2.98 (จัดการ class imbalance)

---

## 📱 วิธีใช้งาน Web App

1. **กรอกข้อมูลผู้ป่วย**
   - เลื่อนแถบเพื่อเลือกอายุ
   - เลือกรหัสโรค ICD-10 ที่เกี่ยวข้อง
   - ระบุโรคประจำตัวและสถานะ HIV

2. **กดปุ่ม "ทำนายผล"**
   - ระบบจะแสดงผลการทำนาย
   - แสดงความน่าจะเป็นของแต่ละผลลัพธ์
   - แสดงกราฟและคำอธิบาย

3. **ตีความผลลัพธ์**
   - ดูผลการทำนาย (รอดชีวิต/เสียชีวิต)
   - ตรวจสอบความน่าจะเป็น
   - อ่านคำอธิบายเพิ่มเติม

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
X_train = pd.read_csv('your_X_train.csv')
y_train = pd.read_csv('your_y_train.csv')

# สร้างและ train โมเดล
model = xgb.XGBClassifier(
    colsample_bytree=0.7,
    learning_rate=0.05,
    max_depth=4,
    n_estimators=300,
    scale_pos_weight=2.9815837937384897,
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

- **Python**: 3.8+
- **Streamlit**: 1.31.0 - สำหรับสร้าง Web Interface
- **XGBoost**: 2.0.3 - Machine Learning Algorithm
- **Pandas**: 2.1.4 - Data Manipulation
- **NumPy**: 1.26.3 - Numerical Computing
- **Scikit-learn**: 1.4.0 - ML Tools
- **Matplotlib**: 3.8.2 - Plotting
- **Seaborn**: 0.13.1 - Statistical Visualization
- **Plotly**: 5.18.0 - Interactive Plots

---

## ⚠️ ข้อควรระวัง (Disclaimer)

> **ระบบนี้เป็นเครื่องมือช่วยตัดสินใจเท่านั้น**
> 
> ผลการทำนายจากระบบควรใช้ร่วมกับ:
> - การวินิจฉัยของแพทย์ผู้เชี่ยวชาญ
> - ผลการตรวจทางห้องปฏิบัติการ
> - ประวัติการรักษาของผู้ป่วย
> - ปัจจัยทางคลินิกอื่นๆ
>
> **ห้ามใช้เป็นเครื่องมือเดียวในการตัดสินใจทางการแพทย์**

---

## 🤝 การมีส่วนร่วม (Contributing)

ยินดีรับ Pull Requests! สำหรับการเปลี่ยนแปลงที่สำคัญ กรุณาเปิด Issue ก่อนเพื่อหารือเกี่ยวกับการเปลี่ยนแปลง

### ขั้นตอนการ Contribute:
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

---

## 👨‍💻 ผู้พัฒนา (Developer)

- **Project**: TB Survival Prediction System
- **Model**: XGBoost Binary Classifier
- **Framework**: Streamlit
- **Year**: 2024

---

## 📞 ติดต่อ (Contact)

หากมีคำถามหรือข้อเสนอแนะ กรุณาติดต่อ:
- 📧 Email: your-email@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/your-username/tb-prediction-system/issues)

---

## 🙏 Acknowledgments

- ขอบคุณข้อมูลและแนวคิดจากงานวิจัยด้านโรควัณโรคปอด
- ขอบคุณ XGBoost และ Streamlit communities
- ขอบคุณผู้ใช้งานทุกท่านสำหรับ feedback

---

**Made with ❤️ for Medical Data Science**
