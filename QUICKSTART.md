# 🚀 Quick Start Guide

## การเริ่มต้นใช้งานอย่างรวดเร็ว

### วิธีที่ 1: ใช้ Setup Script (แนะนำ)

#### สำหรับ Windows:
```cmd
setup.bat
```

#### สำหรับ macOS/Linux:
```bash
chmod +x setup.sh
./setup.sh
```

---

### วิธีที่ 2: Setup ด้วยตนเอง

#### ขั้นตอนที่ 1: Clone Repository
```bash
git clone https://github.com/your-username/tb-prediction-system.git
cd tb-prediction-system
```

#### ขั้นตอนที่ 2: สร้าง Virtual Environment
```bash
python -m venv venv
```

#### ขั้นตอนที่ 3: เปิดใช้งาน Virtual Environment

**Windows:**
```cmd
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

#### ขั้นตอนที่ 4: ติดตั้ง Dependencies
```bash
pip install -r requirements.txt
```

#### ขั้นตอนที่ 5: Train โมเดล (ถ้าต้องการ)
```bash
python train_model.py
```

#### ขั้นตอนที่ 6: รัน Streamlit App
```bash
streamlit run tb_prediction_app.py
```

---

## 📱 การใช้งาน Web Application

1. **เปิดเบราว์เซอร์** ไปที่ `http://localhost:8501`

2. **กรอกข้อมูลผู้ป่วย:**
   - ปรับอายุด้วย slider
   - เลือก checkbox สำหรับรหัสโรค ICD-10
   - ระบุโรคประจำตัวและสถานะ HIV

3. **กดปุ่ม "ทำนายผล"**

4. **ดูผลลัพธ์:**
   - ผลการทำนาย (รอดชีวิต/เสียชีวิต)
   - ความน่าจะเป็นของแต่ละผลลัพธ์
   - กราฟแสดงผล

---

## 🔧 การแก้ปัญหาเบื้องต้น

### ปัญหา: ติดตั้ง dependencies ไม่สำเร็จ
**วิธีแก้:**
```bash
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

### ปัญหา: Streamlit รันไม่ได้
**วิธีแก้:**
```bash
# ตรวจสอบว่าอยู่ใน virtual environment
python -c "import sys; print(sys.prefix)"

# ติดตั้ง streamlit ใหม่
pip uninstall streamlit
pip install streamlit
```

### ปัญหา: โมเดลไม่พบ
**วิธีแก้:**
```bash
# Train โมเดลใหม่
python train_model.py

# หรือแอปจะใช้โมเดลตัวอย่างอัตโนมัติ
```

---

## 📊 ตัวอย่างการใช้งาน

### ตัวอย่างที่ 1: ผู้ป่วยความเสี่ยงต่ำ
```
อายุ: 35
ICD-10: ไม่มี
โรคประจำตัว: ไม่มี
HIV: ไม่มี
→ ผลลัพธ์: รอดชีวิต (Success)
```

### ตัวอย่างที่ 2: ผู้ป่วยความเสี่ยงสูง
```
อายุ: 70
ICD-10: มี Code 4, 5
โรคประจำตัว: มี
HIV: มี
→ ผลลัพธ์: เสียชีวิต (Death)
```

---

## 💡 เคล็ดลับ

1. **ใช้ threshold = 0.6** เป็นค่าเริ่มต้น แต่สามารถปรับได้ในโค้ด
2. **ตรวจสอบ Feature Importance** ใน `models/feature_importance.png`
3. **ดู Confusion Matrix** ใน `models/confusion_matrix.png`
4. **อ่าน Model Metadata** ใน `models/model_metadata.json`

---

## 📞 ต้องการความช่วยเหลือ?

- 📖 อ่าน [README.md](README.md) ฉบับเต็ม
- 🐛 รายงานปัญหาที่ [GitHub Issues](https://github.com/your-username/tb-prediction-system/issues)
- 📧 ติดต่อ: your-email@example.com

---

**Happy Predicting! 🎉**
