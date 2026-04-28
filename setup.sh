#!/bin/bash

# Setup script สำหรับ TB Prediction System
# TB Prediction System Setup Script

echo "=========================================="
echo "🏥 TB Prediction System Setup"
echo "=========================================="
echo ""

# ตรวจสอบว่ามี Python หรือไม่
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 ไม่พบในระบบ"
    echo "กรุณาติดตั้ง Python 3.8 หรือสูงกว่า"
    exit 1
fi

echo "✅ พบ Python: $(python3 --version)"
echo ""

# สร้าง virtual environment
echo "📦 กำลังสร้าง Virtual Environment..."
python3 -m venv venv

if [ $? -eq 0 ]; then
    echo "✅ สร้าง Virtual Environment สำเร็จ"
else
    echo "❌ ไม่สามารถสร้าง Virtual Environment ได้"
    exit 1
fi

# เปิดใช้งาน virtual environment
echo ""
echo "🔄 กำลังเปิดใช้งาน Virtual Environment..."

if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    # Windows
    source venv/Scripts/activate
else
    # macOS/Linux
    source venv/bin/activate
fi

# ติดตั้ง dependencies
echo ""
echo "📥 กำลังติดตั้ง Dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ ติดตั้ง Dependencies สำเร็จ"
else
    echo "❌ ไม่สามารถติดตั้ง Dependencies ได้"
    exit 1
fi

# สร้างโฟลเดอร์
echo ""
echo "📁 กำลังสร้างโฟลเดอร์..."
mkdir -p models
mkdir -p data
mkdir -p notebooks

echo "✅ สร้างโฟลเดอร์สำเร็จ"

# Train โมเดล
echo ""
echo "🤖 Train โมเดล (y/n)? "
read -r response

if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo "🔧 กำลัง Train โมเดล..."
    python train_model.py
    
    if [ $? -eq 0 ]; then
        echo "✅ Train โมเดลสำเร็จ"
    else
        echo "⚠️  Train โมเดลไม่สำเร็จ แต่สามารถใช้โมเดลตัวอย่างได้"
    fi
else
    echo "⏭️  ข้าม Train โมเดล (จะใช้โมเดลตัวอย่าง)"
fi

# สรุปผล
echo ""
echo "=========================================="
echo "✅ Setup เสร็จสมบูรณ์!"
echo "=========================================="
echo ""
echo "🚀 วิธีใช้งาน:"
echo ""
echo "1. เปิดใช้งาน Virtual Environment:"
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    echo "   venv\\Scripts\\activate"
else
    echo "   source venv/bin/activate"
fi
echo ""
echo "2. รัน Streamlit App:"
echo "   streamlit run tb_prediction_app.py"
echo ""
echo "3. เปิดเว็บเบราว์เซอร์ที่:"
echo "   http://localhost:8501"
echo ""
echo "=========================================="
