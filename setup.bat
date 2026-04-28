@echo off
REM Setup script สำหรับ TB Prediction System (Windows)
REM TB Prediction System Setup Script for Windows

echo ==========================================
echo 🏥 TB Prediction System Setup (Windows)
echo ==========================================
echo.

REM ตรวจสอบว่ามี Python หรือไม่
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python ไม่พบในระบบ
    echo กรุณาติดตั้ง Python 3.8 หรือสูงกว่า
    pause
    exit /b 1
)

python --version
echo.

REM สร้าง virtual environment
echo 📦 กำลังสร้าง Virtual Environment...
python -m venv venv

if errorlevel 1 (
    echo ❌ ไม่สามารถสร้าง Virtual Environment ได้
    pause
    exit /b 1
)

echo ✅ สร้าง Virtual Environment สำเร็จ
echo.

REM เปิดใช้งาน virtual environment
echo 🔄 กำลังเปิดใช้งาน Virtual Environment...
call venv\Scripts\activate.bat

REM อัพเกรด pip
echo 📥 กำลังอัพเกรด pip...
python -m pip install --upgrade pip

REM ติดตั้ง dependencies
echo.
echo 📥 กำลังติดตั้ง Dependencies...
pip install -r requirements.txt

if errorlevel 1 (
    echo ❌ ไม่สามารถติดตั้ง Dependencies ได้
    pause
    exit /b 1
)

echo ✅ ติดตั้ง Dependencies สำเร็จ
echo.

REM สร้างโฟลเดอร์
echo 📁 กำลังสร้างโฟลเดอร์...
if not exist "models" mkdir models
if not exist "data" mkdir data
if not exist "notebooks" mkdir notebooks

echo ✅ สร้างโฟลเดอร์สำเร็จ
echo.

REM Train โมเดล
set /p train_model="🤖 Train โมเดล (y/n)? "
if /i "%train_model%"=="y" (
    echo 🔧 กำลัง Train โมเดล...
    python train_model.py
    
    if errorlevel 1 (
        echo ⚠️  Train โมเดลไม่สำเร็จ แต่สามารถใช้โมเดลตัวอย่างได้
    ) else (
        echo ✅ Train โมเดลสำเร็จ
    )
) else (
    echo ⏭️  ข้าม Train โมเดล (จะใช้โมเดลตัวอย่าง)
)

REM สรุปผล
echo.
echo ==========================================
echo ✅ Setup เสร็จสมบูรณ์!
echo ==========================================
echo.
echo 🚀 วิธีใช้งาน:
echo.
echo 1. เปิดใช้งาน Virtual Environment:
echo    venv\Scripts\activate
echo.
echo 2. รัน Streamlit App:
echo    streamlit run tb_prediction_app.py
echo.
echo 3. เปิดเว็บเบราว์เซอร์ที่:
echo    http://localhost:8501
echo.
echo ==========================================
echo.
pause
