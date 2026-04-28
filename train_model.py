"""
สคริปต์สำหรับ Train โมเดล XGBoost สำหรับทำนายอัตรารอดชีวิตจากโรควัณโรคปอด
Tuberculosis Survival Prediction Model Training Script

ใช้ตัวแปร 7 ตัว:
- age: อายุ
- ICD_10_0, ICD_10_3, ICD_10_4, ICD_10_5: รหัสโรค ICD-10
- pos_disease_0: โรคประจำตัว
- HIV_0: สถานะ HIV
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime

def create_synthetic_data(n_samples=2000, random_state=42):
    """
    สร้างข้อมูลจำลองสำหรับ demo
    ในการใช้งานจริง ให้เปลี่ยนเป็นการโหลดข้อมูลจริงของคุณ
    """
    np.random.seed(random_state)
    
    # สร้าง features
    data = {
        'age': np.random.randint(18, 85, n_samples),
        'ICD_10_0': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'ICD_10_3': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'ICD_10_4': np.random.choice([0, 1], n_samples, p=[0.75, 0.25]),
        'ICD_10_5': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        'pos_disease_0': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'HIV_0': np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
    }
    
    X = pd.DataFrame(data)
    
    # สร้าง target โดยมีความสัมพันธ์กับ features
    # ยิ่งอายุมาก มี HIV, มีโรคประจำตัว = โอกาสเสียชีวิตสูง
    death_prob = (
        0.1 +  # base probability
        0.3 * (X['age'] > 60).astype(int) +  # อายุมาก
        0.25 * X['HIV_0'] +  # HIV
        0.2 * X['pos_disease_0'] +  # โรคประจำตัว
        0.15 * X['ICD_10_4'] +  # ICD-10 code 4
        0.1 * X['ICD_10_5'] +  # ICD-10 code 5
        0.05 * (X['age'] * X['HIV_0'] / 100)  # interaction
    )
    
    # จำกัดไว้ที่ 0-1
    death_prob = np.clip(death_prob, 0, 1)
    
    # สร้าง target
    y = (np.random.random(n_samples) < death_prob).astype(int)
    
    return X, y

def plot_feature_importance(model, feature_names, save_path='models/feature_importance.png'):
    """แสดงและบันทึกกราฟความสำคัญของ features"""
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importance')
    plt.bar(range(len(importance)), importance[indices], color='steelblue')
    plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ บันทึกกราฟ Feature Importance ที่: {save_path}")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_path='models/confusion_matrix.png'):
    """แสดงและบันทึก Confusion Matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Success', 'Death'],
                yticklabels=['Success', 'Death'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ บันทึก Confusion Matrix ที่: {save_path}")
    plt.close()

def evaluate_with_threshold(model, X_test, y_test, threshold=0.6):
    """ประเมินโมเดลด้วย threshold ที่กำหนด"""
    # ทำนายความน่าจะเป็น
    y_proba = model.predict_proba(X_test)[:, 1]  # ความน่าจะเป็นของ class 1 (Death)
    
    # ใช้ threshold ในการตัดสินใจ
    y_pred = (y_proba >= threshold).astype(int)
    
    return y_pred, y_proba

def main():
    """ฟังก์ชันหลักสำหรับ train โมเดล"""
    
    print("=" * 60)
    print("🏥 TB Survival Prediction Model Training")
    print("=" * 60)
    
    # สร้างโฟลเดอร์สำหรับเก็บโมเดล
    os.makedirs('models', exist_ok=True)
    
    # 1. โหลดหรือสร้างข้อมูล
    print("\n📊 กำลังโหลดข้อมูล...")
    
    # ===== ในการใช้งานจริง แทนที่ส่วนนี้ด้วยการโหลดข้อมูลของคุณ =====
    # X_train = pd.read_csv('data/X_train.csv')
    # y_train = pd.read_csv('data/y_train.csv')
    # X_test = pd.read_csv('data/X_test.csv')
    # y_test = pd.read_csv('data/y_test.csv')
    # ====================================================================
    
    # สำหรับ demo: สร้างข้อมูลจำลอง
    X, y = create_synthetic_data(n_samples=2000)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   ขนาดข้อมูล Train: {X_train.shape}")
    print(f"   ขนาดข้อมูล Test: {X_test.shape}")
    print(f"   จำนวน Features: {X_train.shape[1]}")
    print(f"   Class Distribution (Train): {np.bincount(y_train)}")
    print(f"   Class Distribution (Test): {np.bincount(y_test)}")
    
    # 2. สร้างและ Train โมเดล
    print("\n🔧 กำลัง Train โมเดล XGBoost...")
    
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
    
    # Train โมเดล
    model.fit(
        X_train, 
        y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False
    )
    
    print("   ✅ Train โมเดลเสร็จสิ้น")
    
    # 3. ประเมินผล
    print("\n📈 กำลังประเมินผล...")
    
    # ทำนายด้วย threshold = 0.6
    y_pred, y_proba = evaluate_with_threshold(model, X_test, y_test, threshold=0.6)
    
    # คำนวณ metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    print(f"\n   Accuracy: {accuracy:.4f}")
    print(f"   ROC-AUC Score: {roc_auc:.4f}")
    print(f"   Threshold: 0.6")
    
    print("\n   Classification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Success', 'Death'],
                                digits=4))
    
    # 4. สร้างกราฟ
    print("\n📊 กำลังสร้างกราฟ...")
    plot_feature_importance(model, X_train.columns.tolist())
    plot_confusion_matrix(y_test, y_pred)
    
    # 5. Cross-validation
    print("\n🔄 กำลังทำ Cross-validation...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    print(f"   Cross-validation ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # 6. บันทึกโมเดล
    print("\n💾 กำลังบันทึกโมเดล...")
    model_path = 'models/tb_model.pkl'
    joblib.dump(model, model_path)
    print(f"   ✅ บันทึกโมเดลที่: {model_path}")
    
    # บันทึกข้อมูล metadata
    metadata = {
        'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'n_samples_train': len(X_train),
        'n_samples_test': len(X_test),
        'features': X_train.columns.tolist(),
        'accuracy': float(accuracy),
        'roc_auc': float(roc_auc),
        'threshold': 0.6,
        'hyperparameters': model.get_params()
    }
    
    import json
    with open('models/model_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"   ✅ บันทึก metadata ที่: models/model_metadata.json")
    
    # 7. สรุปผล
    print("\n" + "=" * 60)
    print("✅ การ Train โมเดลเสร็จสมบูรณ์!")
    print("=" * 60)
    print(f"\n📌 สรุปผลการ Train:")
    print(f"   - โมเดล: XGBoost Classifier")
    print(f"   - Features: {len(X_train.columns)} ตัว")
    print(f"   - Accuracy: {accuracy:.2%}")
    print(f"   - ROC-AUC: {roc_auc:.4f}")
    print(f"   - Threshold: 0.6")
    print(f"\n📁 ไฟล์ที่สร้าง:")
    print(f"   - {model_path}")
    print(f"   - models/model_metadata.json")
    print(f"   - models/feature_importance.png")
    print(f"   - models/confusion_matrix.png")
    print("\n🚀 พร้อมใช้งานด้วยคำสั่ง: streamlit run tb_prediction_app.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
