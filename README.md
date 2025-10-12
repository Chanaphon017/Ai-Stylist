# 🧥 AI Stylist | Color Recommender

แอปช่วยแนะนำ “คู่สีเสื้อผ้า” จากรูปภาพจริง  
ตรวจจับชิ้นเสื้อผ้าด้วย YOLOv8-seg, ดูดสีหลักต่อชิ้น แล้วสร้างพาเลตสี (Complementary / Analogous / Triadic)  
พร้อมให้คะแนนความเข้ากันของชุด และคำแนะนำสไตล์แบบอ่านง่าย

---

## ฟีเจอร์เด่น

- **อัปโหลดรูปบุคคล** → ตรวจจับชิ้นเสื้อผ้าแบบ Segmentation
- **ดึงสีหลักของแต่ละชิ้น** (K-Means + กรองสีขาว/ดำ/เทา)
- **แนะนำโทนสีอย่างไวต่อชิ้น**  
  - คู่ตรงข้าม (Complementary)  
  - ใกล้เคียง (Analogous)  
  - สามเหลี่ยม (Triadic)
- **ให้คะแนนความเข้ากันของสองชิ้นที่ใหญ่สุดในภาพ** (0–100)  
  - พร้อมบอกประเภทหลัก (Monochrome / Analogous / Triadic / Complementary) และคำแนะนำใส่จริง
- **UI ภาษาไทย อ่านง่าย**  
  - แท็บภาพรวม (ต้นฉบับ + ภาพใส่กรอบตรวจจับ)  
  - แท็บผลลัพธ์รายชิ้น (การ์ดสีหลัก + จับคู่สีแบบเร็ว 3 แบบ)
- **ผู้ใช้ปรับได้:** Confidence, Image size  
- **ล็อก IoU ไว้ในโค้ด** เพื่อความเสถียร (ไม่แสดงใน UI)

---

## โครงสร้างโปรเจกต์

```
ai-stylist02/
├── app.py              # ตัวแอป Streamlit (UI ไทย + ให้คะแนน + แนะนำสี)
├── best.pt             # โมเดล YOLOv8-seg (วางโฟลเดอร์เดียวกับ app.py)
├── requirements.txt    # ไลบรารีที่ใช้
├── search.png          # ไอคอนหน้าแอป (ใส่ได้ถ้ามี)
└── README.md           # คู่มือนี้
```

---

## วิธีติดตั้งและรัน

1. **ติดตั้งไลบรารี**
    ```sh
    pip install -r requirements.txt
    ```

2. **วางโมเดล**
    - นำไฟล์ `best.pt` (YOLOv8-seg ที่เทรนไว้) มาไว้ข้าง `app.py`

3. **รันแอป**
    ```sh
    streamlit run app.py
    ```

> ถ้าไอคอนไม่ขึ้น ให้แน่ใจว่า `search.png` อยู่ข้าง `app.py`

---

## วิธีใช้งาน

1. เปิดแอป → อัปโหลดรูป .jpg/.png/.webp
2. ดูแท็บ **ภาพรวม**: รูปต้นฉบับ + รูปที่ใส่กรอบตรวจจับ
3. ไปที่ **ผลลัพธ์รายชิ้น**:  
   - การ์ดรายชิ้นจะแสดงสีหลัก และจับคู่สีแบบเร็ว 3 แบบ
   - ด้านบนมีบล็อกคำแนะนำสไตล์: คะแนนรวม (0–100), ประเภทหลัก และทิปสั้น ๆ
4. ปุ่ม **ดาวน์โหลดภาพผลลัพธ์** อยู่ท้ายแท็บภาพรวม

---

## หลักการให้คะแนน

- ให้คะแนนจากสองชิ้นที่ใหญ่สุดในภาพ
- ดูความต่างมุมสี (Δh°) บนวงล้อสี (HSV) เทียบกับแม่แบบ:
  - Monochrome ≈ 0°
  - Analogous ≈ 30°
  - Triadic ≈ 120°
  - Complementary ≈ 180°
- เพิ่มตัวปรับเชิงการรับรู้ (ΔE00 / CIEDE2000) กันคะแนนลวงจากสีที่ดู “ต่างมากเกินไปหรือคล้ายเกินไป”
- โค้ดที่เกี่ยวข้องอยู่ใน `app.py` ฟังก์ชัน:
  - `score_rules(...)` (คำนวณคะแนน)
  - `advice_text(...)` (ข้อความแนะนำ)

---

## ตารางแม็ปชื่อคลาส EN → TH

| EN                     | TH                |
|------------------------|-------------------|
| long_sleeved_dress     | ชุดเดรส           |
| long_sleeved_outwear   | เสื้อคลุมแขนยาว   |
| long_sleeved_shirt     | เสื้อแขนยาว       |
| short_sleeved_dress    | ชุดเดรส           |
| short_sleeved_outwear  | เสื้อคลุมแขนสั้น  |
| short_sleeved_shirt    | เสื้อแขนสั้น      |
| shorts                 | กางเกง            |
| skirt                  | กระโปรง           |
| sling                  | เสื้อ             |
| sling_dress            | ชุดเดรส           |
| trousers               | กางเกง            |
| vest                   | เสื้อ             |
| vest_dress             | ชุดเดรส           |

> ในแอปแสดง “ภาษาไทยเท่านั้น” เพื่อความอ่านง่าย

---

## ค่าที่ปรับได้ / ล็อกไว้

- **ปรับได้ใน Sidebar:**
  - Confidence: 0.05–0.85 (ค่าเริ่มต้น 0.25)
  - Image size: 320 / 480 / 640 / 800 (ค่าเริ่มต้น 640)
- **ล็อกไว้ในโค้ด:**
  - IoU = 0.45 (เพื่อความเสถียรของผลลัพธ์)

---

## ไลบรารีที่ใช้

- ultralytics==8.3.0
- streamlit
- pillow
- numpy
- scikit-learn
- opencv-python-headless

> รายละเอียด/พินเวอร์ชันอยู่ใน `requirements.txt`

---

## ทิป / แก้ปัญหาทั่วไป

### สีรูปเพี้ยน (ส้ม/เขียวทั้งภาพ)
เกิดจากใช้ `results.plot()` ซึ่งคืนภาพ BGR ไปแสดงเป็น RGB  
ใช้วิธีนี้แทน:
```python
annot_pil = res.plot(pil=True)   # ได้ PIL (RGB) ตรง ๆ
st.image(annot_pil, use_column_width=True)
```
หรือ:
```python
import cv2
annot_bgr = res.plot()
annot_rgb = cv2.cvtColor(annot_bgr, cv2.COLOR_BGR2RGB)
st.image(annot_rgb, use_column_width=True)
```
### โมเดล best.pt 
- https://drive.google.com/drive/folders/1l4f5AQY_4VghwC_XZ1_QZwgBjRAT9lwc?usp=sharing  

### โหลดโมเดลไม่ผ่าน
- ตรวจสอบพาธไฟล์ `best.pt` และสิทธิ์อ่านไฟล์

### ไม่เจอ mask / จับไม่ได้
- ตรวจสอบว่าโมเดลเป็น YOLOv8-seg
- ลองรูปที่ชัด/เห็นตัวคนมากขึ้น
- เพิ่ม Confidence เล็กน้อยหรือลดลงตามเคส

---

## เครดิต

พัฒนาโดย Chanaphon Phetnoi  
รหัสนักศึกษา 664230017 | ห้อง 66/45  
นักศึกษาสาขาเทคโนโลยีสารสนเทศ



  python -m pip install -U pip setuptools wheel
  pip install -r requirements.txt

  streamlit run app.py
