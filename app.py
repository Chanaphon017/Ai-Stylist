import streamlit as st
import numpy as np
from PIL import Image, ImageColor
import requests
from segmentation_utils import segment_clothes, extract_part

# --- ติดตั้ง scikit-learn และ webcolors อัตโนมัติถ้าไม่มี ---
try:
    from sklearn.cluster import KMeans
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'scikit-learn'])
    from sklearn.cluster import KMeans
try:
    import webcolors
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'webcolors'])
    import webcolors
try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
import io

# ---------------- ฟังก์ชันวิเคราะห์สี ----------------
def get_dominant_colors(image, k=10):
    # ใช้ rembg ลบพื้นหลัง (background) ออกก่อนเสมอ
    img = image
    if REMBG_AVAILABLE:
        try:
            img = remove_background(image)
        except Exception:
            img = image
    img = img.resize((800, 800))
    arr = np.array(img)
    # ถ้าเป็น RGBA (หลัง rembg) ให้ mask เฉพาะ pixel ที่ alpha > 0 (foreground)
    if arr.shape[-1] == 4:
        arr_rgb = arr[...,:3]
        alpha = arr[...,3]
        arr_fg = arr_rgb[alpha > 0]
    else:
        arr_fg = arr.reshape(-1, 3)
    # กรอง pixel ที่เป็นขาว/เทา (background) ออก (แต่ไม่กรองดำ)
    mask = ~(
        ((arr_fg[:,0]>220) & (arr_fg[:,1]>220) & (arr_fg[:,2]>220)) |  # ขาว
        ((np.abs(arr_fg[:,0]-arr_fg[:,1])<10) & (np.abs(arr_fg[:,1]-arr_fg[:,2])<10) & (arr_fg[:,0]>80) & (arr_fg[:,0]<200)) # เทา
    )
    arr_fg = arr_fg[mask]
    # กรองโทนสีเนื้อ (skin tone) ออก (RGB เฉดส้ม/ชมพูอ่อน)
    def is_skin(rgb):
        r, g, b = rgb
        return (
            (r > 95 and g > 40 and b > 20 and max([r,g,b]) - min([r,g,b]) > 15 and abs(r-g) > 15 and r > g and r > b) or
            (r > 200 and g > 160 and b > 130)  # ผิวขาวมาก
        )
    arr_fg = np.array([pix for pix in arr_fg if not is_skin(pix)])
    if len(arr_fg) < k:
        arr_fg = arr.reshape(-1,3)  # fallback ใช้ทุก pixel
    # ลดจำนวน pixel เพื่อความเร็ว
    if len(arr_fg) > 20000:
        idx = np.random.choice(len(arr_fg), 20000, replace=False)
        arr_fg = arr_fg[idx]
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=5).fit(arr_fg)
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    sorted_idx = np.argsort(-counts)
    colors = kmeans.cluster_centers_[sorted_idx].astype(int)
    return colors

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % tuple(rgb)

def get_color_name(rgb_tuple):
    try:
        return webcolors.rgb_to_name(tuple(rgb_tuple))
    except ValueError:
        min_diff = float('inf')
        closest_name = ''
        # รองรับ webcolors หลายเวอร์ชัน
        if hasattr(webcolors, 'CSS3_NAMES'):
            color_names = webcolors.CSS3_NAMES
        elif hasattr(webcolors, 'CSS3_NAMES_TO_HEX'):
            color_names = list(webcolors.CSS3_NAMES_TO_HEX.keys())
        elif hasattr(webcolors, 'HTML4_NAMES_TO_HEX'):
            color_names = list(webcolors.HTML4_NAMES_TO_HEX.keys())
        elif hasattr(webcolors, 'CSS21_NAMES_TO_HEX'):
            color_names = list(webcolors.CSS21_NAMES_TO_HEX.keys())
        else:
            color_names = ['black', 'white', 'red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'gray']
        for name in color_names:
            try:
                r_c, g_c, b_c = webcolors.name_to_rgb(name)
                diff = np.linalg.norm(np.array([r_c, g_c, b_c]) - np.array(rgb_tuple))
                if diff < min_diff:
                    min_diff = diff
                    closest_name = name
            except Exception:
                continue
        return closest_name

# ---------------- ฟังก์ชันวิเคราะห์ความเข้ากันของสี ----------------
def evaluate_color_match(colors):
    scores = []
    for i in range(len(colors)):
        for j in range(i+1, len(colors)):
            diff = np.linalg.norm(colors[i] - colors[j])
            scores.append(diff)
    avg_diff = np.mean(scores)
    std_diff = np.std(scores)
    min_diff = np.min(scores)
    max_diff = np.max(scores)
    # ปรับช่วงคะแนนให้ยืดหยุ่นขึ้น (40-90) และคำแนะนำละเอียดขึ้น
    if avg_diff < 30:
        return "โทนสีใกล้เคียงกันมาก เหมาะกับแนวมินิมอล/สุภาพ/เรียบหรู ดูสบายตา", 90
    elif avg_diff < 45:
        return "สีใกล้เคียงกัน เหมาะกับแนว casual, minimal, business casual, everyday look", 85
    elif avg_diff < 60:
        if std_diff < 15:
            return "สีหลักและสีรองกลมกลืน เหมาะกับแนว smart casual, soft tone", 80
        else:
            return "มีสีหลักโดดเด่น สีรองช่วยเสริม เหมาะกับแนวแฟชั่น/โมเดิร์น", 75
    elif avg_diff < 75:
        if std_diff > 30:
            return "สีตัดกันพอดี ดูมีสไตล์ เหมาะกับแนว creative, modern, street", 70
        else:
            return "สีตัดกันเล็กน้อย เพิ่มความน่าสนใจ เหมาะกับแนว everyday, pop", 68
    elif avg_diff < 90:
        if max_diff > 150:
            return "สีตัดกันชัดเจน เหมาะกับแนวแฟชั่นจัดจ้าน/สายฝอ/ปาร์ตี้/experimental", 60
        else:
            return "สีตัดกันแรงแต่ยังดูดี เหมาะกับแนวแฟชั่น/creative/statement look", 55
    else:
        return "สีตัดกันแรงมาก อาจดูขัดตา เหมาะกับลุคแฟชั่นจัดเต็มหรือ experimental (ควรเลือกสีรองใหม่)", 40

# ---------------- ฟังก์ชันทำนายแนวแต่งตัว ----------------
def predict_style(colors):
    # วิเคราะห์จากสีหลักและสีรอง
    main_color = colors[0]
    if len(colors) > 1:
        second_color = colors[1]
    else:
        second_color = main_color
    r, g, b = main_color
    r2, g2, b2 = second_color
    # เงื่อนไขซับซ้อนขึ้นและเพิ่มสไตล์ใหม่ ๆ
    if r < 80 and g < 80 and b < 80:
        return "แนวเท่ (Street / ดาร์กแฟชั่น) - โทนเข้ม/ดำ/เทา"
    elif r > 200 and g > 200 and b > 200:
        return "แนวหวาน (Pastel / ญี่ปุ่น) - โทนขาว/พาสเทล"
    elif r > 200 and g < 100 and b < 100:
        if abs(r2 - r) > 80 or abs(g2 - g) > 80 or abs(b2 - b) > 80:
            return "แนวแฟชั่นจัดจ้าน (สายฝอ/Pop) - สีสดตัดกัน"
        else:
            return "แนวแฟชั่นสดใส (Pop/Colorful)"
    elif abs(r-g) < 30 and abs(g-b) < 30 and abs(r-b) < 30 and r > 150:
        return "แนวสุภาพ/เรียบหรู (Smart Casual/Minimal) - โทนสีเดียวกัน"
    elif (r > 180 and g > 180) or (g > 180 and b > 180) or (r > 180 and b > 180):
        return "แนวหวาน/สดใส (Pastel/ญี่ปุ่น/เกาหลี)"
    elif max(r, g, b) - min(r, g, b) > 150:
        if r > 200 and g > 200 and b < 100:
            return "แนว Summer สดใส (เหลือง/ส้ม/ฟ้า)"
        elif r < 100 and g > 150 and b < 100:
            return "แนว Earth Tone (เขียว/น้ำตาล/ธรรมชาติ)"
        elif r > 200 and g < 100 and b > 200:
            return "แนว Neon/Retro (ชมพู/ม่วง/ฟ้า)"
        elif r < 100 and g < 100 and b > 180:
            return "แนว Denim/Blue Jeans (น้ำเงิน/ฟ้า)"
        elif r > 180 and g > 120 and b < 80:
            return "แนว Autumn (น้ำตาล/ส้ม/เหลือง)"
        elif r < 100 and g > 100 and b > 100:
            return "แนว Winter (ฟ้า/เขียว/ขาว)"
        elif r > 200 and g > 200 and b > 200:
            return "แนว Spring (พาสเทล/สดใส)"
        elif r > 180 and g > 180 and b > 180:
            return "แนว Monochrome (ขาว/เทา/ดำ)"
        else:
            return "แนวแฟชั่น/สตรีท/Creative - สีตัดกันชัด"
    elif r > 150 and g > 150 and b < 100:
        return "แนว Luxury/Business (ทอง/เหลือง/น้ำตาล)"
    elif r < 100 and g > 150 and b > 150:
        return "แนว Sport/Active (ฟ้า/เขียว/น้ำเงิน)"
    else:
        return "แนวมินิมอล / สุภาพ / Everyday Look"

def remove_background(image):
    if not REMBG_AVAILABLE:
        return image
    try:
        img_rgba = image.convert("RGBA")
        img_no_bg = remove(img_rgba)
        return img_no_bg.convert("RGB")
    except Exception:
        return image

def remove_background_bytes(image_bytes):
    """ลบพื้นหลังจาก bytes (input: bytes, output: PIL Image RGBA หรือ None) พร้อม refine ขอบให้เนียน"""
    if not REMBG_AVAILABLE:
        return None
    try:
        input_image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        output_image = remove(input_image)
        # --- refine ขอบ alpha ให้เนียนขึ้น ---
        output_image = refine_alpha_edges(output_image, method="morph+blur", ksize=5, blur_sigma=1.2)
        return output_image
    except Exception:
        return None

def manual_remove_bg(image, bg_color, tolerance=30):
    """
    ลบพื้นหลังโดยใช้ threshold สี (bg_color: hex หรือ tuple, tolerance: int)
    คืนค่า RGBA (พื้นหลังโปร่งใส)
    """
    img = image.convert("RGBA")
    arr = np.array(img)
    if isinstance(bg_color, str):
        bg_rgb = ImageColor.getrgb(bg_color)
    else:
        bg_rgb = tuple(bg_color)
    diff = np.abs(arr[...,:3] - np.array(bg_rgb)).sum(axis=-1)
    mask = diff < tolerance
    arr[...,3][mask] = 0
    return Image.fromarray(arr)

def call_hf_fashion_classifier(image: Image.Image):
    """
    ส่งภาพไปยัง HuggingFace Spaces Fashion-Classifier API และคืนค่าผลลัพธ์
    """
    API_URL = "https://hf.space/embed/KP-whatever/Fashion-Classifier/api/predict/"
    import io
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    buffered.seek(0)
    files = {"data": ("image.png", buffered, "image/png")}
    try:
        response = requests.post(API_URL, files=files, timeout=30)
        if response.status_code == 200:
            result = response.json()
            # ปรับตามโครงสร้าง JSON ที่ได้จาก API
            if "data" in result and len(result["data"]) > 0:
                return result["data"][0]
            return str(result)
        else:
            return f"API Error: {response.status_code}"
    except Exception as e:
        return f"API Error: {e}"

def advanced_predict_style(colors, image=None):
    """
    วิเคราะห์สไตล์แฟชั่นโดยใช้ตรรกะที่ซับซ้อนขึ้น เช่น
    - วิเคราะห์ distribution ของสีทั้งภาพ (ไม่ใช่แค่ 2 สี)
    - ตรวจจับความสดใส/ความหม่น/ความ contrast
    - ใช้ข้อมูล saturation, brightness, และความหลากหลายของสี
    - คืนค่าความมั่นใจ (probability) ร่วมด้วย
    """
    import colorsys
    arr = np.array(image.resize((200,200))) if image is not None else None
    if arr is not None:
        arr = arr.reshape(-1,3)
        hsv = np.array([colorsys.rgb_to_hsv(*(pix/255.0)) for pix in arr])
        mean_sat = np.mean(hsv[:,1])
        mean_val = np.mean(hsv[:,2])
        std_val = np.std(hsv[:,2])
        color_variety = len(np.unique(arr, axis=0))
    else:
        mean_sat = mean_val = std_val = color_variety = 0
    main_color = colors[0]
    second_color = colors[1] if len(colors) > 1 else main_color
    r, g, b = main_color
    r2, g2, b2 = second_color
    # ตรรกะใหม่: วิเคราะห์ความสดใส, contrast, ความหลากหลายของสี, ฯลฯ
    if mean_val > 0.8 and mean_sat < 0.25:
        return ("แนวสุภาพ/เรียบหรู (Minimal/Smart Casual)", 95)
    elif mean_sat > 0.6 and color_variety > 10000:
        return ("แนวแฟชั่นสดใส/Pop/Colorful", 92)
    elif mean_val < 0.3 and std_val < 0.1:
        return ("แนวเท่/ดาร์กแฟชั่น (Street/Dark)", 90)
    elif mean_sat < 0.2 and std_val < 0.15:
        return ("แนว Monochrome/Classic", 88)
    elif abs(r-g)<20 and abs(g-b)<20 and mean_sat<0.3:
        return ("แนว Everyday Look / Casual", 85)
    elif mean_sat > 0.5 and mean_val > 0.5:
        return ("แนว Summer/สดใส/สายฝอ", 87)
    elif mean_sat < 0.3 and mean_val < 0.5:
        return ("แนว Earth Tone/Autumn", 83)
    elif color_variety < 2000:
        return ("แนว Minimal/เรียบง่าย", 80)
    else:
        return ("แนวแฟชั่น/Creative/Experimental", 75)

def refine_alpha_edges(image_rgba, method="morph+blur", ksize=3, blur_sigma=1.0):
    """
    ปรับขอบ alpha channel ให้คมขึ้น (morphological + blur)
    image_rgba: PIL Image RGBA
    method: "morph+blur" (default), "sharpen"
    คืนค่า PIL Image RGBA
    ถ้าไม่มี cv2 จะคืนภาพเดิม
    """
    try:
        import cv2
    except ImportError:
        # ถ้าไม่มี cv2 ให้คืนภาพเดิม ไม่ error
        return image_rgba
    arr = np.array(image_rgba)
    alpha = arr[...,3]
    kernel = np.ones((ksize,ksize), np.uint8)
    alpha_morph = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel)
    alpha_blur = cv2.GaussianBlur(alpha_morph, (ksize|1,ksize|1), blur_sigma)
    if method == "sharpen":
        sharp = cv2.addWeighted(alpha_blur, 1.5, cv2.GaussianBlur(alpha_blur, (0,0), 2), -0.5, 0)
        alpha_final = np.clip(sharp, 0, 255).astype(np.uint8)
    else:
        alpha_final = alpha_blur
    arr[...,3] = alpha_final
    return Image.fromarray(arr)

# ---------------- UI ----------------
st.set_page_config(page_title="AI Stylist", layout="centered")

# Custom CSS for modern look
st.markdown("""
    <style>
    body, .stApp {background: linear-gradient(135deg, #181c24 0%, #232936 100%); font-family: 'Sarabun', 'Prompt', 'Kanit', 'Segoe UI', sans-serif;}
    .main-title {text-align:center; font-size:2.8rem; font-weight:800; color:#f7f7fa; margin-bottom:0.3em; letter-spacing:1px; text-shadow:0 2px 12px #0006;}
    .subtitle {text-align:center; color:#b0b3b8; font-size:1.25rem; margin-bottom:2.2em; letter-spacing:0.5px;}
    .color-card, .score-card, .style-card {backdrop-filter: blur(2px); border-radius:16px; box-shadow:0 4px 24px #0003; padding:1.5em 2em; margin-bottom:2em; border:1.5px solid #23293655;}
    .color-card {background:rgba(35,41,54,0.95);}
    .score-card {background:linear-gradient(90deg,#232936,#2e3a4d 80%);}
    .style-card {background:linear-gradient(90deg,#232936,#3a2e4d 80%);}
    .color-label {font-weight:700; font-size:1.15rem; color:#f7f7fa; letter-spacing:0.5px;}
    .color-hex {font-family:monospace; font-size:1.1rem; color:#b0b3b8;}
    .footer {text-align:center; color:#b0b3b8; font-size:1.05rem; margin-top:2.5em; letter-spacing:0.5px;}
    .stButton>button {background:linear-gradient(90deg,#3a3f51,#232936); color:#fff; border-radius:10px; font-weight:700; border:none; padding:0.6em 2em; font-size:1.1rem; transition:0.2s;}
    .stButton>button:hover {background:#1a1d25; color:#ffe082;}
    .stFileUploader label {font-size:1.1rem; color:#f7f7fa; font-weight:600;}
    .stColorPicker {margin-bottom:1.2em;}
    .stImage {border-radius:14px; box-shadow:0 2px 16px #0005; margin-bottom:1.5em;}
    h4 {font-weight:700; color:#ffe082; letter-spacing:0.5px;}
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Sarabun:wght@400;700;800&family=Prompt:wght@400;700&family=Kanit:wght@400;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">👗 AI Stylist</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">อัปโหลดรูปเสื้อผ้าของคุณเพื่อรับคำแนะนำแฟชั่นและโทนสีที่เหมาะกับคุณ</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📸 เลือกรูปภาพการแต่งตัวของคุณ", type=["jpg", "jpeg", "png"], label_visibility="visible")

if uploaded_file:
    image_bytes = uploaded_file.read()
    # --- ลบพื้นหลังด้วย rembg (AI-based) + refine ขอบให้เนียน ---
    image_nobg_rgba = remove_background_bytes(image_bytes)
    if image_nobg_rgba is None:
        # fallback manual_remove_bg + refine
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_nobg_rgba = manual_remove_bg(image, '#ffffff', 255)
        image_nobg_rgba = refine_alpha_edges(image_nobg_rgba, method="morph+blur", ksize=3, blur_sigma=0.8)
    image_nobg = image_nobg_rgba.convert("RGB")
    # --- Human Parsing: แยกเฉพาะเสื้อผ้า (upper/lower) ---
    mask = segment_clothes(image_nobg)
    if mask is not None:
        # 5=upper-clothes, 6=pants (LIP dataset)
        clothes_rgba = extract_part(image_nobg, mask, part_labels=[5,6])
        # ใช้เฉพาะ pixel ที่เป็นเสื้อผ้า (alpha>0)
        image_for_color = clothes_rgba
    else:
        image_for_color = image_nobg
    # แสดงภาพต้นฉบับที่ผู้ใช้อัปโหลด
    st.image(Image.open(io.BytesIO(image_bytes)), caption="ภาพต้นฉบับที่คุณอัปโหลด", use_container_width=True)

    # --- วิเคราะห์ dominant color รวมของตัวคน (เฉพาะเสื้อผ้า) ---
    colors = get_dominant_colors(image_for_color)
    main_color = colors[0]
    second_color = colors[1] if len(colors) > 1 else colors[0]
    main_hex = rgb_to_hex(main_color)
    second_hex = rgb_to_hex(second_color)
    main_name = get_color_name(main_color)
    second_name = get_color_name(second_color)

    # --- UI: แสดง dominant color รวม ---
    st.markdown('<div class="color-card">', unsafe_allow_html=True)
    st.markdown("<h4 style='margin-bottom:0.5em;'>🎨 สีหลัก และสีรอง</h4>", unsafe_allow_html=True)
    cols = st.columns(2)
    with cols[0]:
        st.markdown(f"<span class='color-label'>สีหลัก</span>: <span style='color:{main_hex};font-weight:700;'>{main_hex}</span> <span class='color-hex'>{tuple(main_color)}</span> <b>{main_name}</b>", unsafe_allow_html=True)
        st.color_picker("สีหลัก", main_hex, key="main_color_upper")
    with cols[1]:
        st.markdown(f"<span class='color-label'>สีรอง</span>: <span style='color:{second_hex};font-weight:700;'>{second_hex}</span> <span class='color-hex'>{tuple(second_color)}</span> <b>{second_name}</b>", unsafe_allow_html=True)
        st.color_picker("สีรอง", second_hex, key="second_color_lower")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- ความเข้ากันของสี (หลัก-รอง) ---
    st.markdown('<div class="score-card">', unsafe_allow_html=True)
    st.markdown("<h4 style='margin-bottom:0.5em;'>✅ ความเข้ากันของสี (หลัก-รอง)</h4>", unsafe_allow_html=True)
    feedback, score = evaluate_color_match([main_color, second_color])
    def color_theory_tag(main, second):
        import colorsys
        h1, s1, v1 = colorsys.rgb_to_hsv(*(np.array(main)/255.0))
        h2, s2, v2 = colorsys.rgb_to_hsv(*(np.array(second)/255.0))
        dh = abs(h1-h2)
        if dh < 0.08 or dh > 0.92:
            return "(Color Harmony: Analogous)"
        elif abs(dh-0.5) < 0.08:
            return "(Color Harmony: Complementary)"
        elif abs(dh-1/3) < 0.08 or abs(dh-2/3) < 0.08:
            return "(Color Harmony: Triadic)"
        elif abs(dh-0.25) < 0.06 or abs(dh-0.75) < 0.06:
            return "(Color Harmony: Tetradic)"
        else:
            return "(Color Harmony: Custom)"
    theory = color_theory_tag(main_color, second_color)
    st.markdown(f"<b>คะแนน:</b> <span style='font-size:1.5rem;color:#1a7f6b'>{score}/100</span>", unsafe_allow_html=True)
    st.markdown(f"<b>คำแนะนำ:</b> {feedback} <span style='color:#ffe082;font-size:1.05rem;'>{theory}</span>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- ทำนายสไตล์ (dominant color รวม) ---
    st.markdown('<div class="style-card">', unsafe_allow_html=True)
    st.markdown("<h4 style='margin-bottom:0.5em;'>💡 สไตล์ที่ AI คิดว่าใช่คุณ</h4>", unsafe_allow_html=True)
    style, style_prob = advanced_predict_style([main_color, second_color], image)
    st.markdown(f"<b>{style}</b> <span style='color:#b0b3b8;font-size:1.1rem;'>(ความมั่นใจ {style_prob:.1f}%)</span>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="footer">👨‍💻 พัฒนาโดย นักศึกษาสาขาเทคโนโลยีสารสนเทศ</div>', unsafe_allow_html=True)

    with st.expander("🎓 ทฤษฎีสีในแฟชั่น: ประเภทการจับคู่สี (Color Harmony Theory)"):
        st.markdown("""
        <div style='text-align:center; color:#ffe082; font-size:1.18rem; font-weight:700; margin-bottom:0.7em;'>ประเภทการจับคู่สีหลักในแฟชั่น</div>
        <ol style='color:#f7f7fa; font-size:1.08rem;'>
        <li><b>Complementary (สีตรงข้าม)</b><br>
        <span style='color:#b0b3b8;'>สีที่อยู่ตรงข้ามกันบนวงล้อสี<br>สร้างความโดดเด่นและความคมชัด<br>เหมาะกับการออกงานที่ต้องการความโดดเด่น</span></li>
        <li><b>Analogous (สีใกล้เคียง)</b><br>
        <span style='color:#b0b3b8;'>สีที่อยู่ใกล้กันบนวงล้อสี<br>สร้างความกลมกลืนและดูนุ่มนวล<br>เหมาะกับลุคลำลองและการทำงาน</span></li>
        <li><b>Triadic (สีสามเหลี่ยม)</b><br>
        <span style='color:#b0b3b8;'>สีที่ห่างกันเท่าๆ กันบนวงล้อสี<br>สร้างความสมดุลและน่าสนใจ<br>เหมาะกับการแต่งตัวที่ต้องการความสร้างสรรค์</span></li>
        </ol>
        """, unsafe_allow_html=True)
