# app.py  (UI Polished, Thai-only, IoU locked, Style Advice + Color Wheel CW + Achroma neutrals)
# ─────────────────────────────────────────────────────────────────────────────
# Streamlit app: อัปโหลดรูป → YOLOv8 (seg) แยกชิ้นเสื้อผ้า → ดูดสีหลัก
# → แนะนำสี + ให้คะแนนความเข้ากัน (Monochrome / Analogous / Triadic / Complementary)
# + วงล้อสี "หมุนตามเข็ม" พร้อมจุดองศา (เส้นแยกสีตามทฤษฎี + legend) และทำให้จุดสีฐานเด่น
# สูตรคะแนนรวม: Overall = 100 × f(Δh°) × g(ΔE₀₀)
# ใช้โมเดล best.pt (YOLOv8-seg)
# ─────────────────────────────────────────────────────────────────────────────

# ===== 1) IMPORTS & GLOBAL CONFIG =====
import io, math, base64, textwrap
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from sklearn.cluster import KMeans
import streamlit as st

st.set_page_config(
    page_title="AI Stylist | Color Recommender",
    page_icon="search.png",
    layout="wide"
)

MODEL_PATH = "best.pt"
CONF_TH  = 0.25
IMG_SIZE = 640
IOU_TH   = 0.45  # locked

EN2TH = {
    "long_sleeved_dress":   "ชุดเดรส",
    "long_sleeved_outwear": "เสื้อคลุมแขนยาว",
    "long_sleeved_shirt":   "เสื้อแขนยาว",
    "short_sleeved_dress":  "ชุดเดรส",
    "short_sleeved_outwear":"เสื้อคลุมแขนสั้น",
    "short_sleeved_shirt":  "เสื้อแขนสั้น",
    "shorts":               "กางเกง",
    "skirt":                "กระโปรง",
    "sling":                "เสื้อ",
    "sling_dress":          "ชุดเดรส",
    "trousers":             "กางเกง",
    "vest":                 "เสื้อ",
    "vest_dress":           "ชุดเดรส",
}

# ===== 2) THEME & STYLES =====
st.markdown("""
<style>
:root{
  --bg:#0f1216; --panel:#141922; --muted:#9aa4b2; --brand:#74c0fc; --chip:#1b2330;
  --card:#11161d; --border:#1f2632; --ok:#2f9e44;
  --font:'Inter',ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,'Helvetica Neue',Arial;
}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg); color:#e6e9ef; font-family:var(--font);}
section[data-testid="stSidebar"]{background:var(--panel); border-right:1px solid var(--border);}
.block-container{padding-top:2rem; padding-bottom:3rem;}
h1,h2,h3{letter-spacing:.3px}
.small{font-size:.9rem;color:var(--muted)}
.card{background:var(--card); border:1px solid var(--border); border-radius:16px; padding:18px}
.swatch{border-radius:12px; border:1px solid var(--border); height:56px}
.hex{font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;}
.badge{
  display:inline-flex; align-items:center; gap:.45rem; padding:.28rem .6rem;
  border-radius:999px; background:var(--chip); border:1px solid var(--border); margin:.2rem .35rem .2rem 0;
  font-size:.85rem; color:#d1d7e0;
}
.footer{opacity:.6; font-size:.9rem; margin-top:.75rem}
.stTabs [data-baseweb="tab-list"]{gap:8px}
.stTabs [data-baseweb="tab"]{background:var(--chip); border-radius:10px; padding:10px 14px; border:1px solid var(--border);}
img{border-radius:12px}
.kpi{display:grid; grid-template-columns: 1fr 1fr; gap:12px; margin-bottom:12px}
.kpi > div{background:#0e141b; border:1px solid var(--border); border-radius:12px; padding:10px 12px}
.kpi .num{font-size:1.4rem; font-weight:700}
</style>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

st.markdown("""
<style>
.row-palette{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:6px}
.chip{height:40px;border-radius:10px;border:1px solid var(--border)}
.label{font-size:.85rem;color:#9aa4b2;margin-top:6px}
.hex-sm{font-family:ui-monospace;font-size:.85rem;opacity:.85}
</style>
""", unsafe_allow_html=True)

# ===== 3) COLOR UTILITIES =====
def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def hsv01_to_rgb(h, s, v):
    import colorsys
    r,g,b = colorsys.hsv_to_rgb(_clamp01(h), _clamp01(s), _clamp01(v))
    R,G,B = int(round(r*255)), int(round(g*255)), int(round(b*255))
    return (R,G,B), f"#{R:02X}{G:02X}{B:02X}"

def rgb255_to_hsv01(rgb):
    import colorsys
    r,g,b = rgb
    return colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)

def color_harmonies_from_rgb(rgb):
    # ใช้ฮิว HSV (0–360) แต่การแสดงบนวงล้อหมุนตามเข็ม
    h,s,v = rgb255_to_hsv01(rgb)
    comp = hsv01_to_rgb(h + .5, s, v)      # +180°
    ana1 = hsv01_to_rgb(h + .0833, s, v)   # +30°
    ana2 = hsv01_to_rgb(h - .0833, s, v)   # -30°
    tri1 = hsv01_to_rgb(h + 1/3, s, v)     # +120°
    tri2 = hsv01_to_rgb(h - 1/3, s, v)     # -120°
    return {
        "base": (rgb, f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"),
        "complementary": comp,
        "analogous": [ana1, ana2],
        "triadic": [tri1, tri2]
    }

def _filter_pixels(pixels: np.ndarray) -> np.ndarray:
    if pixels.size == 0:
        return pixels
    R,G,B = pixels[:,0], pixels[:,1], pixels[:,2]
    not_white = ~((R>235)&(G>235)&(B>235))
    not_black = ~((R<15)&(G<15)&(B<15))
    not_gray  = ~((np.abs(R-G)<12) & (np.abs(G-B)<12))
    return pixels[not_white & not_black & not_gray]

def dominant_color_from_masked_pixels(pixels: np.ndarray, n_clusters: int = 3):
    if pixels.size == 0:
        return (200,200,200), "#C8C8C8"
    px = _filter_pixels(pixels)
    if px.size == 0:
        px = pixels
    if len(px) > 6000:
        idx = np.random.choice(len(px), 6000, replace=False)
        px = px[idx]
    km = KMeans(n_clusters=n_clusters, n_init=5, random_state=0)
    labels = km.fit_predict(px.astype(np.float32))
    centers = km.cluster_centers_.astype(int)
    _, counts = np.unique(labels, return_counts=True)
    rgb = tuple(int(x) for x in centers[int(np.argmax(counts))])
    return rgb, f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"

def color_tile(title: str, items):
    st.markdown(f"**{title}**")
    cols = st.columns(len(items))
    for c,(rgb,hx) in zip(cols, items):
        with c:
            st.markdown(f"<div class='swatch' style='background:{hx}'></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='hex' style='margin-top:.4rem'>{hx}</div>", unsafe_allow_html=True)


# ===== 4) COLOR DIFFERENCE & HUE HELPERS =====
def _pivot_rgb(c):
    c = c / 255.0
    return ((c + 0.055)/1.055)**2.4 if c > 0.04045 else c/12.92

def rgb_to_lab(rgb):
    r,g,b = rgb
    r, g, b = _pivot_rgb(r), _pivot_rgb(g), _pivot_rgb(b)
    X = r*0.4124564 + g*0.3575761 + b*0.1804375
    Y = r*0.2126729 + g*0.7151522 + b*0.0721750
    Z = r*0.0193339 + g*0.1191920 + b*0.9503041
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    def f(t): return t**(1/3) if t > 0.008856 else (7.787*t + 16/116)
    fx, fy, fz = f(X/Xn), f(Y/Yn), f(Z/Zn)
    L = 116*fy - 16
    a = 500*(fx - fy)
    b = 200*(fy - fz)
    return (L,a,b)

def deltaE00(rgb1, rgb2):
    L1,a1,b1 = rgb_to_lab(rgb1); L2,a2,b2 = rgb_to_lab(rgb2)
    avg_Lp = (L1 + L2)/2.0
    C1 = math.sqrt(a1*a1 + b1*b1); C2 = math.sqrt(a2*a2 + b2*b2)
    avg_C1C2 = (C1 + C2)/2.0
    G = 0.5 * (1 - math.sqrt((avg_C1C2**7) / (avg_C1C2**7 + 25**7)))
    a1p = (1+G) * a1; a2p = (1+G) * a2
    C1p = math.sqrt(a1p*a1p + b1*b1); C2p = math.sqrt(a2p*a2p + b2*b2)
    avg_Cp = (C1p + C2p)/2.0
    h1p = math.degrees(math.atan2(b1, a1p)) % 360; h2p = math.degrees(math.atan2(b2, a2p)) % 360
    if abs(h1p - h2p) > 180: avg_hp = (h1p + h2p + 360)/2.0
    else: avg_hp = (h1p + h2p)/2.0
    T = (1 - 0.17*math.cos(math.radians(avg_hp - 30))
         + 0.24*math.cos(math.radians(2*avg_hp))
         + 0.32*math.cos(math.radians(3*avg_hp + 6))
         - 0.20*math.cos(math.radians(4*avg_hp - 63)))
    deltahp = h2p - h1p
    if abs(deltahp) > 180:
        deltahp -= 360 * np.sign(deltahp)
    delta_Lp = L2 - L1
    delta_Cp = C2p - C1p
    delta_Hp = 2*math.sqrt(C1p*C2p)*math.sin(math.radians(deltahp/2.0))
    Sl = 1 + (0.015*(avg_Lp - 50)**2)/math.sqrt(20 + (avg_Lp - 50)**2)
    Sc = 1 + 0.045*avg_Cp
    Sh = 1 + 0.015*avg_Cp*T
    Rt = -2*math.sqrt((avg_Cp**7)/(avg_Cp**7 + 25**7)) * math.sin(math.radians(60*math.exp(-((avg_hp-275)/25)**2)))
    kL = kC = kH = 1
    return math.sqrt((delta_Lp/(kL*Sl))**2 + (delta_Cp/(kC*Sc))**2 + (delta_Hp/(kH*Sh))**2 + Rt*(delta_Cp/(kC*Sc))*(delta_Hp/(kH*Sh)))

def hue_deg(rgb):
    # ส่งคืนองศาฮิว 0–360 (ตัวเลขเดียวกับ HSV ปกติ) แต่ถูกแสดงบนวงล้อ "ตามเข็ม"
    h, s, v = rgb255_to_hsv01(rgb)
    return (h*360.0) % 360.0, s, v

def circ_diff_deg(h1, h2):
    d = abs((h1 - h2) % 360.0)
    return 360.0 - d if d > 180 else d

# ===== 5) TH UI HELPERS & SCORING =====
SCHEME_TH = {"complementary":"คู่ตรงข้าม", "analogous":"ใกล้เคียง", "triadic":"สามเหลี่ยม"}
TARGET_DEG = {"Monochrome":0, "Analogous":30, "Triadic":120, "Complementary":180}

SCHEME_STYLE = {
    "base":           {"name": "สีหลัก",                 "hex": "#DF0000"},
    "analogous":      {"name": "ใกล้เคียง (±30°)",      "hex": "#1ADF00"},
    "triadic":        {"name": "สามเหลี่ยม (±120°)",    "hex": "#FFA600"},
    "complementary":  {"name": "คู่ตรงข้าม (180°)",     "hex": "#3B2DFFFF"},
}

# พาเล็ตกลางสำหรับชิ้นที่ "ไม่มีฮิว"
NEUTRAL_RECS = [
    ("#FFFFFF", "ขาว"),
    ("#BFBFBF", "เทา"),
    ("#000000", "ดำ"),
]

ACH_S_TH   = 0.05
WHITE_V_TH = 0.85
BLACK_V_TH = 0.15

def achromatic_label_from_sv(s, v):
    if s >= ACH_S_TH: return None
    if v >= WHITE_V_TH: return "ขาว"
    if v <= BLACK_V_TH: return "ดำ"
    return "เทา"

def hue_name_th(deg: float, s: float = 1.0) -> str:
    anchors = [
        (0, "แดง"), (30, "ส้ม"), (60, "เหลือง"),
        (90, "เหลือง-เขียว"), (120, "เขียว"), (150, "เขียวอมฟ้า"),
        (180, "ไซแอน"), (210, "ฟ้า"), (240, "น้ำเงิน"),
        (270, "ม่วงน้ำเงิน"), (300, "ม่วง-ชมพู"), (330, "ชมพู-โรส"),
    ]
    best_name, best_d = anchors[0][1], 1e9
    for a, name in anchors:
        d = abs((deg - a + 180) % 360 - 180)
        if d < best_d: best_d, best_name = d, name
    return best_name

T_MONO = 60.0; T_ANA = 60.0; T_TRI = 90.0; T_COMP = 90.0; DE_SCALE = 55.0
TIP_LIMIT = 3

def score_rules(rgb1, rgb2):
    h1, _, _ = hue_deg(rgb1); h2, _, _ = hue_deg(rgb2)
    dh = circ_diff_deg(h1, h2)
    mono = 100 * (1 - min(dh / T_MONO, 1.0))
    ana  = 100 * (1 - min(abs(dh - 30.0) / T_ANA, 1.0))
    d_tri = min(abs(dh - 120.0), abs(dh - 240.0))
    tri  = 100 * (1 - min(d_tri / T_TRI, 1.0))
    d_comp = abs(dh - 180.0)
    comp = 100 * (1 - min(d_comp / T_COMP, 1.0))
    per_rule = {"Monochrome":float(np.clip(mono,0,100)),
                "Analogous":float(np.clip(ana,0,100)),
                "Triadic":float(np.clip(tri,0,100)),
                "Complementary":float(np.clip(comp,0,100))}
    primary = max(per_rule, key=per_rule.get)
    de = deltaE00(rgb1, rgb2); g = 1.0 - min(de / DE_SCALE, 1.0)
    f = max(per_rule.values()) / 100.0
    overall = 100.0 * f * g
    return per_rule, float(overall), primary, {"f":f,"g":g,"dh":dh,"de":de}

def advice_text(primary: str, primary_th: str, score: float, dh: float, de00: float):
    if score >= 90: level = "ยอดเยี่ยม"
    elif score >= 80: level = "ดีเยี่ยม"
    elif score >= 70: level = "ดีมาก"
    elif score >= 60: level = "ดี"
    elif score >= 50: level = "พอใช้"
    elif score >= 40: level = "ควรปรับเล็กน้อย"
    else: level = "ควรปรับ"
    head = f"แนวโน้มหลัก: {primary_th} → ระดับความเข้ากัน **{level}**"
    base_map = {
        "Monochrome":["เพิ่มมิติด้วย texture/วัสดุเล็กน้อย","ไล่เฉดสว่าง-เข้มให้ไม่กลืนกัน","ใช้เครื่องประดับสีโลหะเป็นจุดโฟกัส"],
        "Analogous":["เลือก 1 สีหลัก + 1–2 สีรองแบบจาง","แทรกโทนกลาง (ขาว/เทา/ครีม) ให้ภาพเนียน","เล่นเลเยอร์เฉดอ่อน-กลาง-เข้ม"],
        "Triadic":["จัดสัดส่วน 60-30-10 คุมความสด","ลดความอิ่มสีของสีรองลงนิดถ้าดูแรงไป","ใช้ลายเส้น/บล็อกเล็กๆ ให้บาลานซ์"],
        "Complementary":["คุมสัดส่วนประมาณ 70-30 ให้สีรองเป็นจุดเน้น","ลดความอิ่มสีฝั่งรองเล็กน้อยให้ดูหรู","แทรกโทนกลาง (ขาว/เทา/น้ำตาลอ่อน)"],
    }
    if primary == "Monochrome": target_err = abs(dh - 0)
    elif primary == "Analogous": target_err = abs(dh - 30)
    elif primary == "Complementary": target_err = abs(dh - 180)
    else: target_err = min(abs(dh - 120), abs(dh - 240))
    dynamic = None
    if target_err > 40: dynamic = "ขยับเฉดของชิ้นรองเข้าเป้า ~15–30°"
    elif de00 >= 50: dynamic = "ลดความต่างสว่าง/อิ่มสีของสีรองลงเล็กน้อย"
    elif de00 <= 10: dynamic = "เพิ่มจุดเน้นที่สว่างขึ้น/เข้มขึ้นเล็กน้อย"
    tips = base_map[primary][:2]
    if dynamic: tips.append(dynamic)
    return head, tips[:TIP_LIMIT]

def rec_tile(rec_hex: str, label_th: str):
    st.markdown(
        f"""
        <div style="height:44px;border-radius:10px;border:1px solid var(--border);background:{rec_hex}"></div>
        <div class="small" style="margin-top:6px">{label_th} · {rec_hex}</div>
        """, unsafe_allow_html=True
    )

def small_swatch(hexcode: str, note_html: str = ""):
    st.markdown(
        f"""
        <div style="height:36px;border-radius:8px;border:1px solid var(--border);background:{hexcode}"></div>
        <div class="small" style="margin-top:6px">ฐานสิบหก: <code>{hexcode}</code>{note_html}</div>
        """, unsafe_allow_html=True
    )

# ===== 6) COLOR WHEEL UTILS (Clockwise rendering) =====
def _hex_to_rgb(hx: str):
    hx = hx.lstrip("#"); return (int(hx[0:2],16), int(hx[2:4],16), int(hx[4:6],16))

def hue_deg_from_hex(hx: str) -> float:
    rgb = _hex_to_rgb(hx)
    h, s, v = rgb255_to_hsv01(rgb)
    return (h * 360.0) % 360.0  # ตัวเลขเดียวกับ HSV; วงล้อของเราวาดตามเข็ม

def _hsv_to_rgb_np(h, s, v):
    i = np.floor(h*6).astype(int) % 6
    f = h*6 - np.floor(h*6)
    p = v*(1-s); q = v*(1-f*s); t = v*(1-(1-f)*s)
    r = np.select([i==0,i==1,i==2,i==3,i==4,i==5],[v,q,p,p,t,v], default=v)
    g = np.select([i==0,i==1,i==2,i==3,i==4,i==5],[t,v,v,q,p,p], default=v)
    b = np.select([i==0,i==1,i==2,i==3,i==4,i==5],[p,p,t,v,v,q], default=v)
    return r,g,b

def _hex_to_rgba(hx: str, alpha: int = 230):
    r, g, b = _hex_to_rgb(hx)
    return (r, g, b, int(alpha))

def make_color_wheel_image(markers, size=340, bg_rgb=(15,18,22)):
    """วาดภาพวงล้อสีแบบหมุนตามเข็ม + เส้นทฤษฎี + ไฮไลต์สีฐาน"""
    N=size; cx=cy=N//2; rmax=N//2 - 6
    img=np.zeros((N,N,3),dtype=np.uint8); img[:]=np.array(bg_rgb,dtype=np.uint8)

    # พื้นหลังวงล้อ (CW)
    y,x=np.ogrid[-cy:N-cy,-cx:N-cx]
    rr=np.sqrt(x*x+y*y); mask=rr<=rmax
    ang=(np.degrees(np.arctan2(y,x))%360)/360.0   # CW
    sat=np.clip(rr/rmax,0,1); val=np.ones_like(ang)
    r,g,b=_hsv_to_rgb_np(ang,sat,val)
    rgb=(np.stack([r,g,b],axis=-1)*255).astype(np.uint8)
    canvas=img.copy(); canvas[mask]=rgb[mask]

    pil=Image.fromarray(canvas,"RGB")
    draw=ImageDraw.Draw(pil, "RGBA")
    draw.ellipse([cx-rmax,cy-rmax,cx+rmax,cy+rmax], outline=(31,38,50,255), width=2)

    others = [m for m in markers if not m.get("is_base")]
    bases  = [m for m in markers if m.get("is_base")]

    def _draw_marker(m, is_base=False):
        theta=math.radians((m["deg"]%360.0))
        rx=cx+int(rmax*math.cos(theta))
        ry=cy+int(rmax*math.sin(theta))   # CW

        group = m.get("group", "base")
        col_hex = SCHEME_STYLE.get(group, SCHEME_STYLE["base"])["hex"]
        line_col = _hex_to_rgba(col_hex, 230 if not is_base else 180)
        line_w   = 3 if not is_base else 4

        draw.line([cx,cy,rx,ry], fill=line_col, width=line_w)

        rdot = 10 if is_base else 7
        dot_fill  = _hex_to_rgb(m["hex"])
        draw.ellipse([rx-rdot,ry-rdot,rx+rdot,ry+rdot], fill=dot_fill, outline=line_col, width=3 if is_base else 2)

        if is_base:
            halo_r = rdot + 6
            draw.ellipse([rx-halo_r, ry-halo_r, rx+halo_r, ry+halo_r], outline=(255,77,77,120), width=5)

    for m in others: _draw_marker(m, is_base=False)
    for m in bases:  _draw_marker(m, is_base=True)

    return pil

# ===== 7) ITEM CARD RENDER =====
def render_item_card_th(item_label: str, conf_i: float, har: dict, deg, hue_name, is_achromatic: bool = False, default_open: bool = False):
    base_hex = har["base"][1]
    comp_hex = har.get("complementary", (None, None))[1]
    ana_hexes = [x[1] for x in har.get("analogous", [])]
    tri_hexes = [x[1] for x in har.get("triadic", [])]

    with st.expander(f"{item_label} · ความมั่นใจ {conf_i:.2f}", expanded=default_open):
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**สีหลัก**")
        note = f" · ฮิว ≈ <b>{deg:.1f}°</b> · {hue_name}" if (hue_name and deg is not None) else f" · {hue_name}"
        small_swatch(base_hex, note_html=note)

        st.markdown("**จับคู่สี (สีที่แนะนำ)**")
        if is_achromatic:
            c1,c2,c3 = st.columns(3)
            with c1: rec_tile(NEUTRAL_RECS[0][0], NEUTRAL_RECS[0][1])  # ขาว
            with c2: rec_tile(NEUTRAL_RECS[1][0], NEUTRAL_RECS[1][1])  # เทา
            with c3: rec_tile(NEUTRAL_RECS[2][0], NEUTRAL_RECS[2][1])  # ดำ
        else:
            c1,c2,c3=st.columns(3)
            with c1: rec_tile(comp_hex, SCHEME_TH["complementary"])
            with c2: rec_tile(ana_hexes[0], SCHEME_TH["analogous"])
            with c3: rec_tile(tri_hexes[0], SCHEME_TH["triadic"])

        st.markdown("</div>", unsafe_allow_html=True)

# ===== 8) MODEL LOADING =====
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    return YOLO(path)

def _ensure_rgb_like_ref(annot_np: np.ndarray, ref_np: np.ndarray) -> np.ndarray:
    if annot_np.ndim != 3 or annot_np.shape[2] != 3: return annot_np
    try:
        if annot_np.shape == ref_np.shape:
            a=annot_np.astype(np.float32); r=ref_np.astype(np.float32)
            mse_id=np.mean((a-r)**2); mse_sw=np.mean((a[:,:,::-1]-r)**2)
            return annot_np if mse_id <= mse_sw else annot_np[:,:,::-1]
    except Exception:
        pass
    if np.mean(ref_np[...,2])>np.mean(ref_np[...,0]) and np.mean(annot_np[...,0])>np.mean(annot_np[...,2]):
        return annot_np[:,:,::-1]
    return annot_np

def _class_color(cid: int) -> tuple[int,int,int]:
    palette = [
        (199, 38, 255), (255, 165, 0), (0, 180, 255),
        (40, 200, 120), (255, 90, 90), (255, 220, 0)
    ]
    return palette[cid % len(palette)]

def draw_boxes_no_masks(img_np, res, names):
    pil = Image.fromarray(img_np.copy())
    draw = ImageDraw.Draw(pil)
    font = ImageFont.load_default()
    if res.boxes is None or len(res.boxes) == 0:
        return np.array(pil)
    xyxys = res.boxes.xyxy.cpu().numpy().astype(int)
    cids  = res.boxes.cls.cpu().numpy().astype(int) if res.boxes.cls is not None else [-1]*len(xyxys)
    confs = res.boxes.conf.cpu().numpy() if res.boxes.conf is not None else [0.0]*len(xyxys)
    for (x1,y1,x2,y2), cid, conf in zip(xyxys, cids, confs):
        color = _class_color(max(cid,0))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label_en = names.get(int(cid), str(int(cid)))
        label = f"{label_en} {conf:.2f}"
        try:
            bbox = draw.textbbox((0,0), label, font=font)
            tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        except Exception:
            tw, th = font.getsize(label)
        pad = 4
        y0 = max(0, y1 - th - 2*pad)
        draw.rectangle([x1, y0, x1 + tw + 2*pad, y0 + th + 2*pad], fill=color)
        text_col = (0,0,0) if sum(color) > 550 else (255,255,255)
        draw.text((x1 + pad, y0 + pad), label, fill=text_col, font=font)
    return np.array(pil)

# ===== 10) SIDEBAR =====
with st.sidebar:
    st.header("ตั้งค่า")
    model_path = st.text_input("ตำแหน่งโมเดล (.pt)", MODEL_PATH)
    conf  = st.slider("Confidence", 0.05, 0.85, CONF_TH, 0.01)
    imgsz = st.select_slider("Image size (px)", options=[320,480,640,800], value=IMG_SIZE)

# ===== 11) HEADER =====
st.markdown("""
<div class="card" style="margin-bottom:18px">
  <h1 style="margin:0 0 6px">🧥 AI Stylist </h1>
  <div class="small">อัปโหลดรูป ระบบจะแยกชิ้นเสื้อผ้าด้วย YOLOv8 → ดูดสีหลัก → ให้คะแนนความเข้ากัน และแนะนำโทนสี</div>
</div>
""", unsafe_allow_html=True)

# ===== 12) LOAD MODEL =====
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"โหลดโมเดลไม่สำเร็จ: {e}")
    st.stop()

if getattr(model, "task", None) != "segment":
    st.warning("โมเดลนี้อาจไม่ใช่ **segmentation** (YOLOv8-seg) — ผลลัพธ์อาจไม่มี masks")

# ===== 13) UPLOAD IMAGE =====
uploaded = st.file_uploader("อัปโหลดรูปภาพ", type=["jpg","jpeg","png","webp"])
if not uploaded:
    st.info("ลากรูปมาวาง หรือกดเลือกไฟล์ เพื่อเริ่มวิเคราะห์ ✨")
    st.stop()

img_pil = Image.open(uploaded).convert("RGB")
img_np = np.array(img_pil)

# ===== 14) INFERENCE =====
with st.spinner("กำลังตรวจจับเสื้อผ้า…"):
    res = model.predict(img_np, conf=conf, iou=IOU_TH, imgsz=imgsz, verbose=False)[0]

# ===== 15) ANNOTATED IMAGE =====
try:
    annot_rgb = res.plot(masks=False, boxes=True, labels=True)
except TypeError:
    annot_rgb = draw_boxes_no_masks(img_np, res, model.names)
annot_rgb = _ensure_rgb_like_ref(annot_rgb, img_np)

# ===== 16) TABS =====
tab1, tab2 = st.tabs(["ภาพรวม", "ผลลัพธ์รายชิ้น"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**ภาพต้นฉบับ**"); st.image(img_pil, use_column_width=True)
    with c2:
        st.markdown("**ภาพใส่กรอบตรวจจับ**"); st.image(annot_rgb, use_column_width=True)

    names = model.names
    if res.boxes is not None and len(res.boxes) > 0:
        cls_ids = [int(x) for x in res.boxes.cls.cpu().tolist()]
        uniq_th = []
        for cid in cls_ids:
            en = names.get(cid, str(cid)); th = EN2TH.get(en, en)
            if th not in uniq_th: uniq_th.append(th)
        st.markdown("<div style='margin-top:10px'>รายการที่ตรวจจับได้:</div>", unsafe_allow_html=True)
        st.markdown("".join(f"<span class='badge'>{th}</span>" for th in uniq_th), unsafe_allow_html=True)
    else:
        st.info("ไม่พบชิ้นเสื้อผ้าในภาพนี้")

    buf = io.BytesIO(); Image.fromarray(annot_rgb.astype(np.uint8)).save(buf, format="PNG")
    st.download_button("⬇️ ดาวน์โหลดภาพผลลัพธ์ (PNG)", buf.getvalue(), "result.png", "image/png")

with tab2:
    if res.masks is None or res.masks.data is None or len(res.masks.data) == 0:
        st.warning("ภาพนี้ไม่มี mask ของเสื้อผ้า (ลองภาพอื่นดูนะ)")
        st.stop()

    masks = res.masks.data.cpu().numpy()
    H, W = img_np.shape[:2]
    boxes = res.boxes
    names = model.names

    pieces = []
    for i in range(len(masks)):
        cls_id  = int(boxes.cls[i].item()) if boxes.cls is not None else -1
        conf_i  = float(boxes.conf[i].item()) if boxes.conf is not None else 0.0
        name_en = names.get(cls_id, f"id:{cls_id}")
        name_th = EN2TH.get(name_en, name_en)

        m_img = Image.fromarray((masks[i] > 0.5).astype(np.uint8)*255).resize((W, H), Image.NEAREST)
        mask_bool = np.array(m_img).astype(bool)
        area = int(mask_bool.sum())

        pixels = img_np[mask_bool]
        dom_rgb, dom_hex = dominant_color_from_masked_pixels(pixels, n_clusters=3)
        har = color_harmonies_from_rgb(dom_rgb)

        deg, s, v = hue_deg(dom_rgb)
        ach = achromatic_label_from_sv(s, v)
        if ach is not None:
            deg_disp = None; hue_label = f"ไม่มีฮิว · {ach}"
            is_ach = True
        else:
            deg_disp = deg; hue_label = hue_name_th(deg, s)
            is_ach = False

        pieces.append({
            "name_en": name_en, "name_th": name_th, "conf": conf_i, "area": area,
            "dom_rgb": dom_rgb, "dom_hex": dom_hex, "har": har,
            "hue_deg": deg_disp, "hue_name": hue_label, "is_achromatic": is_ach
        })

    # ── คำแนะนำสไตล์ + วงล้อสี
    st.markdown("### 🧾 คำแนะนำสไตล์")
    if len(pieces) >= 2:
        pieces_sorted = sorted(pieces, key=lambda x: x["area"], reverse=True)
        p1, p2 = pieces_sorted[0], pieces_sorted[1]

        per_rule, overall, primary, dbg = score_rules(p1["dom_rgb"], p2["dom_rgb"])
        primary_th = {
            "Monochrome":"โทนเดียว (Monochrome)", "Analogous":"เฉดใกล้เคียง (Analogous)",
            "Triadic":"สามเฉดสมดุล (Triadic)", "Complementary":"คู่ตรงข้าม (Complementary)"
        }[primary]

        dh, de00 = dbg["dh"], dbg["de"]
        f_pct, g_pct = dbg["f"]*100, dbg["g"]*100
        target = TARGET_DEG[primary]

        head, tips = advice_text(primary, primary_th, overall, dh, de00)
        tips_html = "<ul style='margin:6px 0 0 18px'>" + "".join(f"<li>{t}</li>" for t in tips) + "</ul>"

        # เตรียมรูปวงล้อ
        base_hex = p1["dom_hex"]
        comp_hex = p1["har"]["complementary"][1]
        ana_hex1 = p1["har"]["analogous"][0][1]; ana_hex2 = p1["har"]["analogous"][1][1]
        tri_hex1 = p1["har"]["triadic"][0][1];   tri_hex2 = p1["har"]["triadic"][1][1]

        markers = [
            {"deg": hue_deg_from_hex(base_hex), "hex": base_hex, "label":"หลัก", "is_base": True,  "group":"base"},
            {"deg": hue_deg_from_hex(ana_hex1), "hex": ana_hex1, "label":"ใกล้เคียง +30°",  "group":"analogous"},
            {"deg": hue_deg_from_hex(ana_hex2), "hex": ana_hex2, "label":"ใกล้เคียง -30°",  "group":"analogous"},
            {"deg": hue_deg_from_hex(tri_hex1), "hex": tri_hex1, "label":"สามเหลี่ยม +120°","group":"triadic"},
            {"deg": hue_deg_from_hex(tri_hex2), "hex": tri_hex2, "label":"สามเหลี่ยม -120°","group":"triadic"},
            {"deg": hue_deg_from_hex(comp_hex), "hex": comp_hex, "label":"คู่ตรงข้าม 180°",  "group":"complementary"},
        ]
        wheel_img = make_color_wheel_image(markers, size=360)
        wheel_buf = io.BytesIO(); wheel_img.save(wheel_buf, format="PNG")
        wheel_b64 = base64.b64encode(wheel_buf.getvalue()).decode("ascii")

        legend = (
            f'<div class="small" style="margin-top:6px">'
            'คำอธิบายเส้น:'
            f'<span class="badge" style="background:rgba(27,35,48,.7);">'
            f'<span style="display:inline-block;width:10px;height:10px;border-radius:999px;'
            f'background:{SCHEME_STYLE["analogous"]["hex"]};margin-right:6px"></span>'
            f'{SCHEME_STYLE["analogous"]["name"]}</span>'
            f'<span class="badge" style="background:rgba(27,35,48,.7);">'
            f'<span style="display:inline-block;width:10px;height:10px;border-radius:999px;'
            f'background:{SCHEME_STYLE["triadic"]["hex"]};margin-right:6px"></span>'
            f'{SCHEME_STYLE["triadic"]["name"]}</span>'
            f'<span class="badge" style="background:rgba(27,35,48,.7);">'
            f'<span style="display:inline-block;width:10px;height:10px;border-radius:999px;'
            f'background:{SCHEME_STYLE["complementary"]["hex"]};margin-right:6px"></span>'
            f'{SCHEME_STYLE["complementary"]["name"]}</span>'
            f'<span class="badge" style="background:rgba(27,35,48,.7);">'
            f'<span style="display:inline-block;width:10px;height:10px;border-radius:999px;'
            f'background:{SCHEME_STYLE["base"]["hex"]};margin-right:6px"></span>'
            f'{SCHEME_STYLE["base"]["name"]}</span>'
            '</div>'
        )

        card_html = textwrap.dedent(f"""
        <div class="card">
          <div class="kpi">
            <div><div class="small">คะแนนรวม</div><div class="num">{overall:.1f}/100</div></div>
            <div><div class="small">ประเภทหลัก</div><div class="num">{primary_th}</div></div>
          </div>

          <div class="small" style="margin-bottom:10px">
            คู่ที่ประเมิน: <span class='badge'>{p1['name_th']}</span> + <span class='badge'>{p2['name_th']}</span>
            &nbsp;|&nbsp; สีหลัก: <code>{p1['dom_hex']}</code> &amp; <code>{p2['dom_hex']}</code><br/>
            ความต่างมุมสี (Δh°) = <b>{dh:.1f}°</b> &nbsp;|&nbsp; เป้าหมายแม่แบบ ≈ <b>{target}°</b>
            &nbsp;|&nbsp; สูตร: <code>Overall = 100 × f(Δh°) × g(ΔE₀₀)</code>
            ⇒ f = <b>{f_pct:.0f}%</b>, g = <b>{g_pct:.0f}%</b> (ΔE₀₀=<b>{de00:.1f}</b>)
          </div>

          <div style="display:flex; gap:18px; align-items:flex-start">
            <div style="flex:2; min-width:0;">
              <div style="margin-top:6px"><b>{head}</b>{tips_html}</div>
            </div>
            <div style="flex:1; min-width:280px; text-align:center;">
              <img src="data:image/png;base64,{wheel_b64}" style="width:100%; max-width:360px; border-radius:12px; border:1px solid var(--border)" />
              <div class="small" style="margin-top:6px; opacity:.85">วงล้อสี (หมุนตามเข็ม) &amp; จุดองศาโทนที่แนะนำ</div>
              {legend}
            </div>
          </div>
        </div>
        """).strip()

        st.markdown(card_html, unsafe_allow_html=True)

        st.markdown("โทนสีที่แนะนำ (สำหรับ Accessories หรือการตกแต่งเพิ่มเติม):")
        colA, colB, colC = st.columns([1,1,1])
        with colA: color_tile("คู่ตรงข้าม", [p1["har"]["complementary"]])
        with colB: color_tile("ใกล้เคียง", p1["har"]["analogous"])
        with colC: color_tile("สามเหลี่ยม", p1["har"]["triadic"])
    else:
        st.info("ต้องการอย่างน้อย 2 ชิ้น เพื่อประเมินความเข้ากันของชุด")

    st.markdown("---")

    for idx, p in enumerate(pieces):
        render_item_card_th(
            p["name_th"], p["conf"], p["har"], p["hue_deg"], p["hue_name"],
            is_achromatic=p["is_achromatic"], default_open=(idx==0)
        )

# ===== 17) FOOTER =====
st.markdown('''<div class="footer" style="text-align:center; margin-top:24px">
<span style="font-size:1em;font-weight:600;color:#d1d7e0;">พัฒนาโดย Chanaphon Phetnoi</span><br>
<span style="font-size:0.95em;color:#9aa4b2;">รหัสนักศึกษา 664230017 | ห้อง 66/45</span><br>
<span style="font-size:0.95em;color:#9aa4b2;">นักศึกษาสาขาเทคโนโลยีสารสนเทศ</span>
</div>''', unsafe_allow_html=True)

