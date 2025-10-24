# -*- coding: utf-8 -*-
"""
Durian Leaf Disease (Rules + Deep) — รีไฟน์กฎสี/ขนาดก้อน/การกระจายจุด
- UPDATED: ใบไหม้บังคับกรอบภายใน [100, 250] px 
- NEW: ขยายช่วงสีใบจุด (ส้ม–เหลือง–น้ำตาลสว่าง), ลด S ขั้นต่ำ, green ring test,
       เงื่อนไขจำนวนจุด ≥ N และ "การกระจาย" ของจุด (spread) ทั้งภาพ
- FIX: เพิ่ม show_rgb() และใช้แทน st.image() ทุกจุด เพื่อหลีกเลี่ยง dtype/shape issues บน Cloud
"""

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import cv2

# ---- Deep Features (optional) ----
import torch
import torchvision.models as models
from torchvision import transforms
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ============== Style / UI ==============
st.set_page_config(page_title="ระบบการวิเคราะห์โรคจากใบทุเรียนเบื้องต้น", page_icon="🌿", layout="centered")
st.markdown("""
<style>
h1 { font-size: 24px !important; color: #1f6feb !important; }
h2 { font-size: 20px !important; color: rgb(34,139,34) !important; }
h3 { font-size: 18px !important; color: rgb(170,29,18) !important; }
</style>""", unsafe_allow_html=True)

st.title("ระบบการวิเคราะห์โรคจากใบทุเรียนเบื้องต้น 🌿")
st.header("(Preliminary Durian Leaf Disease Analysis System)")

# ============== Helper: แสดงภาพแบบปลอดภัย ==============
def show_rgb(img_any, caption=None):
    """
    แสดงภาพให้ปลอดภัยกับ Streamlit: บังคับ dtype/shape เป็น RGB uint8
    รองรับทั้ง numpy array และ PIL.Image
    """
    from PIL import Image
    import numpy as np, cv2 as _cv2

    # PIL → แสดงได้เลย
    if isinstance(img_any, Image.Image):
        st.image(img_any, caption=caption, use_container_width=True)
        return

    # แปลงเป็น ndarray
    arr = np.array(img_any)

    # หากเป็น float/ints อื่น → clip แล้วแปลงเป็น uint8
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    # แปลงเป็น RGB 3 ช่อง
    if arr.ndim == 2:
        # grayscale → RGB
        arr = _cv2.cvtColor(arr, _cv2.COLOR_GRAY2RGB)
    elif arr.ndim == 3:
        if arr.shape[2] == 4:
            # RGBA → RGB
            arr = arr[:, :, :3]
        elif arr.shape[2] == 3:
            pass
        else:
            # ช่องสีไม่ปกติ → พยายามบังคับเป็น 3 ช่อง
            arr = arr[:, :, :3] if arr.shape[2] > 3 else _cv2.cvtColor(arr, _cv2.COLOR_GRAY2RGB)

    st.image(arr, caption=caption, use_container_width=True)

# ============== ค่าคงที่ภาพ ==============
TARGET_W, TARGET_H = 1750, 800
BORDER = 30  # ตัดขอบ 30px ทุกด้าน

def resize_contain_pad(img: Image.Image, target_w: int, target_h: int, bg=(0,0,0)) -> Image.Image:
    contained = ImageOps.contain(img, (target_w, target_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), bg)
    ox = (target_w - contained.width)//2
    oy = (target_h - contained.height)//2
    canvas.paste(contained, (ox, oy))
    return canvas

def to_analysis_rgb(pil_img: Image.Image) -> np.ndarray:
    img_r = resize_contain_pad(pil_img, TARGET_W, TARGET_H)
    arr = np.array(img_r)  # RGB
    return arr[BORDER:TARGET_H-BORDER, BORDER:TARGET_W-BORDER, :]

# ============== ตัวกรองเงาวาว ==============
def remove_specular(arr_rgb: np.ndarray, v_thr: int, s_max: int, l_thr: int) -> np.ndarray:
    bgr  = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
    hsv  = cv2.cvtColor(bgr,  cv2.COLOR_BGR2HSV)
    lab  = cv2.cvtColor(bgr,  cv2.COLOR_BGR2Lab)
    V,S  = hsv[:,:,2], hsv[:,:,1]
    L    = lab[:,:,0].astype(np.float32) * (100.0/255.0)
    spec = ((V >= v_thr) & (S <= s_max)) | (L >= l_thr)
    spec = cv2.medianBlur((spec.astype(np.uint8)*255), 3)
    keep = (spec == 0)

    out = arr_rgb.copy()
    if 0 < np.count_nonzero(~keep) < out.shape[0]*out.shape[1]:
        blur = cv2.medianBlur(cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR), 5)
        blur = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)
        out[~keep] = blur[~keep]
    return out

# ============== Utilities (HSV/Lab masks) ==============
def hsv_lab(arr_rgb):
    bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    H = hsv[:,:,0].astype(np.int16)      # 0..179 (OpenCV)
    S = hsv[:,:,1].astype(np.int16)
    V = hsv[:,:,2].astype(np.int16)
    A = lab[:,:,1].astype(np.int16) - 128
    B = lab[:,:,2].astype(np.int16) - 128
    return H,S,V,A,B

# ---------- GREEN mask สำหรับ “ล้อมด้วยสีเขียว” ----------
def green_mask(arr_rgb, h_min=35, h_max=85, s_min=35, v_min=35):  # UPDATED: s_min 35
    H,S,V,_,_ = hsv_lab(arr_rgb)
    mask = (H>=h_min) & (H<=h_max) & (S>=s_min) & (V>=v_min)
    return (mask.astype(np.uint8)*255)

# ============== Rules: ใบจุด (HSV/Lab) ==============
def rules_spot_mask(arr_rgb: np.ndarray,
                    hue_min:int, hue_max:int, a_min:int, b_min:int,
                    s_min:int, v_min:int,                       # NEW: S/V ปรับได้จาก UI
                    spot_min_area:int, spot_max_area:int,
                    green_ring_ratio:float=0.55, ring_px:int=5,
                    spread_min_px:int=120):                     # NEW: เกณฑ์ "การกระจาย"
    """
    เลือก 'ใบจุด' = จุดเล็กสี เหลือง–ส้ม–น้ำตาลสว่าง
    - ขยาย H ให้ครอบส้ม/เหลือง/น้ำตาลสว่าง
    - ลด S ขั้นต่ำ (ดึงจุดซีดได้ดีขึ้น)
    - บังคับให้จุดถูกล้อมด้วยสีเขียว (green ring)
    - กรองขนาดจุด และตรวจ 'การกระจาย' ของ centroids ทั้งภาพ
    """
    bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)

    H = hsv[:,:,0].astype(np.int16)   # OpenCV 0..179 = องศา/2
    S = hsv[:,:,1].astype(np.int16)
    V = hsv[:,:,2].astype(np.int16)
    A = lab[:,:,1].astype(np.int16) - 128
    B = lab[:,:,2].astype(np.int16) - 128

    # ===== สี "เหลือง–ส้ม–น้ำตาลสว่าง" =====
    H_MIN_SPOT = hue_min         # แนะนำ 10
    H_MAX_SPOT = hue_max         # แนะนำ 35
    S_MIN_SPOT = s_min           # NEW
    V_MIN_SPOT = v_min           # NEW
    A_MIN_SPOT = a_min
    B_MIN_SPOT = b_min

    # ---- รวมเกณฑ์สี ----
    mask_h   = (H >= H_MIN_SPOT) & (H <= H_MAX_SPOT)
    mask_sv  = (S >= S_MIN_SPOT) & (V >= V_MIN_SPOT)
    mask_lab = (A >= A_MIN_SPOT) & (B >= B_MIN_SPOT)
    mask     = (mask_h & mask_sv & mask_lab).astype(np.uint8) * 255

    # ทำความสะอาดรูปทรง
    mask = cv2.medianBlur(mask, 3)
    k3   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k3, iterations=1)

    # คอมโพเนนต์ + จำกัดขนาดจุด (≈ ไม่เกิน 30×30 px)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # ตรวจว่า "ล้อมด้วยสีเขียว"
    gmask = green_mask(arr_rgb)
    keep = np.zeros_like(mask)
    filtered_stats = []
    kept_centers = []  # NEW: เก็บ centroid ที่ผ่านทั้งหมด
    for i in range(1, n):
        x,y,w,h,area = stats[i,0], stats[i,1], stats[i,2], stats[i,3], stats[i, cv2.CC_STAT_AREA]
        if not (spot_min_area <= area <= spot_max_area):
            continue

        # ring รอบจุด (วงแหวน 5 px) ต้องเป็นสีเขียวตามสัดส่วน
        pad = ring_px
        x0, y0 = max(0,x-pad), max(0,y-pad)
        x1, y1 = min(mask.shape[1], x+w+pad), min(mask.shape[0], y+h+pad)
        ring = np.zeros((y1-y0, x1-x0), np.uint8)
        ring[(labels[y0:y1,x0:x1] == i)] = 255
        ring = cv2.dilate(ring, k3, iterations=1)
        ring = ring - (labels[y0:y1,x0:x1]==i).astype(np.uint8)*255

        green_pixels = np.count_nonzero(gmask[y0:y1,x0:x1] & ring)
        total_ring   = np.count_nonzero(ring)
        if total_ring == 0:
            continue
        if green_pixels / float(total_ring) >= green_ring_ratio:
            keep[labels == i] = 255
            filtered_stats.append(stats[i])
            cx, cy = centroids[i]
            kept_centers.append((float(cx), float(cy)))  # NEW

    # ----- NEW: เช็ก "การกระจาย" ของจุด (spread) -----
    spread_ok = True
    if len(kept_centers) >= 2:
        xs = np.array([c[0] for c in kept_centers], dtype=np.float32)
        ys = np.array([c[1] for c in kept_centers], dtype=np.float32)
        # ใช้ std ของแกนที่มากกว่าเป็นตัวแทนการกระจาย
        spread_value = max(xs.std(), ys.std())
        spread_ok = (spread_value >= float(spread_min_px))
    # หากจุดน้อยกว่า 2 ให้ถือว่า spread ผ่านอัตโนมัติ

    viz = cv2.dilate(keep, k3, iterations=1)
    return keep, viz, filtered_stats, spread_ok  # UPDATED: คืน spread_ok

# ============== Rules: ใบไหม้ (เหลือง/น้ำตาล/เทา) ==============
def rules_blight_mask(arr_rgb: np.ndarray,
                      min_bbox:int, max_bbox:int,
                      merge_px:int=12):
    """
    - พิจารณา Lab a*, b* สำหรับเหลือง/น้ำตาล และ S ต่ำ (เทา/ซีด)
    - รวมก้อนใกล้กันด้วย dilation
    - 'ผ่านเกณฑ์' เมื่อกรอบคอมโพเนนต์มี w,h ภายในช่วง [min_bbox, max_bbox] เท่านั้น
    """
    H,S,V,A,B = hsv_lab(arr_rgb)
    brown_yellow = (A >= 12) & (B >= 25)
    grayish = (S <= 45) & (V >= 90) & (V <= 220)
    base = ((brown_yellow | grayish).astype(np.uint8) * 255)

    base = cv2.medianBlur(base, 5)
    k5   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    base = cv2.morphologyEx(base, cv2.MORPH_CLOSE, k5, iterations=2)
    base = cv2.dilate(base, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(merge_px,merge_px)), iterations=1)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(base, connectivity=8)
    final  = np.zeros_like(base)
    bboxes = []
    for i in range(1, n):
        x,y,w,h = stats[i,0], stats[i,1], stats[i,2], stats[i,3]
        if (w >= min_bbox and h >= min_bbox and w <= max_bbox and h <= max_bbox):
            final[labels == i] = 255
            bboxes.append((x,y,w,h))
    return final, bboxes

# ============== Healthy-check (ไม่มี brown cluster ≥ 10×10) ==============
def has_large_brown_cluster(arr_rgb: np.ndarray, min_bbox:int=10) -> bool:
    H,S,V,A,B = hsv_lab(arr_rgb)
    brown = (A >= 14) & (B >= 22) & (S >= 30) & (V <= 215)
    mask = (brown.astype(np.uint8)*255)
    mask = cv2.medianBlur(mask, 3)
    k3   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k3, iterations=1)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    for i in range(1, n):
        w, h = stats[i,2], stats[i,3]
        if w >= min_bbox and h >= min_bbox:
            return True
    return False

# ============== วาดผล ==============
def draw_red_circles(arr_rgb: np.ndarray, mask255: np.ndarray) -> np.ndarray:
    out = arr_rgb.copy()
    contours, _ = cv2.findContours(mask255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cnt.shape[0] < 5: continue
        (x,y), r = cv2.minEnclosingCircle(cnt)
        if r < 1: continue
        cv2.circle(out, (int(x), int(y)), int(r)+3, (255,0,0), 2)  # หมายเหตุ: OpenCV ใช้ BGR
    return out

def draw_orange_boxes(arr_rgb: np.ndarray, bboxes) -> np.ndarray:
    out = arr_rgb.copy()
    for (x,y,w,h) in bboxes:
        cv2.rectangle(out, (x,y), (x+w, y+h), (255,165,0), 3)     # BGR
    return out

# ============== Deep (ตัวเลือก) ==============
@st.cache_resource(show_spinner=False)
def get_backbone(backbone_name="resnet50"):
    if backbone_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = torch.nn.Identity()
    else:
        model = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT)
        model.classifier = torch.nn.Identity()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    tfm = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return model, tfm, device

def deep_feature(arr_rgb: np.ndarray, backbone_name="resnet50") -> np.ndarray:
    model, tfm, device = get_backbone(backbone_name)
    pil = Image.fromarray(arr_rgb)
    x = tfm(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(x).detach().cpu().numpy().ravel()
    return feat

def train_pca_svm(X: np.ndarray, y: np.ndarray, n_components: int, C: float, kernel: str):
    n_components = int(min(max(4, n_components), max(4, len(X)-1)))
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, whiten=True, random_state=42)),
        ("svm", SVC(C=C, kernel=kernel, probability=True, class_weight="balanced", random_state=42))
    ])
    pipe.fit(X, y)
    return pipe, n_components

# ============== Sidebar Params ==============
with st.sidebar:
    st.subheader("โหมดการจำแนก")
    mode = st.radio("เลือกวิธี", ["Rules Features(RGB/HSV/LAB)", "Deep Features(PCA/SVM)"], index=0)

    st.divider()
    st.subheader("Specular Removal (ตัวกรองเงาวาว)")
    v_thr = st.slider("HSV: V ≥", 0, 255, 240, 1)
    s_max = st.slider("HSV: S ≤", 0, 255, 60, 1)
    l_thr = st.slider("Lab: L* ≥", 0, 100, 88, 1)

    st.subheader("Rules — Leaf-spot (จุดเล็ก)")
    hue_min = st.slider("Hue min (°)", 0, 180, 10, 1)     # UPDATED: 10
    hue_max = st.slider("Hue max (°)", 0, 180, 35, 1)     # UPDATED: 35
    a_min   = st.slider("a* min", -128, 127, 10, 1)       # UPDATED: 10
    b_min   = st.slider("b* min", -128, 127, 22, 1)       # UPDATED: 22
    s_min_spot = st.slider("S min (spot)", 0, 255, 30, 1) # NEW: ลด S ขั้นต่ำ
    v_min_spot = st.slider("V min (spot)", 0, 255, 140, 1)# NEW: ปรับ V ขั้นต่ำ
    spot_min_area = st.number_input("พื้นที่จุดต่ำสุด (px)", 1, 200, 25, 1)
    spot_max_area = st.number_input("พื้นที่จุดสูงสุด (px)", 5, 2000, 400, 1)  # ~30x30 คือ 900
    count_thr     = st.slider("จำนวนจุดขั้นต่ำ", 0, 500, 8, 1)
    green_ratio   = st.slider("สัดส่วนสีเขียวล้อมจุด (ring%)", 0.0, 1.0, 0.50, 0.05)  # UPDATED: 50%
    spread_min_px = st.slider("เกณฑ์กระจาย (std px)", 0, 400, 120, 5)                 # NEW

    st.subheader("Rules — Leaf-blight (ก้อนใหญ่)")
    blight_min_bbox = st.number_input("ความกว้าง/สูงกรอบขั้นต่ำ (px)", 10, 2000, 100, 1)
    blight_max_bbox = st.number_input("ความกว้าง/สูงกรอบสูงสุด (px)", 50, 5000, 250, 1)

    st.subheader("Healthy guard")
    healthy_brown_min_bbox = st.number_input("ห้ามมี brown cluster ≥ (px)", 1, 200, 10, 1)

    st.divider()
    st.subheader("Deep Features (ตัวเลือก)")
    backbone     = st.selectbox("Backbone", ["resnet50", "densenet201"], index=0)
    n_components = st.slider("PCA components", 4, 256, 64, 4)
    C_val        = st.selectbox("SVM C", [0.5,1,2,4,8], index=1)
    kernel       = st.selectbox("SVM kernel", ["rbf","linear"], index=0)

# ============== โหมด Rules ==============
def run_rules_mode():
    st.subheader("อัปโหลดภาพเพื่อวิเคราะห์ (Rules Features(RGB/HSV/LAB))")
    files = st.file_uploader("เลือกรูป JPG/PNG/BMP/TIFF/WebP", type=["jpg","jpeg","png","bmp","tiff","webp"], accept_multiple_files=True, key="rules_files")
    if not files:
        st.info("โปรดอัปโหลดภาพอย่างน้อย 1 ไฟล์ (Please upload at least one image)")
        return

    for up in files:
        img  = Image.open(up).convert("RGB")
        area = remove_specular(to_analysis_rgb(img), v_thr, s_max, l_thr)

        # 1) ตรวจใบไหม้ก่อน (priority สูงกว่า) — เงื่อนไข bbox ภายใน [min,max]
        m_b, boxes = rules_blight_mask(area, int(blight_min_bbox), int(blight_max_bbox))
        if np.any(m_b):
            vis = draw_orange_boxes(area, boxes)
            st.markdown(f"**{up.name}** → ⚠️ ต้นทุเรียนเป็นโรคใบไหม้ (Leaf blight disease)")
            show_rgb(vis)
            continue

        # 2) ตรวจใบจุด
        m_s, m_viz, stats_list, spread_ok = rules_spot_mask(
            area, hue_min, hue_max, a_min, b_min,
            int(s_min_spot), int(v_min_spot),            # NEW
            int(spot_min_area), int(spot_max_area),
            green_ring_ratio=float(green_ratio), ring_px=5,
            spread_min_px=int(spread_min_px)             # NEW
        )
        n_spots = len(stats_list)
        if (n_spots >= int(count_thr)) and spread_ok:     # UPDATED: ต้อง "หลายจุด" และ "กระจาย"
            vis = draw_red_circles(area, m_viz)
            st.markdown(f"**{up.name}** → ⚠️ ต้นทุเรียนเป็นโรคใบจุด (Leaf spot disease)")
            show_rgb(vis)
            continue

        # 3) Healthy guard
        if has_large_brown_cluster(area, int(healthy_brown_min_bbox)):
            st.markdown(f"**{up.name}** → ⚠️ พบกลุ่มสีน้ำตาลขนาดเกิน {healthy_brown_min_bbox}×{healthy_brown_min_bbox} px (สงสัยเป็นโรคใบ)")
            show_rgb(area)
        else:
            st.markdown(f"**{up.name}** → ✅ ต้นทุเรียนมีคุณภาพดี (Good quality)")
            show_rgb(area)

# ============== โหมด Deep (ตัวเลือก) ==============
def run_deep_mode():
    st.subheader("ฝึกโมเดล (Healthy / Leaf-spot / Leaf-blight) → Deep Features → PCA → SVM")
    c1, c2, c3 = st.columns(3)
    with c1:
        healthy_files = st.file_uploader("ฝึก: **คุณภาพดี**", type=["jpg","jpeg","png","bmp","tiff","webp"], accept_multiple_files=True, key="train_h")
    with c2:
        spot_files = st.file_uploader("ฝึก: **โรคใบจุด**", type=["jpg","jpeg","png","bmp","tiff","webp"], accept_multiple_files=True, key="train_s")
    with c3:
        blight_files = st.file_uploader("ฝึก: **โรคใบไหม้**", type=["jpg","jpeg","png","bmp","tiff","webp"], accept_multiple_files=True, key="train_b")

    if st.button("ฝึกโมเดล"):
        if not healthy_files or not spot_files or not blight_files:
            st.error("ต้องอัปโหลดครบทั้ง 3 คลาส")
            return
        X, y = [], []
        for up in healthy_files:
            arr = remove_specular(to_analysis_rgb(Image.open(up).convert("RGB")), v_thr, s_max, l_thr)
            X.append(deep_feature(arr, backbone)); y.append(0)
        for up in spot_files:
            arr = remove_specular(to_analysis_rgb(Image.open(up).convert("RGB")), v_thr, s_max, l_thr)
            X.append(deep_feature(arr, backbone)); y.append(1)
        for up in blight_files:
            arr = remove_specular(to_analysis_rgb(Image.open(up).convert("RGB")), v_thr, s_max, l_thr)
            X.append(deep_feature(arr, backbone)); y.append(2)

        X = np.array(X); y = np.array(y)
        if len(set(y)) < 3 or len(y) < 9:
            st.error("ข้อมูลฝึกยังน้อยเกินไป (แนะนำคลาสละ ≥ 10 ภาพ)")
            return
        clf, used_pca = train_pca_svm(X, y, int(n_components), float(C_val), kernel)
        st.session_state["clf"] = clf
        st.success(f"ฝึกโมเดลเสร็จ (ใช้ PCA = {used_pca})")

    st.subheader("ทดสอบภาพใหม่")
    test_files = st.file_uploader("อัปโหลดภาพทดสอบ", type=["jpg","jpeg","png","bmp","tiff","webp"], accept_multiple_files=True, key="test_files")
    if not test_files:
        st.info("อัปลองภาพทดสอบหลังฝึกโมเดล")
        return
    if "clf" not in st.session_state:
        st.warning("ยังไม่มีโมเดลในหน่วยความจำ — โปรดฝึกโมเดลก่อน")
        return

    clf = st.session_state["clf"]
    label_map = {
        0: "✅ ต้นทุเรียนมีคุณภาพดี (Good quality)",
        1: "⚠️ ต้นทุเรียนเป็นโรคใบจุด (Leaf spot disease)",
        2: "⚠️ ต้นทุเรียนเป็นโรคใบไหม้ (Leaf blight disease)",
    }

    for up in test_files:
        arr  = remove_specular(to_analysis_rgb(Image.open(up).convert("RGB")), v_thr, s_max, l_thr)
        feat = deep_feature(arr, backbone)
        prob = clf.predict_proba([feat])[0]
        yhat = int(np.argmax(prob))
        vis  = arr.copy()

        # ซ้อนผล rules เพื่อช่วยอธิบายภาพ
        if yhat == 2:
            m_b, boxes = rules_blight_mask(arr, int(blight_min_bbox), int(blight_max_bbox))
            if np.any(m_b): vis = draw_orange_boxes(arr, boxes)
        elif yhat == 1:
            m_s, m_viz, _, _ = rules_spot_mask(
                arr, hue_min, hue_max, a_min, b_min,
                int(s_min_spot), int(v_min_spot),
                int(spot_min_area), int(spot_max_area),
                green_ring_ratio=float(green_ratio), ring_px=5,
                spread_min_px=int(spread_min_px)
            )
            if np.any(m_s): vis = draw_red_circles(arr, m_viz)

        st.markdown(f"**{up.name}** → {label_map[yhat]}")
        show_rgb(vis)

# ============== Main ==============
if mode.startswith("Rules"):
    run_rules_mode()
else:
    run_deep_mode()
