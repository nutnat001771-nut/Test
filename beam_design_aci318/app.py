"""
โปรแกรมออกแบบคานคอนกรีตเสริมเหล็กหน้าตัดสี่เหลี่ยม ตามมาตรฐาน ACI 318-19
หน่วย: SI (mm, MPa, kN, kN·m)

ขอบเขต:
  - ออกแบบรับแรงดัด (เสริมเหล็กรับแรงดึงอย่างเดียว / เสริมเหล็กรับแรงอัด)
  - ตรวจสอบกำลังดัดด้วย strain compatibility
  - ออกแบบรับแรงเฉือน (Table 22.5.5.1 รวม size effect)
  - ตรวจสอบระยะเรียงเหล็ก ควบคุมรอยร้าว และความลึกขั้นต่ำ

รัน:  streamlit run app.py
"""

import math

import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.patches import Circle, Rectangle

# ---------------------------------------------------------------------------
# ค่าคงที่
# ---------------------------------------------------------------------------
ES = 200_000.0          # MPa  (20.2.2.2)
EPS_CU = 0.003          # (22.2.2.1)
KSC_TO_MPA = 0.0980665
SQRT_FC_MAX = 8.3       # MPa  (22.5.3.1)
FYT_MAX_SHEAR = 420.0   # MPa  (Table 20.2.2.4a)

MAIN_BARS = {"DB10": 10, "DB12": 12, "DB16": 16, "DB20": 20,
             "DB25": 25, "DB28": 28, "DB32": 32}
STIRRUP_BARS = {"RB6": 6, "RB9": 9, "DB10": 10, "DB12": 12}
STEEL_GRADES = {"SD30 (fy = 295 MPa)": 295.0, "SD40 (fy = 390 MPa)": 390.0,
                "SD50 (fy = 490 MPa)": 490.0, "Grade 60 (fy = 420 MPa)": 420.0,
                "SR24 (fy = 235 MPa)": 235.0}


def bar_area(db: float) -> float:
    return math.pi * db ** 2 / 4.0


# ---------------------------------------------------------------------------
# ฟังก์ชันตามมาตรฐาน ACI 318-19
# ---------------------------------------------------------------------------
def beta1(fc: float) -> float:
    """Table 22.2.2.4.3"""
    if fc <= 28.0:
        return 0.85
    if fc >= 55.0:
        return 0.65
    return 0.85 - 0.05 * (fc - 28.0) / 7.0


def phi_flexure(eps_t: float, fy: float) -> tuple[float, str]:
    """Table 21.2.2 (ACI 318-19: tension-controlled เมื่อ εt ≥ εty + 0.003)"""
    eps_ty = fy / ES
    if eps_t >= eps_ty + 0.003:
        return 0.90, "Tension-controlled"
    if eps_t <= eps_ty:
        return 0.65, "Compression-controlled"
    return 0.65 + 0.25 * (eps_t - eps_ty) / 0.003, "Transition"


def as_min_flexure(fc: float, fy: float, bw: float, d: float) -> float:
    """9.6.1.2"""
    return max(0.25 * math.sqrt(fc) / fy, 1.4 / fy) * bw * d


def min_clear_spacing(db: float, dagg: float) -> float:
    """25.2.1"""
    return max(25.0, db, 4.0 / 3.0 * dagg)


def bars_per_layer_max(b, cover, ds, db, s_clear):
    width = b - 2 * cover - 2 * ds
    return max(int((width - db) // (db + s_clear)) + 1, 0)


def arrange_layers(n_total: int, n_max: int) -> list[int]:
    if n_max < 2:
        return []
    layers = []
    remaining = n_total
    while remaining > 0:
        n = min(remaining, n_max)
        layers.append(n)
        remaining -= n
    return layers


def layer_depths(h, cover, ds, db, n_layers, s_vert, top=False):
    """ระยะจากผิวบนถึงศูนย์กลางเหล็กแต่ละชั้น"""
    depths = []
    for i in range(n_layers):
        y = cover + ds + db / 2 + i * (db + s_vert)
        depths.append(y if top else h - y)
    return depths


def analyze_section(b, h, fc, fy, steel):
    """
    วิเคราะห์กำลังดัดด้วย strain compatibility
    steel: list ของ (As, depth_from_top)  ทั้งเหล็กบนและล่าง
    คืนค่า dict: c, a, Mn (N·mm), eps_t, phi, ...
    """
    b1 = beta1(fc)
    dt = max(d for _, d in steel)

    def forces(c):
        a = min(b1 * c, h)
        cc = 0.85 * fc * a * b
        items = []
        for As_i, d_i in steel:
            eps = EPS_CU * (c - d_i) / c          # + = อัด
            fs = max(min(ES * eps, fy), -fy)
            if fs > 0 and d_i < a:                 # หักพื้นที่คอนกรีตที่ถูกแทนที่
                fs -= 0.85 * fc
            items.append((As_i * fs, d_i, fs))
        return a, cc, items

    def net(c):
        _, cc, items = forces(c)
        return cc + sum(f for f, _, _ in items)

    lo, hi = 1e-3, 5.0 * h
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if net(mid) > 0:
            hi = mid
        else:
            lo = mid
    c = 0.5 * (lo + hi)
    a, cc, items = forces(c)
    mn = -(cc * a / 2 + sum(f * d_i for f, d_i, _ in items))
    eps_t = EPS_CU * (dt - c) / c
    phi, mode = phi_flexure(eps_t, fy)
    return {"c": c, "a": a, "Cc": cc, "Mn": mn, "eps_t": eps_t, "dt": dt,
            "phi": phi, "mode": mode, "steel": items}


def design_flexure(Mu, b, d, d_c, fc, fy):
    """
    หาพื้นที่เหล็กที่ต้องการ (N·mm, mm, MPa)
    จำกัด εt = εty + 0.003 เพื่อให้ φ = 0.90 ถ้าเกินจะเสริมเหล็กรับแรงอัด
    """
    b1 = beta1(fc)
    eps_ty = fy / ES
    phi = 0.90
    c_max = EPS_CU / (EPS_CU + eps_ty + 0.003) * d
    a_max = b1 * c_max
    mn1_max = 0.85 * fc * a_max * b * (d - a_max / 2)

    if Mu / phi <= mn1_max:
        rn = Mu / (phi * b * d ** 2)
        k = 1 - 2 * rn / (0.85 * fc)
        rho = 0.85 * fc / fy * (1 - math.sqrt(max(k, 0.0)))
        return {"doubly": False, "As": rho * b * d, "As_c": 0.0,
                "rho": rho, "Rn": rn, "mn1_max": mn1_max}

    as1 = 0.85 * fc * a_max * b / fy
    mn2 = Mu / phi - mn1_max
    eps_sc = EPS_CU * (c_max - d_c) / c_max
    fs_c = min(ES * eps_sc, fy)
    cs = mn2 / (d - d_c)
    if fs_c - 0.85 * fc <= 0:
        return {"doubly": True, "error": "เหล็กรับแรงอัดไม่มีประสิทธิภาพ "
                "(d' ใหญ่เกินไป) ควรเพิ่มขนาดหน้าตัด", "mn1_max": mn1_max}
    return {"doubly": True, "As": as1 + cs / fy, "As_c": cs / (fs_c - 0.85 * fc),
            "fs_c": fs_c, "mn1_max": mn1_max, "mn2": mn2}


def shear_vc(fc, bw, d, rho_w, lam, Nu, Ag, has_avmin):
    """Table 22.5.5.1 (N, mm, MPa)"""
    sfc = min(math.sqrt(fc), SQRT_FC_MAX)
    axial = min(Nu / (6 * Ag), 0.05 * fc)
    lam_s = min(math.sqrt(2 / (1 + d / 250)), 1.0)
    rho_term = rho_w ** (1 / 3)
    if has_avmin:
        vc_a = (0.17 * lam * sfc + axial) * bw * d
        vc_b = (0.66 * lam * rho_term * sfc + axial) * bw * d
        vc = max(vc_a, vc_b)
        eq = "max[(a), (b)]"
    else:
        vc = (0.66 * lam_s * lam * rho_term * sfc + axial) * bw * d
        eq = "(c) รวม λs"
    vc = max(vc, 0.0)
    vc_lim = 0.42 * lam * sfc * bw * d
    return min(vc, vc_lim), lam_s, eq


def av_min_per_s(fc, bw, fyt):
    """9.6.3.4 (mm²/mm)"""
    return max(0.062 * math.sqrt(fc) * bw / fyt, 0.35 * bw / fyt)


def s_max_shear(Vs, fc, bw, d, lam):
    """9.7.6.2.2"""
    if Vs <= 0.33 * lam * math.sqrt(fc) * bw * d:
        return min(d / 2, 600.0)
    return min(d / 4, 300.0)


def s_max_crack(fy, cc):
    """24.3.2 (fs = 2/3 fy)"""
    fs = 2 / 3 * fy
    return min(380 * (280 / fs) - 2.5 * cc, 300 * (280 / fs))


def h_min_beam(L, support, fy):
    """Table 9.3.1.1 (+ 9.3.1.1.1 สำหรับ fy ≠ 420)"""
    div = {"ช่วงเดียวธรรมดา (Simply supported)": 16,
           "ต่อเนื่องปลายเดียว (One end continuous)": 18.5,
           "ต่อเนื่องทั้งสองปลาย (Both ends continuous)": 21,
           "คานยื่น (Cantilever)": 8}[support]
    return L / div * (0.4 + fy / 700)


# ---------------------------------------------------------------------------
# วาดหน้าตัด
# ---------------------------------------------------------------------------
def draw_section(b, h, cover, ds, bot_layers, db, top_layers, db_c, s_vert):
    fig, ax = plt.subplots(figsize=(3.6, 3.6 * h / b if b > 0 else 4))
    ax.add_patch(Rectangle((0, 0), b, h, fc="#d9d9d9", ec="black", lw=1.5))
    ax.add_patch(Rectangle((cover + ds / 2, cover + ds / 2),
                           b - 2 * cover - ds, h - 2 * cover - ds,
                           fill=False, ec="#1f4e9c", lw=max(ds / 4, 1)))

    def place(layers, dia, top):
        for i, n in enumerate(layers):
            y_top = cover + ds + dia / 2 + i * (dia + s_vert)
            y = h - y_top if top else y_top
            x0, x1 = cover + ds + dia / 2, b - cover - ds - dia / 2
            xs = [x0] if n == 1 else [x0 + j * (x1 - x0) / (n - 1) for j in range(n)]
            for x in xs:
                ax.add_patch(Circle((x, y), dia / 2, fc="#b22222", ec="black"))

    place(bot_layers, db, top=False)
    place(top_layers, db_c, top=True)
    ax.set_xlim(-0.1 * b, 1.1 * b)
    ax.set_ylim(-0.1 * h, 1.1 * h)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(b / 2, -0.06 * h, f"b = {b:.0f} mm", ha="center", va="top", fontsize=8)
    ax.text(-0.04 * b, h / 2, f"h = {h:.0f} mm", ha="right", va="center",
            rotation=90, fontsize=8)
    fig.tight_layout()
    return fig


def ok(flag: bool) -> str:
    return "✅ ผ่าน" if flag else "❌ ไม่ผ่าน"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="ออกแบบคาน ACI 318-19", page_icon="🏗️", layout="wide")
st.title("🏗️ ออกแบบคานคอนกรีตเสริมเหล็ก ตามมาตรฐาน ACI 318-19")
st.caption("หน้าตัดสี่เหลี่ยมผืนผ้า · หน่วย SI (mm, MPa, kN, kN·m) · "
           "ผลลัพธ์ใช้เพื่อประกอบการออกแบบ วิศวกรผู้ออกแบบต้องตรวจสอบเอง")

with st.sidebar:
    st.header("1) วัสดุ")
    unit = st.radio("หน่วยกำลังวัสดุ", ["MPa", "ksc"], horizontal=True)
    if unit == "MPa":
        fc = st.number_input("f'c (MPa)", 17.0, 80.0, 24.0, 1.0)
    else:
        fc = st.number_input("f'c (ksc)", 180.0, 800.0, 240.0, 10.0) * KSC_TO_MPA
    fy = STEEL_GRADES[st.selectbox("เหล็กเสริมหลัก", list(STEEL_GRADES), index=1)]
    fyt_in = STEEL_GRADES[st.selectbox("เหล็กปลอก", list(STEEL_GRADES), index=4)]
    fyt = min(fyt_in, FYT_MAX_SHEAR)
    concrete = st.selectbox("ชนิดคอนกรีต", ["Normal weight (λ = 1.0)",
                                             "Sand-lightweight (λ = 0.85)",
                                             "All-lightweight (λ = 0.75)"])
    lam = {"N": 1.0, "S": 0.85, "A": 0.75}[concrete[0]]
    dagg = st.number_input("ขนาดมวลรวมใหญ่สุด (mm)", 10.0, 40.0, 20.0, 5.0)

    st.header("2) หน้าตัด")
    b = st.number_input("ความกว้าง b (mm)", 150.0, 2000.0, 300.0, 25.0)
    h = st.number_input("ความลึก h (mm)", 200.0, 3000.0, 600.0, 25.0)
    cover = st.number_input("ระยะหุ้ม (clear cover) (mm)", 20.0, 100.0, 40.0, 5.0)
    L = st.number_input("ความยาวช่วงคาน L (m)", 1.0, 30.0, 6.0, 0.5)
    support = st.selectbox("สภาพจุดรองรับ", [
        "ช่วงเดียวธรรมดา (Simply supported)",
        "ต่อเนื่องปลายเดียว (One end continuous)",
        "ต่อเนื่องทั้งสองปลาย (Both ends continuous)",
        "คานยื่น (Cantilever)"])

    st.header("3) แรงประลัย (Factored)")
    Mu_kNm = st.number_input("Mu (kN·m)", 0.0, 10000.0, 250.0, 10.0)
    Vu_kN = st.number_input("Vu ที่หน้าตัดวิกฤต (kN)", 0.0, 10000.0, 200.0, 10.0)
    Nu_kN = st.number_input("Nu แรงอัดตามแนวแกน (kN, + = อัด)", -2000.0, 5000.0, 0.0, 10.0)

    st.header("4) เหล็กเสริม")
    ds_name = st.selectbox("เหล็กปลอก", list(STIRRUP_BARS), index=1)
    ds = STIRRUP_BARS[ds_name]
    db_name = st.selectbox("เหล็กรับแรงดึง (ล่าง)", list(MAIN_BARS), index=4)
    db = MAIN_BARS[db_name]
    dbc_name = st.selectbox("เหล็กรับแรงอัด (บน)", list(MAIN_BARS), index=3)
    db_c = MAIN_BARS[dbc_name]
    s_vert = st.number_input("ระยะช่องว่างระหว่างชั้นเหล็ก (mm)", 25.0, 100.0, 25.0, 5.0)

Mu = Mu_kNm * 1e6
Vu = Vu_kN * 1e3
Nu = Nu_kN * 1e3
Ag = b * h

# ---------------------------- ออกแบบแรงดัด ---------------------------------
d_est = h - cover - ds - db / 2
dc_est = cover + ds + db_c / 2
des = design_flexure(Mu, b, d_est, dc_est, fc, fy)
As_min = as_min_flexure(fc, fy, b, d_est)

tab_flex, tab_shear, tab_serv, tab_sum = st.tabs(
    ["📐 แรงดัด (Flexure)", "✂️ แรงเฉือน (Shear)",
     "📏 ระยะเรียง / ความลึกขั้นต่ำ", "📋 สรุปผล"])

with tab_flex:
    st.subheader("ขั้นที่ 1: หาปริมาณเหล็กเสริมที่ต้องการ")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("β₁", f"{beta1(fc):.3f}")
    c2.metric("d (ประมาณ)", f"{d_est:.0f} mm")
    c3.metric("φMn,max เหล็กเดี่ยว", f"{0.9 * des['mn1_max'] / 1e6:.1f} kN·m",
              help="ที่ εt = εty + 0.003 (tension-controlled, φ = 0.90)")
    c4.metric("As,min (9.6.1.2)", f"{As_min:.0f} mm²")

    if "error" in des:
        st.error(des["error"])
        As_req, Asc_req = As_min, 0.0
    else:
        As_req = max(des["As"], As_min)
        Asc_req = des["As_c"]
        if des["doubly"]:
            st.warning("Mu เกินกำลังของคานเสริมเหล็กเดี่ยว → **ต้องเสริมเหล็กรับแรงอัด**")
            st.write(f"- f's ที่ c_max = {des['fs_c']:.0f} MPa")
        else:
            st.info(f"คานเสริมเหล็กรับแรงดึงอย่างเดียว  "
                    f"(Rn = {des['Rn']:.3f} MPa, ρ = {des['rho']:.5f})")
        cc1, cc2 = st.columns(2)
        cc1.metric("As ต้องการ", f"{As_req:.0f} mm²")
        cc2.metric("A's ต้องการ", f"{Asc_req:.0f} mm²")

    st.subheader("ขั้นที่ 2: เลือกเหล็กเสริมและตรวจสอบกำลัง")
    n_sug = max(2, math.ceil(As_req / bar_area(db)))
    nc_sug = max(2, math.ceil(Asc_req / bar_area(db_c)))
    k1, k2 = st.columns(2)
    n_bot = k1.number_input(f"จำนวน {db_name} ล่าง (แนะนำ {n_sug})", 2, 40, n_sug)
    n_top = k2.number_input(f"จำนวน {dbc_name} บน (แนะนำ {nc_sug})", 2, 40, nc_sug)
    use_top = st.checkbox("นำเหล็กบนมาคิดในกำลังดัด (strain compatibility)",
                          value=des.get("doubly", False))

    s_clear_b = min_clear_spacing(db, dagg)
    s_clear_t = min_clear_spacing(db_c, dagg)
    nmax_b = bars_per_layer_max(b, cover, ds, db, s_clear_b)
    nmax_t = bars_per_layer_max(b, cover, ds, db_c, s_clear_t)
    bot_layers = arrange_layers(n_bot, nmax_b)
    top_layers = arrange_layers(n_top, nmax_t)

    if not bot_layers or not top_layers:
        st.error("ความกว้างคานไม่พอจัดเหล็กอย่างน้อย 2 เส้นต่อชั้น")
        st.stop()

    Ab, Abc = bar_area(db), bar_area(db_c)
    bot_d = layer_depths(h, cover, ds, db, len(bot_layers), s_vert)
    top_d = layer_depths(h, cover, ds, db_c, len(top_layers), s_vert, top=True)
    steel = [(n * Ab, y) for n, y in zip(bot_layers, bot_d)]
    if use_top:
        steel += [(n * Abc, y) for n, y in zip(top_layers, top_d)]

    As_prov = n_bot * Ab
    Asc_prov = n_top * Abc
    d_act = sum(n * y for n, y in zip(bot_layers, bot_d)) / n_bot
    res = analyze_section(b, h, fc, fy, steel)
    phiMn = res["phi"] * res["Mn"]
    As_min_act = as_min_flexure(fc, fy, b, d_act)

    left, right = st.columns([1, 2])
    with left:
        st.pyplot(draw_section(b, h, cover, ds, bot_layers, db,
                               top_layers, db_c, s_vert))
        st.caption(f"ล่าง: {n_bot}-{db_name} ({' + '.join(map(str, bot_layers))} ต่อชั้น)  \n"
                   f"บน: {n_top}-{dbc_name}  · ปลอก {ds_name}")
    with right:
        r1, r2, r3 = st.columns(3)
        r1.metric("As ใช้", f"{As_prov:.0f} mm²")
        r2.metric("d จริง", f"{d_act:.1f} mm")
        r3.metric("dt", f"{res['dt']:.1f} mm")
        r1.metric("c", f"{res['c']:.1f} mm")
        r2.metric("a = β₁c", f"{res['a']:.1f} mm")
        r3.metric("εt", f"{res['eps_t']:.5f}")
        r1.metric("φ", f"{res['phi']:.3f}", res["mode"], delta_color="off")
        r2.metric("Mn", f"{res['Mn'] / 1e6:.1f} kN·m")
        r3.metric("φMn", f"{phiMn / 1e6:.1f} kN·m",
                  f"{phiMn / 1e6 - Mu_kNm:+.1f} เทียบ Mu")

        with st.expander("หน่วยแรงในเหล็กแต่ละชั้น"):
            st.table([{"ระยะจากผิวบน (mm)": f"{y:.1f}",
                       "fs (MPa, + อัด)": f"{fs:.1f}",
                       "แรง (kN)": f"{F / 1e3:.1f}"} for F, y, fs in res["steel"]])

    flex_checks = {
        "φMn ≥ Mu": phiMn >= Mu,
        "As ≥ As,min (9.6.1.2)": As_prov >= As_min_act,
        "εt ≥ 0.004 (9.3.3.1)": res["eps_t"] >= 0.004,
    }
    for k, v in flex_checks.items():
        st.write(f"{ok(v)} — {k}")

# ---------------------------- ออกแบบแรงเฉือน -------------------------------
with tab_shear:
    phi_v = 0.75
    d = d_act
    sfc = min(math.sqrt(fc), SQRT_FC_MAX)
    rho_w = As_prov / (b * d)
    n_legs = st.number_input("จำนวนขาเหล็กปลอก", 2, 8, 2)
    Av = n_legs * bar_area(ds)
    avmin_s = av_min_per_s(fc, b, fyt)

    vc_with, lam_s, _ = shear_vc(fc, b, d, rho_w, lam, Nu, Ag, True)
    vs_req = max(Vu / phi_v - vc_with, 0.0)
    vs_lim = 0.66 * sfc * b * d
    s_strength = Av * fyt * d / vs_req if vs_req > 0 else float("inf")
    s_avmin = Av / avmin_s
    s_code = s_max_shear(vs_req, fc, b, d, lam)
    s_design = min(s_strength, s_avmin, s_code)
    s_suggest = max(50.0, math.floor(s_design / 25.0) * 25.0)

    st.subheader("ค่ากำลังเฉือนของคอนกรีต")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("d", f"{d:.1f} mm")
    a2.metric("ρw", f"{rho_w:.4f}")
    a3.metric("λs (size effect)", f"{lam_s:.3f}")
    a4.metric("fyt ที่ใช้", f"{fyt:.0f} MPa",
              help="จำกัด fyt ≤ 420 MPa (Table 20.2.2.4a)")

    a1.metric("Vs ต้องการ", f"{vs_req / 1e3:.1f} kN")
    a2.metric("s จากกำลัง", "∞" if math.isinf(s_strength) else f"{s_strength:.0f} mm")
    a3.metric("s จาก Av,min", f"{s_avmin:.0f} mm")
    a4.metric("s,max (9.7.6.2.2)", f"{s_code:.0f} mm")

    if vs_req > vs_lim:
        st.error("Vs > 0.66√f'c·bw·d (22.5.1.2) → **ต้องขยายหน้าตัด**")

    s_use = st.number_input(f"ระยะเรียงเหล็กปลอกที่ใช้ s (mm) — แนะนำ ≤ {s_suggest:.0f}",
                            25.0, 600.0, float(min(s_suggest, 600.0)), 25.0)
    has_avmin = Av / s_use >= avmin_s
    vc, _, vc_eq = shear_vc(fc, b, d, rho_w, lam, Nu, Ag, has_avmin)
    vs = Av * fyt * d / s_use
    vs_eff = min(vs, vs_lim)
    phiVn = phi_v * (vc + vs_eff)
    vu_need_avmin = phi_v * 0.083 * lam * sfc * b * d

    b1c, b2c, b3c = st.columns(3)
    b1c.metric("Vc", f"{vc / 1e3:.1f} kN", vc_eq, delta_color="off")
    b2c.metric("Vs", f"{vs / 1e3:.1f} kN")
    b3c.metric("φVn", f"{phiVn / 1e3:.1f} kN", f"{phiVn / 1e3 - Vu_kN:+.1f} เทียบ Vu")
    st.caption(f"φ = 0.75 · ต้องใช้ Av,min เมื่อ Vu > φ·0.083λ√f'c·bw·d = "
               f"{vu_need_avmin / 1e3:.1f} kN (9.6.3.1)")

    shear_checks = {
        "φVn ≥ Vu": phiVn >= Vu,
        "Vs ≤ 0.66√f'c·bw·d (22.5.1.2)": vs_req <= vs_lim,
        "s ≤ s,max (9.7.6.2.2)": s_use <= s_max_shear(vs, fc, b, d, lam),
        "Av/s ≥ Av,min/s (9.6.3.4)": has_avmin or Vu <= vu_need_avmin,
    }
    for k, v in shear_checks.items():
        st.write(f"{ok(v)} — {k}")
    st.success(f"ใช้เหล็กปลอก {ds_name} {n_legs} ขา @ {s_use:.0f} mm")

# ---------------------- ระยะเรียง / ความลึกขั้นต่ำ ----------------------------
with tab_serv:
    n1 = bot_layers[0]
    width_c = b - 2 * (cover + ds) - db
    s_cc = width_c / (n1 - 1)
    s_clear_act = s_cc - db
    cc_clear = cover + ds
    s_crack = s_max_crack(fy, cc_clear)
    hmin = h_min_beam(L * 1000, support, fy)

    st.subheader("ระยะเรียงเหล็กเสริมหลัก (ชั้นล่างสุด)")
    e1, e2, e3 = st.columns(3)
    e1.metric("ระยะช่องว่างจริง", f"{s_clear_act:.0f} mm")
    e2.metric("ช่องว่างขั้นต่ำ (25.2.1)", f"{s_clear_b:.0f} mm")
    e3.metric("เหล็กสูงสุดต่อชั้น", f"{nmax_b} เส้น")

    st.subheader("ควบคุมรอยร้าว (24.3.2)")
    f1, f2 = st.columns(2)
    f1.metric("ระยะ c/c จริง", f"{s_cc:.0f} mm")
    f2.metric("s,max (fs = ⅔fy)", f"{s_crack:.0f} mm")

    st.subheader("ความลึกขั้นต่ำเพื่อไม่ต้องคำนวณการแอ่นตัว (Table 9.3.1.1)")
    g1, g2 = st.columns(2)
    g1.metric("h,min", f"{hmin:.0f} mm")
    g2.metric("h ที่ใช้", f"{h:.0f} mm")

    serv_checks = {
        "ช่องว่างระหว่างเหล็ก ≥ ค่าขั้นต่ำ (25.2.1)": s_clear_act >= s_clear_b - 1e-6,
        "ระยะเรียงควบคุมรอยร้าว (24.3.2)": s_cc <= s_crack,
        "h ≥ h,min (9.3.1.1)": h >= hmin,
    }
    for k, v in serv_checks.items():
        st.write(f"{ok(v)} — {k}")
    if h < hmin:
        st.info("h น้อยกว่าค่าขั้นต่ำ → ต้องคำนวณการแอ่นตัวตามหัวข้อ 24.2")

# --------------------------------- สรุป -----------------------------------
with tab_sum:
    all_checks = {**flex_checks, **shear_checks, **serv_checks}
    if all(all_checks.values()):
        st.success("หน้าตัดผ่านทุกรายการตรวจสอบ ✅")
    else:
        st.error("มีรายการที่ไม่ผ่าน กรุณาปรับแบบ ❌")
    st.markdown(f"""
| รายการ | ค่า |
|---|---|
| หน้าตัด | {b:.0f} × {h:.0f} mm, cover {cover:.0f} mm |
| วัสดุ | f'c = {fc:.1f} MPa, fy = {fy:.0f} MPa, fyt = {fyt:.0f} MPa |
| เหล็กล่าง | {n_bot}-{db_name} (As = {As_prov:.0f} mm²) |
| เหล็กบน | {n_top}-{dbc_name} (A's = {Asc_prov:.0f} mm²) |
| เหล็กปลอก | {ds_name} {n_legs} ขา @ {s_use:.0f} mm |
| Mu / φMn | {Mu_kNm:.1f} / {phiMn / 1e6:.1f} kN·m (ratio {Mu / phiMn:.2f}) |
| Vu / φVn | {Vu_kN:.1f} / {phiVn / 1e3:.1f} kN (ratio {Vu / phiVn:.2f}) |
""")
    st.table([{"รายการตรวจสอบ": k, "ผล": ok(v)} for k, v in all_checks.items()])
    st.caption("ไม่ครอบคลุม: แรงบิด, การแอ่นตัวโดยละเอียด, ระยะฝังและการต่อทาบเหล็ก, "
               "หน้าตัดรูปตัว T, ข้อกำหนดด้านแผ่นดินไหว (Chapter 18)")
