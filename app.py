import json
import os
import tempfile
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st
from PIL import Image

# Gemini Developer API (AI Studio)
from google import genai
from google.genai import errors as genai_errors

# Vertex AI (Imagen fallback)
from google.cloud import aiplatform
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel


# =====================
# CONFIG
# =====================
TEXT_MODEL = "gemini-2.5-flash"
GEMINI_IMAGE_MODEL_DEFAULT = "gemini-2.5-flash-image"   # AI Studio image model
IMAGEN_MODEL_DEFAULT = "imagen-3.0-generate-002"        # Vertex AI Imagen (fallback)


# =====================
# UI STYLE
# =====================
st.set_page_config(page_title="Tripfit Moodboard", layout="wide")
st.markdown(
    """
<style>
.block-container { padding-top: 0.9rem; max-width: 1200px; }
h1 { margin-bottom: 0.2rem; }
.subtle { color: rgba(0,0,0,0.55); }
.hero {
  background: linear-gradient(135deg, rgba(255,255,255,0.82), rgba(255,255,255,0.58));
  border: 1px solid rgba(0,0,0,0.06);
  border-radius: 26px;
  padding: 18px;
  box-shadow: 0 12px 44px rgba(0,0,0,0.07);
}
.card {
  background: rgba(255,255,255,0.80);
  border: 1px solid rgba(0,0,0,0.07);
  border-radius: 20px;
  padding: 14px 14px;
  box-shadow: 0 8px 30px rgba(0,0,0,0.05);
}
.hr { height: 1px; background: rgba(0,0,0,0.08); margin: 12px 0; }
.pill {
  display: inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(0,0,0,0.06);
  margin-right: 6px;
  margin-bottom: 6px;
  font-size: 0.85rem;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("Tripfit 🍌 Moodboard")
st.markdown('<div class="subtle">이미지가 안 나오는 문제를 “코드로” 끝내려면, Gemini 실패 시 Vertex(Imagen)로 자동 우회가 필요해.</div>', unsafe_allow_html=True)

# =====================
# WEATHER (simple + intuitive)
# =====================
def geocode_city(city: str) -> Optional[Tuple[float, float, str, str]]:
    url = "https://geocoding-api.open-meteo.com/v1/search"
    r = requests.get(url, params={"name": city, "count": 1, "language": "ko", "format": "json"}, timeout=20)
    r.raise_for_status()
    data = r.json()
    results = data.get("results") or []
    if not results:
        return None
    it = results[0]
    return float(it["latitude"]), float(it["longitude"]), it.get("name", city), it.get("country", "")


def forecast_daily(lat: float, lon: float, start: date, end: date) -> Dict[str, Any]:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,windspeed_10m_max",
        "timezone": "auto",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def _rain_icon(pop: float) -> str:
    if pop >= 70:
        return "🌧️"
    if pop >= 40:
        return "🌦️"
    if pop >= 20:
        return "☁️"
    return "☀️"


def weather_cards(f: Dict[str, Any]) -> List[Dict[str, Any]]:
    d = f.get("daily", {}) or {}
    times = d.get("time", []) or []
    tmax = d.get("temperature_2m_max", []) or []
    tmin = d.get("temperature_2m_min", []) or []
    pop = d.get("precipitation_probability_max", []) or []
    wind = d.get("windspeed_10m_max", []) or []

    out = []
    for i in range(len(times)):
        out.append({
            "date": times[i],
            "icon": _rain_icon(float(pop[i])),
            "tmin": int(round(float(tmin[i]))),
            "tmax": int(round(float(tmax[i]))),
            "pop": int(round(float(pop[i]))),
            "wind": int(round(float(wind[i]))),
        })
    return out


# =====================
# Gemini client (AI Studio)
# =====================
def gemini_api_key() -> Optional[str]:
    k = st.session_state.get("gemini_key")
    if k and k.strip():
        return k.strip()
    if "GEMINI_API_KEY" in st.secrets:
        return st.secrets["GEMINI_API_KEY"]
    return None


def gemini_client() -> genai.Client:
    k = gemini_api_key()
    if not k:
        raise RuntimeError("Gemini API Key 없음")
    return genai.Client(api_key=k)


def gemini_generate_images(prompts: List[str], model: str) -> List[Image.Image]:
    """
    중요: 여기서는 config(response_modalities) 같은 거 안 씀.
    (환경에 따라 그게 400/403 유발하는 케이스가 있음)
    """
    client = gemini_client()
    imgs: List[Image.Image] = []
    for p in prompts:
        resp = client.models.generate_content(model=model, contents=[p])
        parts = getattr(resp, "parts", None)
        if not parts and hasattr(resp, "candidates") and resp.candidates:
            parts = resp.candidates[0].content.parts

        for part in parts or []:
            if getattr(part, "inline_data", None) is not None:
                imgs.append(part.as_image())
                break
    return imgs


# =====================
# Vertex AI (Imagen fallback)
# =====================
def init_vertex(project_id: str, location: str, sa_json_text: str) -> None:
    """
    service account json을 업로드/붙여넣기 받아서 임시 파일로 저장 후 인증.
    """
    if not project_id.strip() or not location.strip():
        raise RuntimeError("Vertex project_id/location 필요")

    # 임시 파일로 저장
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    tf.write(sa_json_text.encode("utf-8"))
    tf.flush()
    tf.close()

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tf.name

    # init
    aiplatform.init(project=project_id, location=location)
    vertexai.init(project=project_id, location=location)


def imagen_generate_images(
    prompts: List[str],
    model_name: str,
    aspect_ratio: str = "1:1",
) -> List[Image.Image]:
    """
    Vertex Imagen으로 이미지 생성.
    """
    model = ImageGenerationModel.from_pretrained(model_name)
    imgs: List[Image.Image] = []
    for p in prompts:
        res = model.generate_images(
            prompt=p,
            number_of_images=1,
            aspect_ratio=aspect_ratio,
            safety_filter_level="block_some",
        )
        # res.images[0] -> vertexai Image (PIL로 변환)
        if res.images:
            imgs.append(res.images[0]._pil_image)  # preview SDK 내부 필드 사용
    return imgs


# =====================
# Mood prompts
# =====================
def build_prompts(city: str, season: str, style: str, vibe: str) -> List[str]:
    return [
        f"Photorealistic street-style fashion photo in {city} during {season}. Style: {style}. Vibe: {vibe}. Full body, natural light, no text, high detail.",
        f"Photorealistic outfit flat-lay on warm neutral background. Destination: {city}, season: {season}. Style: {style}. Include 7-9 items, no text, high detail.",
        f"Photorealistic candid travel moment in {city} during {season}. Style: {style}. Vibe: {vibe}. Lifestyle, cinematic light, no text.",
        f"Photorealistic fashion editorial inspired by {city}. Season: {season}. Style: {style}. Vibe: {vibe}. Clean composition, premium look, no text.",
    ]


# =====================
# Sidebar
# =====================
with st.sidebar:
    st.markdown("### 1) Inputs")
    city = st.text_input("City", value="Tokyo")
    c1, c2 = st.columns(2)
    start = c1.date_input("From", value=date.today() + timedelta(days=7))
    end = c2.date_input("To", value=date.today() + timedelta(days=10))
    style = st.selectbox("Style", ["미니멀", "빈티지", "스트릿", "클래식", "러블리", "시티보이/시티걸", "고프코어", "기타"])
    vibe = st.text_input("Vibe", value="clean, chic, city walk, travel street style")
    season = st.text_input("Season", value="current season")

    st.markdown("---")
    st.markdown("### 2) Image backend")
    backend = st.radio("Generate with", ["Gemini (AI Studio key)", "Vertex (Imagen fallback)"], index=0)

    st.markdown("#### Gemini")
    st.text_input("Gemini API Key", type="password", key="gemini_key", placeholder="paste AI Studio key")
    gemini_image_model = st.text_input("Gemini image model", value=GEMINI_IMAGE_MODEL_DEFAULT)

    st.markdown("#### Vertex (only if Gemini fails / or choose this)")
    st.text_input("GCP Project ID", key="gcp_project_id", placeholder="e.g. my-project-123")
    st.text_input("Location", key="gcp_location", value="us-central1")
    imagen_model = st.text_input("Imagen model", value=IMAGEN_MODEL_DEFAULT)
    sa_json = st.text_area("Service Account JSON", height=150, placeholder='paste service account json here')


# =====================
# HERO
# =====================
st.markdown('<div class="hero">', unsafe_allow_html=True)
topL, topR = st.columns([1.4, 1])

with topL:
    st.markdown("### 🍌 Moodboard (4컷 크게)")
    st.markdown('<div class="subtle">버튼 한 번에 4장. Gemini가 거절하면(권한/결제) Vertex(Imagen)로 우회 가능.</div>', unsafe_allow_html=True)

with topR:
    gen = st.button("Generate", type="primary", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)
st.write("")


# =====================
# Generate
# =====================
if gen:
    prompts = build_prompts(city, season, style, vibe)

    # 날씨도 같이(직관 카드)
    weather = None
    place = None
    try:
        geo = geocode_city(city)
        if geo:
            lat, lon, nm, country = geo
            place = f"{nm}, {country}"
            f = forecast_daily(lat, lon, start, end)
            weather = weather_cards(f)
    except Exception:
        weather = None

    # 이미지 생성
    imgs: List[Image.Image] = []
    status = {"used": None, "detail": ""}

    if backend.startswith("Gemini"):
        try:
            imgs = gemini_generate_images(prompts, model=gemini_image_model)
            if not imgs:
                raise RuntimeError("Gemini returned 0 images")
            status["used"] = "Gemini"
        except genai_errors.ClientError as e:
            # Streamlit이 redacted하더라도 status_code는 뜨는 경우가 있음
            code = getattr(e, "status_code", None)
            status["detail"] = f"Gemini ClientError (status={code}). 이미지 모델 권한/결제/쿼터 문제일 확률이 큼."
            # 자동으로 Vertex로 재시도 (sa_json 있을 때만)
            if sa_json and st.session_state.get("gcp_project_id") and st.session_state.get("gcp_location"):
                try:
                    init_vertex(st.session_state["gcp_project_id"], st.session_state["gcp_location"], sa_json)
                    imgs = imagen_generate_images(prompts, model_name=imagen_model, aspect_ratio="1:1")
                    if imgs:
                        status["used"] = "Vertex Imagen (fallback after Gemini)"
                except Exception as ve:
                    status["detail"] += f" | Vertex fallback failed: {ve}"
            else:
                status["detail"] += " | Vertex fallback 설정이 비어있음(서비스계정 JSON/프로젝트 필요)."
        except Exception as e:
            status["detail"] = f"Gemini failed: {e}"

    else:
        try:
            init_vertex(st.session_state["gcp_project_id"], st.session_state["gcp_location"], sa_json)
            imgs = imagen_generate_images(prompts, model_name=imagen_model, aspect_ratio="1:1")
            if not imgs:
                raise RuntimeError("Imagen returned 0 images")
            status["used"] = "Vertex Imagen"
        except Exception as e:
            status["detail"] = f"Vertex Imagen failed: {e}"

    # =====================
    # Render
    # =====================
    if status["used"]:
        st.success(f"✅ Generated with: {status['used']}")
    else:
        st.error("❌ 이미지 생성 실패")
        if status["detail"]:
            st.caption(status["detail"])

    # Moodboard gallery (big 2x2)
    if imgs:
        g1, g2 = st.columns(2)
        if len(imgs) >= 1:
            g1.image(imgs[0], use_container_width=True)
        if len(imgs) >= 2:
            g2.image(imgs[1], use_container_width=True)
        if len(imgs) >= 3:
            g1.image(imgs[2], use_container_width=True)
        if len(imgs) >= 4:
            g2.image(imgs[3], use_container_width=True)

    # Weather card
    if weather:
        st.write("")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🌦️ Weather")
        if place:
            st.caption(place)
        cols = st.columns(min(4, len(weather)))
        for i, w in enumerate(weather[:4]):
            with cols[i]:
                st.markdown(
                    f"**{w['date']}** {w['icon']}  \n"
                    f"**{w['tmin']}° ~ {w['tmax']}°**  \n"
                    f"<span class='subtle'>☔ {w['pop']}% · 💨 {w['wind']}km/h</span>",
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Tip")
    st.markdown(
        "- Gemini로 계속 거절되면, 그건 **권한/결제 문제**라 코드로 못 뚫어.\n"
        "- 이 앱은 그래서 **Vertex Imagen fallback**을 붙여서 이미지가 나오게 만들었어.\n"
        "- Vertex를 쓰려면: Project ID + Location + Service Account JSON이 필요해.",
    )
    st.markdown("</div>", unsafe_allow_html=True)
