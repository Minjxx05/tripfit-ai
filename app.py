import json
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st
from PIL import Image

from google import genai
from google.genai import errors as genai_errors


# =======================
# Models
# =======================
TEXT_MODEL_DEFAULT = "gemini-2.5-flash"
IMAGE_MODEL_DEFAULT = "gemini-2.5-flash-image"  # image model (권한 있으면 동작)


# =======================
# JSON helper
# =======================
def _safe_json_loads(text: str) -> Dict[str, Any]:
    t = (text or "").strip()
    if t.startswith("```"):
        parts = t.split("```")
        if len(parts) >= 3:
            t = parts[1]
    t = t.strip()
    i, j = t.find("{"), t.rfind("}")
    if i != -1 and j != -1 and j > i:
        t = t[i : j + 1]
    return json.loads(t)


# =======================
# Gemini helpers
# =======================
def get_api_key() -> Optional[str]:
    k = st.session_state.get("api_key_input")
    if k and k.strip():
        return k.strip()
    if "GEMINI_API_KEY" in st.secrets:
        return st.secrets["GEMINI_API_KEY"]
    return None


def gemini_client() -> genai.Client:
    k = get_api_key()
    if not k:
        raise RuntimeError("Gemini API Key가 필요합니다.")
    return genai.Client(api_key=k)


def call_text(prompt: str, model: str) -> str:
    client = gemini_client()
    resp = client.models.generate_content(model=model, contents=[prompt])
    return getattr(resp, "text", "") or ""


def call_json(prompt: str, model: str, retries: int = 2) -> Dict[str, Any]:
    rule = "반드시 유효한 JSON만 출력. 다른 텍스트/설명/마크다운/코드펜스 금지."
    last = None
    for _ in range(retries + 1):
        try:
            txt = call_text(rule + "\n" + prompt, model=model)
            return _safe_json_loads(txt)
        except Exception as e:
            last = e
    raise RuntimeError(f"JSON 파싱 실패: {last}")


# =======================
# Weather (Open-Meteo)
# =======================
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


# =======================
# Moodboard
# =======================
def build_mood_prompts(city: str, season: str, vibe: str) -> List[str]:
    return [
        f"Photorealistic street-style travel photo in {city} during {season}. Vibe: {vibe}. Natural light, no text, high detail.",
        f"Photorealistic travel outfit flat-lay on warm neutral background. Destination: {city}, season: {season}. Vibe: {vibe}. 7-9 items, no text.",
        f"Photorealistic candid travel moment in {city} during {season}. Vibe: {vibe}. Cinematic, no text.",
        f"Photorealistic travel editorial inspired by {city}. Season: {season}. Vibe: {vibe}. Premium look, no text.",
    ]


def generate_images_with_gemini(prompts: List[str], image_model: str) -> List[Image.Image]:
    """
    NOTE: 권한/결제/정책으로 서버가 거절하면 여기서 ClientError가 납니다.
    """
    client = gemini_client()
    imgs: List[Image.Image] = []
    for p in prompts:
        resp = client.models.generate_content(model=image_model, contents=[p])
        parts = getattr(resp, "parts", None)
        if not parts and hasattr(resp, "candidates") and resp.candidates:
            parts = resp.candidates[0].content.parts

        for part in parts or []:
            if getattr(part, "inline_data", None) is not None:
                imgs.append(part.as_image())
                break
    return imgs


# =======================
# Itinerary (Gemini text)
# =======================
def generate_itinerary(
    city: str,
    start: date,
    end: date,
    vibe: str,
    prefs: List[str],
    weather: List[Dict[str, Any]],
    text_model: str
) -> Dict[str, Any]:
    weather_lines = []
    for w in weather[:10]:
        weather_lines.append(f"{w['date']}: {w['tmin']}~{w['tmax']}C, rain{w['pop']}%, wind{w['wind']}km/h")
    weather_text = "\n".join(weather_lines) if weather_lines else "날씨 정보 없음"

    prefs_text = ", ".join(prefs) if prefs else "상관없음"

    prompt = f"""
너는 여행 플래너야. 아래 조건으로 '일자별 일정표'를 만들어줘.

[조건]
- 도시: {city}
- 기간: {start.isoformat()} ~ {end.isoformat()}
- 여행 무드: {vibe}
- 선호: {prefs_text}
- 날씨:
{weather_text}

[출력 JSON 스키마]
{{
  "summary": "한 문장 요약",
  "days": [
    {{
      "date": "YYYY-MM-DD",
      "theme": "그날 테마(짧게)",
      "schedule": [
        {{"time": "09:00", "title": "일정", "note": "이유/팁"}},
        {{"time": "13:00", "title": "점심/이동", "note": "팁"}},
        {{"time": "16:00", "title": "일정", "note": "팁"}},
        {{"time": "20:00", "title": "저녁", "note": "팁"}}
      ],
      "weather_tip": "그날 옷/우산/신발 같은 날씨 대응 팁 1줄"
    }}
  ]
}}

[규칙]
- 반드시 JSON만 출력.
- 장소명은 너무 과하게 구체적이지 않아도 되지만(실존 고정 X), 동선이 자연스럽게.
- 비/바람/추위가 있으면 schedule 또는 weather_tip에 반영.
"""
    return call_json(prompt, model=text_model)


# =======================
# UI
# =======================
st.set_page_config(page_title="Tripfit", layout="wide")

st.markdown(
    """
<style>
/* 화면 잘림 방지: width 제한 제거 + 타이틀 줄바꿈 */
.block-container { padding-top: 0.6rem; max-width: none !important; }
h1, h2, h3 { line-height: 1.1; word-break: keep-all; }
.title {
  font-size: clamp(28px, 3.2vw, 44px);
  font-weight: 900;
  letter-spacing: -0.02em;
  margin: 0 0 4px 0;
}
.subtle { color: rgba(0,0,0,0.55); }

.hero {
  background: linear-gradient(135deg, rgba(255,255,255,0.92), rgba(255,255,255,0.66));
  border: 1px solid rgba(0,0,0,0.06);
  border-radius: 26px;
  padding: 18px;
  box-shadow: 0 14px 55px rgba(0,0,0,0.07);
}

.card {
  background: rgba(255,255,255,0.86);
  border: 1px solid rgba(0,0,0,0.07);
  border-radius: 20px;
  padding: 14px 14px;
  box-shadow: 0 10px 34px rgba(0,0,0,0.05);
}

.hr { height: 1px; background: rgba(0,0,0,0.08); margin: 12px 0; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="title">Tripfit ✈️ 여행 무드보드 · 날씨 · 일정</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">무드보드는 크게, 일정은 날씨 기반으로 자동 생성</div>', unsafe_allow_html=True)

# state
st.session_state.setdefault("mood_imgs", [])
st.session_state.setdefault("weather", [])
st.session_state.setdefault("place", "")
st.session_state.setdefault("itinerary", None)

# Sidebar: inputs (필수 기능들 다 여기서)
with st.sidebar:
    st.markdown("### 🔑 Gemini API Key")
    st.text_input("API Key", type="password", key="api_key_input", placeholder="AI Studio key")
    st.caption("✅ ready" if get_api_key() else "키가 필요해요")

    st.markdown("---")
    st.markdown("### 🌍 여행 설정")
    city = st.text_input("도시", value="Tokyo")
    c1, c2 = st.columns(2)
    start = c1.date_input("시작", value=date.today() + timedelta(days=7))
    end = c2.date_input("종료", value=date.today() + timedelta(days=10))

    st.markdown("### ✨ 무드")
    vibe = st.text_input("무드 키워드", value="clean, cinematic, city walk, warm tones")
    season = st.text_input("계절/시즌", value="current season")

    st.markdown("### 🗓️ 일정 선호")
    prefs = st.multiselect(
        "선호 요소",
        ["맛집/카페", "전시/뮤지엄", "쇼핑", "자연/공원", "야경", "로컬 체험", "휴식 위주"],
        default=["맛집/카페", "전시/뮤지엄"]
    )

    st.markdown("---")
    text_model = st.text_input("Text model", value=TEXT_MODEL_DEFAULT)
    image_model = st.text_input("Image model", value=IMAGE_MODEL_DEFAULT)

# HERO: Moodboard (가장 크게)
st.markdown('<div class="hero">', unsafe_allow_html=True)
topL, topR = st.columns([1.35, 1])

with topL:
    st.markdown("## 🖼️ 무드보드 (4컷)")
    st.markdown('<div class="subtle">이미지 생성이 막히면 아래에서 이미지 업로드로 계속 진행 가능</div>', unsafe_allow_html=True)

with topR:
    gen_mood = st.button("무드보드 생성", type="primary", use_container_width=True)
    gen_weather = st.button("날씨 업데이트", use_container_width=True)
    gen_plan = st.button("일정 생성", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)
st.write("")

# Actions
def update_weather():
    try:
        geo = geocode_city(city)
        if not geo:
            st.warning("도시를 찾지 못했어요. 영문 도시명으로도 시도해봐요.")
            return
        lat, lon, nm, country = geo
        f = forecast_daily(lat, lon, start, end)
        st.session_state.weather = weather_cards(f)
        st.session_state.place = f"{nm}, {country}"
    except Exception as e:
        st.warning(f"날씨를 가져오지 못했어요: {e}")

if gen_weather:
    update_weather()

if gen_mood:
    if not get_api_key():
        st.error("Gemini API Key부터 입력해줘.")
    else:
        prompts = build_mood_prompts(city, season, vibe)
        try:
            with st.spinner("이미지 생성 중…"):
                imgs = generate_images_with_gemini(prompts, image_model=image_model)
            if not imgs:
                st.warning("이미지가 0장 반환됐어요. (권한/쿼터/모델 가능)")
            st.session_state.mood_imgs = imgs
        except genai_errors.ClientError as e:
            # 여기서 “왜 안됨”이 아니라, “대안 제공”이 목표
            status = getattr(e, "status_code", None)
            st.error(f"이미지 생성이 서버에서 거절됐어요 (status={status}). 이 키/프로젝트는 이미지 모델 사용이 막혀있을 가능성이 큼.")
            st.session_state.mood_imgs = []
        except Exception as e:
            st.error(f"이미지 생성 실패: {e}")
            st.session_state.mood_imgs = []

if gen_plan:
    if not get_api_key():
        st.error("Gemini API Key부터 입력해줘.")
    else:
        if not st.session_state.weather:
            update_weather()
        try:
            with st.spinner("일정 생성 중…"):
                st.session_state.itinerary = generate_itinerary(
                    city=city, start=start, end=end,
                    vibe=vibe, prefs=prefs,
                    weather=st.session_state.weather,
                    text_model=text_model
                )
        except Exception as e:
            st.error(f"일정 생성 실패: {e}")


# =======================
# Render Moodboard (가장 크게)
# =======================
imgs = st.session_state.mood_imgs
if imgs:
    # 2x2 크게
    g1, g2 = st.columns(2)
    g1.image(imgs[0], use_container_width=True)
    if len(imgs) > 1:
        g2.image(imgs[1], use_container_width=True)
    if len(imgs) > 2:
        g1.image(imgs[2], use_container_width=True)
    if len(imgs) > 3:
        g2.image(imgs[3], use_container_width=True)
else:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 무드보드가 안 나올 때(즉시 해결)")
    st.markdown(
        "- 지금 상태는 **API가 이미지 생성을 거절**하고 있어요. (키/프로젝트 권한 문제)\n"
        "- 당장 결과가 필요하면 아래에서 **이미지를 업로드해서 무드보드를 구성**할 수 있어요.",
    )
    uploads = st.file_uploader(
        "무드보드로 쓸 이미지 4장 업로드(선택)",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True
    )
    if uploads:
        up_imgs = []
        for f in uploads[:4]:
            up_imgs.append(Image.open(f).convert("RGB"))
        if up_imgs:
            c1, c2 = st.columns(2)
            c1.image(up_imgs[0], use_container_width=True)
            if len(up_imgs) > 1:
                c2.image(up_imgs[1], use_container_width=True)
            if len(up_imgs) > 2:
                c1.image(up_imgs[2], use_container_width=True)
            if len(up_imgs) > 3:
                c2.image(up_imgs[3], use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.write("")

# =======================
# Weather + Itinerary (요구사항 반영)
# =======================
bottomL, bottomR = st.columns([1, 1.35])

with bottomL:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🌦️ 날씨")
    if st.session_state.place:
        st.caption(st.session_state.place)
    if st.session_state.weather:
        cols = st.columns(min(5, len(st.session_state.weather)))
        for i, w in enumerate(st.session_state.weather[:5]):
            with cols[i]:
                st.markdown(
                    f"**{w['date']}** {w['icon']}  \n"
                    f"**{w['tmin']}° ~ {w['tmax']}°**  \n"
                    f"<span class='subtle'>☔ {w['pop']}% · 💨 {w['wind']}km/h</span>",
                    unsafe_allow_html=True,
                )
    else:
        st.markdown('<div class="subtle">오른쪽 위 버튼에서 “날씨 업데이트”를 눌러줘.</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with bottomR:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🗓️ 일정")
    it = st.session_state.itinerary
    if it:
        st.markdown(f"**{it.get('summary','')}**")
        for day in it.get("days", []):
            st.markdown(f"#### {day.get('date','')} · {day.get('theme','')}")
            for s in day.get("schedule", []):
                st.write(f"- **{s.get('time','')}** {s.get('title','')} — {s.get('note','')}")
            wt = day.get("weather_tip", "")
            if wt:
                st.caption(wt)
            st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="subtle">오른쪽 위 버튼에서 “일정 생성”을 눌러줘.</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
