import json
import urllib.parse
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st
from PIL import Image
from google import genai
from google.genai import types

# ========= Models =========
TEXT_MODEL = "gemini-2.5-flash"
IMAGE_MODEL = "gemini-2.5-flash-image"  # Nano Banana 계열(네이티브 이미지 생성)

# ========= Shop URLs =========
MUSINSA_SEARCH_BASE = "https://store.musinsa.com/app/product/search?search_type=1&q="
ABLY_SEARCH_BASE = "https://m.a-bly.com/search?keyword="


# ========= Helpers =========
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


def get_api_key() -> Optional[str]:
    # 1) 화면 입력(세션)
    key = st.session_state.get("api_key_input")
    if key and key.strip():
        return key.strip()

    # 2) Streamlit secrets (선택)
    if "GEMINI_API_KEY" in st.secrets:
        return st.secrets["GEMINI_API_KEY"]

    return None


def gemini_client() -> genai.Client:
    key = get_api_key()
    if not key:
        raise RuntimeError("Gemini API Key가 필요합니다.")
    return genai.Client(api_key=key)


def call_gemini_text(prompt: str, temperature: float = 0.7) -> str:
    client = gemini_client()
    resp = client.models.generate_content(
        model=TEXT_MODEL,
        contents=[prompt],
        config=types.GenerateContentConfig(temperature=temperature),
    )
    if getattr(resp, "text", None):
        return resp.text
    return ""


def call_gemini_json(prompt: str, retries: int = 2) -> Dict[str, Any]:
    rule = "반드시 유효한 JSON만 출력. 다른 텍스트/설명/마크다운/코드펜스 금지."
    last = None
    for _ in range(retries + 1):
        try:
            txt = call_gemini_text(rule + "\n" + prompt, temperature=0.4)
            return _safe_json_loads(txt)
        except Exception as e:
            last = e
    raise RuntimeError(f"Gemini JSON 파싱 실패: {last}")


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


def summarize_weather(f: Dict[str, Any]) -> str:
    d = f.get("daily", {}) or {}
    times = d.get("time", []) or []
    tmax = d.get("temperature_2m_max", []) or []
    tmin = d.get("temperature_2m_min", []) or []
    pop = d.get("precipitation_probability_max", []) or []
    wind = d.get("windspeed_10m_max", []) or []

    if not times:
        return "날씨 정보를 가져오지 못했어요."

    lines = []
    for i in range(len(times)):
        lines.append(
            f"{times[i]} · {tmin[i]}~{tmax[i]}°C · ☔ {pop[i]}% · 💨 {wind[i]}km/h"
        )
    return "\n".join(lines)


def make_musinsa_search_url(query: str) -> str:
    return MUSINSA_SEARCH_BASE + urllib.parse.quote(query)


def make_ably_search_url(query: str) -> str:
    return ABLY_SEARCH_BASE + urllib.parse.quote(query)


def normalize_query(brand: str, name: str, extra: str = "") -> str:
    q = " ".join([x for x in [brand, name, extra] if x and x.strip()])
    return " ".join(q.split()).strip()


def generate_moodboard_images(city: str, season: str, style: str, vibe: str, n: int = 4) -> List[Image.Image]:
    client = gemini_client()
    prompts = [
        f"Photorealistic street-style fashion photo in {city} during {season}. Style: {style}. Vibe: {vibe}. Full body, natural light, no text.",
        f"Photorealistic outfit flat-lay on clean background. Destination: {city}, season: {season}. Style: {style}. Include 6-8 items. No text.",
        f"Photorealistic candid travel photo in {city} during {season}. Style: {style}. Vibe: {vibe}. Subject wearing travel outfit. No text.",
        f"Photorealistic fashion editorial inspired by {city} in {season}. Style: {style}. Vibe: {vibe}. Clean composition. No text.",
    ][:n]

    imgs: List[Image.Image] = []
    for p in prompts:
        resp = client.models.generate_content(
            model=IMAGE_MODEL,
            contents=[p],
            config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
        )

        parts = []
        if hasattr(resp, "parts") and resp.parts:
            parts = resp.parts
        elif hasattr(resp, "candidates") and resp.candidates:
            parts = resp.candidates[0].content.parts

        for part in parts:
            if getattr(part, "inline_data", None) is not None:
                imgs.append(part.as_image())
                break

    return imgs


# ========= UI =========
st.set_page_config(page_title="Tripfit", layout="wide")

st.markdown(
    """
<style>
:root { --card: rgba(255,255,255,0.75); }
.block-container { padding-top: 1.2rem; }
.big-title { font-size: 2.0rem; font-weight: 800; letter-spacing: -0.02em; }
.subtle { color: rgba(0,0,0,0.55); }
.card {
  background: var(--card);
  border: 1px solid rgba(0,0,0,0.08);
  border-radius: 18px;
  padding: 16px 16px;
}
.chip {
  display: inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(0,0,0,0.06);
  margin-right: 6px;
  margin-bottom: 6px;
  font-size: 0.85rem;
}
.hr {
  height: 1px;
  background: rgba(0,0,0,0.08);
  margin: 12px 0;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="big-title">Tripfit ✈️👗</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">여행지 분위기 + 날씨 + 취향 → 코디 & 쇼핑 & 패킹 & 무드보드</div>', unsafe_allow_html=True)

# session init
st.session_state.setdefault("outfits", [])
st.session_state.setdefault("packing_list", [])
st.session_state.setdefault("packed_set", set())
st.session_state.setdefault("weather_text", "")
st.session_state.setdefault("confirmed_products", [])  # [{outfit, category, url, note}]

# Sidebar
with st.sidebar:
    st.markdown("### 🔑 Gemini API Key")
    st.text_input(
        "키를 여기 붙여넣기",
        type="password",
        key="api_key_input",
        placeholder="AI Studio에서 발급한 Gemini API Key",
        help="이 키는 브라우저 세션에만 저장됩니다(새로고침/재접속 시 사라짐).",
    )
    has_key = bool(get_api_key())
    st.caption("✅ 연결됨" if has_key else "키를 넣어야 실행돼요.")

    st.markdown("---")
    st.markdown("### 🌍 여행 설정")
    destination = st.text_input("도시", value="Tokyo")
    c1, c2 = st.columns(2)
    start_date = c1.date_input("시작", value=date.today() + timedelta(days=7))
    end_date = c2.date_input("종료", value=date.today() + timedelta(days=10))

    st.markdown("### ✨ 취향")
    style = st.selectbox("스타일", ["미니멀", "빈티지", "스트릿", "클래식", "러블리", "시티보이/시티걸", "고프코어", "기타"])
    vibe = st.text_input("무드 키워드", value="clean, chic, city walk, travel street style")
    season_hint = st.text_input("계절/체감(선택)", value="")
    activities = st.multiselect("일정", ["박물관/미술관", "맛집/카페", "자연/트레킹", "야경/바", "쇼핑", "비즈니스/세미나", "테마파크"], default=["맛집/카페"])
    budget = st.selectbox("예산", ["가성비", "중간", "프리미엄"])

# Main tabs
tab1, tab2, tab3 = st.tabs(["👗 코디 & 쇼핑", "🧳 패킹", "🍌 무드보드"])

# ---------- TAB 1 ----------
with tab1:
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 오늘의 여행 룩 만들기")
        st.markdown('<div class="subtle">날씨부터 읽고, 코디를 감성적으로 뽑아줄게요.</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        go = st.button("✨ 코디 생성", type="primary", use_container_width=True)

    if go:
        if not has_key:
            st.error("Gemini API Key를 먼저 입력해줘.")
            st.stop()
        if end_date < start_date:
            st.error("종료일은 시작일 이후여야 해.")
            st.stop()

        with st.spinner("날씨 불러오는 중…"):
            geo = geocode_city(destination)
            if not geo:
                st.error("도시를 찾지 못했어. 영문 도시명으로도 시도해줘.")
                st.stop()
            lat, lon, city_name, country = geo
            f = forecast_daily(lat, lon, start_date, end_date)
            weather = summarize_weather(f)
            st.session_state.weather_text = weather

        season_line = season_hint.strip() if season_hint.strip() else "날씨 기반"
        tpo = ", ".join(activities) if activities else "일반 여행"

        with st.spinner("룩을 고르는 중…"):
            prompt = f"""
너는 여행 스타일리스트이자 쇼핑 큐레이터야.

[여행]
- 도시: {city_name}, {country}
- 기간: {start_date.isoformat()} ~ {end_date.isoformat()}
- 스타일: {style}
- 무드: {vibe}
- 일정: {tpo}
- 예산: {budget}
- 계절 힌트: {season_line}
- 날씨:
{st.session_state.weather_text}

[출력 JSON 스키마]
{{
  "outfits": [
    {{
      "title": "코디 이름(감성적으로)",
      "mood_tags": ["태그", "태그"],
      "scenario": "언제 입는지(짧게)",
      "why": "이 룩이 좋은 이유(2~3문장)",
      "layering_tip": "온도/비/바람 대응 팁(1~2문장)",
      "items": [
        {{
          "category": "상의/하의/아우터/신발/가방/액세서리",
          "must_have": true,
          "notes": "핏/소재/색/스타일 포인트",
          "product_candidates": [
            {{
              "platform": "MUSINSA|ABLY",
              "brand": "브랜드",
              "product_name": "상품명(검색에 걸리게 구체적으로)",
              "keywords": "검색 보조(색/핏/소재)",
              "price_tier": "가성비|중간|프리미엄"
            }}
          ]
        }}
      ]
    }}
  ],
  "packing_list": ["짐 리스트(중복 제거)"]
}}

[규칙]
- 반드시 JSON만 출력.
- 브랜드 언급 허용.
- URL은 만들지 말고, 검색에 잘 걸리도록 상품명/키워드를 구체화.
"""
            data = call_gemini_json(prompt)
            st.session_state.outfits = data.get("outfits", []) or []
            st.session_state.packing_list = data.get("packing_list", []) or []
            st.session_state.confirmed_products = []

    # Weather card
    if st.session_state.weather_text:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 🌦️ 날씨")
        st.markdown(st.session_state.weather_text)
        st.markdown("</div>", unsafe_allow_html=True)

    # Outfit cards
    if st.session_state.outfits:
        st.markdown("### 룩 카드")
        for oi, outfit in enumerate(st.session_state.outfits, start=1):
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"### {oi}. {outfit.get('title','')}")
            tags = outfit.get("mood_tags", []) or []
            if tags:
                st.markdown("".join([f'<span class="chip">{t}</span>' for t in tags]), unsafe_allow_html=True)

            st.markdown(f"**{outfit.get('scenario','')}**")
            st.markdown(f"{outfit.get('why','')}")
            st.caption(outfit.get("layering_tip", ""))

            st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

            for it in (outfit.get("items", []) or []):
                cat = it.get("category", "")
                must = it.get("must_have", False)
                notes = it.get("notes", "")

                st.markdown(f"**{cat}** {' · 꼭' if must else ''}")
                if notes:
                    st.caption(notes)

                cands = it.get("product_candidates", []) or []
                for ci, c in enumerate(cands):
                    platform = (c.get("platform") or "").strip().upper()
                    brand = (c.get("brand") or "").strip()
                    pname = (c.get("product_name") or "").strip()
                    kw = (c.get("keywords") or "").strip()
                    tier = (c.get("price_tier") or "").strip()

                    q = normalize_query(brand, pname, kw)
                    if not q:
                        continue

                    colA, colB, colC = st.columns([5, 2, 2])

                    with colA:
                        st.markdown(f"- **{brand}** · {pname}  \n  <span class='subtle'>{kw} · {tier}</span>",
                                    unsafe_allow_html=True)

                    with colB:
                        if platform == "ABLY":
                            st.link_button("에이블리 검색", make_ably_search_url(q), use_container_width=True)
                        else:
                            st.link_button("무신사 검색", make_musinsa_search_url(q), use_container_width=True)

                    with colC:
                        with st.popover("🔖 상품 확정"):
                            st.caption("검색에서 마음에 드는 ‘상품 상세 URL’을 붙여넣어 저장.")
                            url = st.text_input(
                                "상품 URL",
                                key=f"url_{oi}_{cat}_{ci}",
                                placeholder="https:// ...",
                            )
                            note = st.text_input(
                                "메모(선택)",
                                key=f"note_{oi}_{cat}_{ci}",
                                placeholder="예: 블랙 M, 롱기장",
                            )
                            if st.button("저장", key=f"save_{oi}_{cat}_{ci}", use_container_width=True):
                                if url and url.strip().startswith("http"):
                                    st.session_state.confirmed_products.append(
                                        {
                                            "outfit": outfit.get("title", ""),
                                            "category": cat,
                                            "brand": brand,
                                            "product_name": pname,
                                            "url": url.strip(),
                                            "note": note.strip(),
                                        }
                                    )
                                    st.success("저장됨")
                                else:
                                    st.error("URL이 유효하지 않아.")

            st.markdown("</div>", unsafe_allow_html=True)
            st.write("")

        if st.session_state.confirmed_products:
            st.markdown("### 🔖 저장한 상품")
            for p in st.session_state.confirmed_products:
                label = f"{p['outfit']} · {p['category']} · {p['brand']} · {p['product_name']}"
                if p.get("note"):
                    label += f"  ({p['note']})"
                st.link_button(label, p["url"], use_container_width=True)


# ---------- TAB 2 ----------
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🧳 패킹 체크")
    st.markdown('<div class="subtle">룩에서 뽑힌 아이템으로 시작해, 너만의 리스트로 다듬어봐.</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if not st.session_state.packing_list:
        st.info("먼저 ‘코디 생성’을 해줘.")
    else:
        add = st.text_input("추가할 아이템", placeholder="예: 접이식 우산, 보조배터리, 히트텍")
        if st.button("추가", use_container_width=True):
            if add.strip() and add.strip() not in st.session_state.packing_list:
                st.session_state.packing_list.append(add.strip())

        st.write("")
        packed = set(st.session_state.packed_set)
        for item in st.session_state.packing_list:
            v = st.checkbox(item, value=(item in packed), key=f"pack_{item}")
            if v:
                packed.add(item)
            else:
                packed.discard(item)
        st.session_state.packed_set = packed

        total = len(st.session_state.packing_list)
        done = len(st.session_state.packed_set)
        st.metric("진행", f"{done}/{total}")

        if st.button("부족한 것만 감성 체크", use_container_width=True):
            if not has_key:
                st.error("Gemini API Key를 입력해줘.")
            else:
                missing = [x for x in st.session_state.packing_list if x not in st.session_state.packed_set]
                if not missing:
                    st.success("완벽해. 그대로 떠나도 돼.")
                else:
                    prompt = f"""
너는 여행 패킹 컨설턴트야.
목적지: {destination}
스타일: {style}
날씨:
{st.session_state.weather_text}

미완료:
{missing}

- 우선순위 TOP 5만
- 각 항목: 왜 필요한지(짧게) + 대체 아이템(있다면)
- 말투는 담백하고 감성 있게, 불릿으로.
"""
                    st.markdown(call_gemini_text(prompt, temperature=0.5))


# ---------- TAB 3 ----------
with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🍌 무드보드")
    st.markdown('<div class="subtle">도시의 공기 + 오늘의 취향을 이미지로.</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    season_for_image = st.text_input("계절(이미지용)", value=season_hint if season_hint.strip() else "current season")

    if st.button("무드보드 생성", type="primary", use_container_width=True):
        if not has_key:
            st.error("Gemini API Key를 입력해줘.")
        else:
            with st.spinner("이미지 생성 중…"):
                imgs = generate_moodboard_images(destination, season_for_image, style, vibe, n=4)

            if not imgs:
                st.warning("이번엔 잘 안 나왔어. 키워드를 조금 바꿔줘.")
            else:
                cols = st.columns(4)
                for i, im in enumerate(imgs):
                    cols[i % 4].image(im, use_container_width=True)
