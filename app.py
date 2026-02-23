import base64
import io
import json
from dataclasses import dataclass
from datetime import datetime, date
from typing import List, Dict, Any, Optional

import streamlit as st
from dateutil.parser import isoparse
from PIL import Image

from openai import OpenAI
from streamlit_calendar import calendar


# -----------------------------
# Mood
# -----------------------------
st.set_page_config(page_title="TRAVELFIT", page_icon="🧳", layout="wide")

st.markdown(
    """
    <style>
      .title {font-size:44px; font-weight:800; letter-spacing:0.5px; margin:0;}
      .sub {opacity:0.8; margin-top:4px;}
      .chip {display:inline-block; padding:6px 10px; border-radius:999px; background:rgba(255,255,255,0.08); margin-right:6px; font-size:12px;}
      .card {padding:16px; border-radius:18px; background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.07);}
      .muted {opacity:0.75;}
      .small {font-size:12px; opacity:0.8;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title">TRAVELFIT</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">여행의 시간표에 맞춰, 오늘의 옷을 고르는 작은 의식.</div>', unsafe_allow_html=True)
st.write("")


# -----------------------------
# Helpers
# -----------------------------
def b64_to_bytes(b64: str) -> bytes:
    return base64.b64decode(b64)

def bytes_to_pil(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")

def file_to_b64(uploaded_file) -> str:
    return base64.b64encode(uploaded_file.getvalue()).decode("utf-8")

def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")

def safe_json(s: str) -> dict:
    try:
        return json.loads(s)
    except Exception:
        return {}

def ensure_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)

def as_calendar_events(itinerary_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # streamlit-calendar (FullCalendar) events format
    events = []
    for i, item in enumerate(itinerary_items):
        title = item.get("title", f"Plan {i+1}")
        start = item.get("start")
        end = item.get("end") or start
        events.append(
            {
                "id": str(i),
                "title": title,
                "start": start,
                "end": end,
                "allDay": item.get("allDay", False),
            }
        )
    return events

def summarize_trip_text(destination: str, trip_start: date, trip_end: date, notes: str, itinerary: List[Dict[str, Any]]) -> str:
    lines = []
    lines.append(f"Destination: {destination}")
    lines.append(f"Dates: {trip_start.isoformat()} to {trip_end.isoformat()}")
    if notes.strip():
        lines.append(f"Notes: {notes.strip()}")
    if itinerary:
        lines.append("Itinerary:")
        for x in itinerary:
            lines.append(f"- {x.get('start','')} ~ {x.get('end','')}: {x.get('title','')}")
    return "\n".join(lines)


# -----------------------------
# Session
# -----------------------------
if "itinerary" not in st.session_state:
    st.session_state.itinerary = []
if "mood_images_b64" not in st.session_state:
    st.session_state.mood_images_b64 = []  # list[str] base64
if "mood_urls" not in st.session_state:
    st.session_state.mood_urls = []         # list[str]
if "outfit_images" not in st.session_state:
    st.session_state.outfit_images = []     # list[bytes]
if "outfit_text" not in st.session_state:
    st.session_state.outfit_text = ""


# -----------------------------
# Sidebar: API + Inputs
# -----------------------------
with st.sidebar:
    st.markdown("### 🔑 API")
    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    st.caption("화면에서 바로 넣고, 어디에도 저장하지 않아요.")

    st.markdown("---")
    st.markdown("### 🧭 여행")
    destination = st.text_input("여행지", placeholder="예: 도쿄, 제주, 파리")
    colA, colB = st.columns(2)
    with colA:
        trip_start = st.date_input("시작", value=date.today())
    with colB:
        trip_end = st.date_input("끝", value=date.today())

    notes = st.text_area(
        "추가 메모",
        placeholder="예: 미팅 1회, 많이 걷기, 비 예보, 사진 많이 찍고 싶음",
        height=120,
    )

    st.markdown("---")
    st.markdown("### 🎛️ 모델")
    text_model = st.text_input("텍스트 추천 모델", value="gpt-5")
    image_model = st.text_input("이미지 생성 모델", value="gpt-image-1.5")
    st.caption("이미지/비전 + 생성은 OpenAI Images/Responses로 연결돼요. :contentReference[oaicite:0]{index=0}")


# -----------------------------
# Layout
# -----------------------------
left, right = st.columns([1.05, 1])

# -----------------------------
# Left: Calendar + Itinerary
# -----------------------------
with left:
    st.markdown("### 📅 일정")
    st.markdown('<div class="muted">시간이 옷의 실루엣을 바꿔요.</div>', unsafe_allow_html=True)
    st.write("")

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)

        # Add itinerary item
        c1, c2, c3 = st.columns([1.2, 1, 1])
        with c1:
            title = st.text_input("일정 이름", key="new_title", placeholder="예: 미술관 / 바다 산책 / 디너")
        with c2:
            start_dt = st.text_input("시작(ISO)", key="new_start", placeholder="2026-02-24T10:00:00")
        with c3:
            end_dt = st.text_input("끝(ISO)", key="new_end", placeholder="2026-02-24T12:00:00")

        add = st.button("➕ 일정 추가", use_container_width=True)
        if add and title and start_dt:
            st.session_state.itinerary.append(
                {"title": title, "start": start_dt, "end": end_dt or start_dt, "allDay": False}
            )

        # Calendar view
        cal_options = {
            "initialView": "timeGridWeek",
            "headerToolbar": {"left": "prev,next today", "center": "title", "right": "dayGridMonth,timeGridWeek,timeGridDay"},
            "editable": False,
            "selectable": False,
            "height": 520,
        }
        cal_state = calendar(
            events=as_calendar_events(st.session_state.itinerary),
            options=cal_options,
            key="cal",
        )

        # Itinerary list
        st.write("")
        if st.session_state.itinerary:
            st.markdown("**리스트**")
            for idx, item in enumerate(st.session_state.itinerary):
                cols = st.columns([0.12, 0.88])
                with cols[0]:
                    if st.button("✕", key=f"del_{idx}"):
                        st.session_state.itinerary.pop(idx)
                        st.rerun()
                with cols[1]:
                    st.markdown(
                        f"<span class='chip'>{item.get('start','')}</span>"
                        f"<span class='chip'>{item.get('end','')}</span>"
                        f"**{item.get('title','')}**",
                        unsafe_allow_html=True
                    )
        else:
            st.markdown("<div class='small'>아직 비어 있어요. 하나만 적어도 충분해요.</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# Right: Moodboard + Outfit
# -----------------------------
with right:
    st.markdown("### 🖼️ 무드보드")
    st.markdown('<div class="muted">당신이 좋아하는 결의 색과 결.</div>', unsafe_allow_html=True)
    st.write("")

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        up = st.file_uploader(
            "이미지 업로드 (여러 장 가능)",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True
        )
        url_text = st.text_area(
            "이미지 URL (한 줄에 하나)",
            placeholder="https://...jpg\nhttps://...png",
            height=90
        )
        cols = st.columns(2)
        with cols[0]:
            if st.button("➕ 무드보드에 담기", use_container_width=True):
                if up:
                    for f in up:
                        st.session_state.mood_images_b64.append(file_to_b64(f))
                if url_text.strip():
                    st.session_state.mood_urls += [u.strip() for u in url_text.splitlines() if u.strip()]
        with cols[1]:
            if st.button("🧼 무드보드 비우기", use_container_width=True):
                st.session_state.mood_images_b64 = []
                st.session_state.mood_urls = []
                st.rerun()

        st.write("")
        # Display moodboard
        mood_cols = st.columns(3)
        idx = 0
        for b64img in st.session_state.mood_images_b64:
            with mood_cols[idx % 3]:
                st.image(bytes_to_pil(b64_to_bytes(b64img)), use_container_width=True)
            idx += 1

        for u in st.session_state.mood_urls:
            with mood_cols[idx % 3]:
                st.image(u, use_container_width=True)
            idx += 1

        if idx == 0:
            st.markdown("<div class='small'>이미지 몇 장이면 충분해요.</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown("### 👗 옷 추천")
    st.markdown('<div class="muted">일정과 무드가 만나는 지점.</div>', unsafe_allow_html=True)
    st.write("")

    # Controls
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        looks_n = st.slider("추천 룩 수", 2, 8, 4)
    with c2:
        temperature = st.slider("텍스트 감도", 0.0, 1.2, 0.7, 0.1)
    with c3:
        style_hint = st.text_input("스타일 한 줄", placeholder="예: 미니멀 / 시티보이 / 로맨틱 / 고프코어")

    go = st.button("✨ 추천 생성", use_container_width=True)

    if go:
        if not api_key:
            st.error("API Key가 필요해요.")
        elif not destination.strip():
            st.error("여행지를 적어줘요.")
        else:
            client = ensure_client(api_key)

            trip_text = summarize_trip_text(
                destination=destination.strip(),
                trip_start=trip_start,
                trip_end=trip_end,
                notes=notes,
                itinerary=st.session_state.itinerary,
            )

            # Build vision inputs (uploaded mood images)
            image_inputs = []
            for b64img in st.session_state.mood_images_b64[:8]:
                image_inputs.append(
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{b64img}",
                    }
                )
            # Also accept URL mood images as vision inputs
            for u in st.session_state.mood_urls[:8]:
                image_inputs.append({"type": "input_image", "image_url": u})

            # 1) Text recommendation (vision + schedule)
            prompt = f"""
너는 스타일리스트.
아래 여행 정보와 일정, (가능하다면) 무드보드 이미지를 바탕으로 {looks_n}개의 룩을 제안해.
조건:
- 룩마다: 이름(짧게), 상의/하의/아우터/신발/가방/액세서리, 소재/컬러, 이유(한 문장), 대안(비/추위/더위 대비)
- 여행지/걷는 양/일정 성격에 맞춰 현실적으로
- 과장 금지, 문장은 짧게, 감성은 조용하게
- 마지막에 '패킹 리스트'를 12개 이하 체크리스트로

추가 스타일 힌트: {style_hint or "없음"}

여행 정보:
{trip_text}
""".strip()

            # Responses API supports text + image inputs. :contentReference[oaicite:1]{index=1}
            input_payload = [{"role": "user", "content": [{"type": "input_text", "text": prompt}] + image_inputs}]

            with st.spinner("룩의 윤곽을 잡는 중..."):
                resp = client.responses.create(
                    model=text_model,
                    input=input_payload,
                    temperature=temperature,
                )
                outfit_text = getattr(resp, "output_text", "") or ""
                st.session_state.outfit_text = outfit_text

            # 2) Generate outfit images (one per look)
            # Use Images API (simple) :contentReference[oaicite:2]{index=2}
            # We’ll ask the model to produce consistent editorial-style images.
            looks_prompts = []
            if outfit_text.strip():
                # crude splitting: try to carve prompts by lines starting with numbering
                lines = [ln.strip() for ln in outfit_text.splitlines() if ln.strip()]
                # make up to looks_n prompts from the top section
                chunk = []
                for ln in lines:
                    chunk.append(ln)
                    if len(chunk) >= 10:
                        looks_prompts.append(" ".join(chunk))
                        chunk = []
                    if len(looks_prompts) >= looks_n:
                        break
                if len(looks_prompts) < looks_n and chunk:
                    looks_prompts.append(" ".join(chunk))

            st.session_state.outfit_images = []
            with st.spinner("이미지로 고요하게 입혀보는 중..."):
                for i in range(looks_n):
                    seed_text = looks_prompts[i] if i < len(looks_prompts) else outfit_text[:800]
                    img_prompt = f"""
Fashion editorial photo, full-body outfit on a model, clean background, soft natural light.
Outfit based on:
{seed_text}

Destination vibe: {destination}
Style hint: {style_hint or "none"}
High realism, detailed fabrics, no text, no logos.
""".strip()

                    img = client.images.generate(
                        model=image_model,
                        prompt=img_prompt,
                        n=1,
                        size="1024x1024",
                    )
                    b64 = img.data[0].b64_json
                    st.session_state.outfit_images.append(b64_to_bytes(b64))

    # Render outputs
    if st.session_state.outfit_text:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(st.session_state.outfit_text)
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.outfit_images:
        st.write("")
        st.markdown("#### 룩 이미지")
        grid = st.columns(2)
        for i, b in enumerate(st.session_state.outfit_images):
            with grid[i % 2]:
                st.image(bytes_to_pil(b), use_container_width=True)


st.write("")
st.caption("Images/Vision & Responses/Images API 기반. :contentReference[oaicite:3]{index=3}")
