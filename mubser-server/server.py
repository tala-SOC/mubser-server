import os
import time
import threading
import asyncio
import base64
import cv2
import numpy as np
import edge_tts
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Header, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response
from pydantic import BaseModel
from google import genai
from google.genai import types as genai_types

# استدعاء الوكلاء الحقيقية من agents.py
from agents import NavAgent, VisionAgent, VisionNavAgent

app = FastAPI(title="Mubser AI Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AQ.Ab8RN6LaWQvXkf_TxWGXujTPu_LDESKXv56_Mqns8yJ_yDVB_w")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

AUDIO_FOLDER = os.path.join(os.getcwd(), 'static_audio')
os.makedirs(AUDIO_FOLDER, exist_ok=True)
app.mount("/audio", StaticFiles(directory=AUDIO_FOLDER), name="audio")

VOICE = "ar-EG-SalmaNeural"

# ==========================================
# النماذج الثقيلة المشتركة (Stateless)
# ==========================================
vision_agent = VisionAgent()         # LandmarkMatcher
vision_nav_agent = VisionNavAgent() # رصد العوائق + Sensor Fusion

# ==========================================
# إدارة الجلسات للمستخدمين
# ==========================================
class SessionManager:
    def __init__(self, session_timeout_seconds=6 * 3600):
        self._sessions = {}
        self._lock = threading.Lock()
        self.session_timeout_seconds = session_timeout_seconds
        self._daily_quota = {}
        self.max_questions_per_day = 50

    def get_nav_agent(self, session_id: str) -> NavAgent:
        with self._lock:
            entry = self._sessions.get(session_id)
            if entry is None:
                entry = {"nav_agent": NavAgent(alert_threshold_m=0.8), "last_active": time.time()}
                self._sessions[session_id] = entry
                print(f"🆕 [SessionManager]: جلسة جديدة: {session_id} (إجمالي الجلسات: {len(self._sessions)})")
            else:
                entry["last_active"] = time.time()
            return entry["nav_agent"]

    def check_and_increment_quota(self, session_id: str) -> bool:
        today = time.strftime("%Y-%m-%d")
        with self._lock:
            record = self._daily_quota.get(session_id)
            if record is None or record["date"] != today:
                record = {"date": today, "count": 0}
            if record["count"] >= self.max_questions_per_day:
                self._daily_quota[session_id] = record
                return False
            record["count"] += 1
            self._daily_quota[session_id] = record
            return True

    def cleanup_stale_sessions(self):
        while True:
            time.sleep(1800)  # كل 30 دقيقة
            now = time.time()
            with self._lock:
                stale = [sid for sid, e in self._sessions.items()
                         if now - e["last_active"] > self.session_timeout_seconds]
                for sid in stale:
                    del self._sessions[sid]
                if stale:
                    print(f"🧹 [SessionManager]: تنظيف {len(stale)} جلسة غير نشطة.")


session_manager = SessionManager()
threading.Thread(target=session_manager.cleanup_stale_sessions, daemon=True).start()


def require_session_id(x_session_id: str = Header(default=None)) -> str:
    if not x_session_id:
        raise HTTPException(status_code=400, detail="مطلوب هيدر X-Session-Id في كل طلب.")
    return x_session_id


def decode_uploaded_frame(image_bytes: bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return frame


class QuestionRequest(BaseModel):
    query: str = ""
    prompt: str = ""


class TTSRequest(BaseModel):
    text: str


# ==========================================
# 1. الأسئلة والصوت (QA & Voice)
# ==========================================
@app.post("/api/tts")
async def generate_tts(data: TTSRequest, session_id: str = Depends(require_session_id)):
    if not data.text or not data.text.strip():
        return {"status": "error", "message": "لا يوجد نص"}

    filename = f"alert_{int(time.time() * 1000)}.mp3"
    audio_path = os.path.join(AUDIO_FOLDER, filename)
    try:
        communicate = edge_tts.Communicate(data.text[:200], VOICE)
        await communicate.save(audio_path)
        public_base_url = os.environ.get("RENDER_EXTERNAL_URL", "http://localhost:8000")
        return {"status": "success", "audio_url": f"{public_base_url}/audio/{filename}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/describe-scene")
async def describe_scene(
    files: list[UploadFile] = File(...),
    user_prompt: str = Form(default="اوصف لي المكان بالتفصيل"),
    session_id: str = Depends(require_session_id),
):
    """
    يقوم باستقبال صور متتالية للمحيط بالإضافة لسؤال/طلب صوتي مخصص من الكفيف (user_prompt)،
    ثم يقوم باستدعاء نموذج Gemini 2.5 Flash وتوليد ملخص بصري مخصص وصوت TTS له.
    """
    if not client:
        return {"status": "error", "description": "خدمة الوصف السحابية غير متاحة حالياً."}

    images_parts = []
    for f in files[:6]:
        content = await f.read()
        frame = decode_uploaded_frame(content)
        if frame is None:
            continue
        success, encoded = cv2.imencode(".jpg", frame)
        if success:
            images_parts.append(
                genai_types.Part.from_bytes(data=encoded.tobytes(), mime_type="image/jpeg")
            )

    if not images_parts:
        return {"status": "error", "description": "لم يتم استقبال صور صالحة."}

    system_instruction = (
        "أنت مساعد بصري ذكي مصمم لمساعدة الأشخاص المكفوفين وضعاف البصر على فهم محيطهم بأمان. "
        "حلّل الصور المرفقة وقدّم وصفًا دقيقًا ومختصرًا وعمليًا للمشهد، مع التركيز على المعلومات "
        "التي تساعد المستخدم على معرفة ما يحيط به واتخاذ قرار آمن.\n\n"

        "اتبع الأولويات التالية عند تحليل الصورة:\n"
        "1. العوائق والمخاطر: حدّد أي عوائق أو مخاطر قريبة من المستخدم، مثل الأشخاص، "
        "الأثاث، الدرج، الأبواب، المركبات أو الأشياء الموجودة في مسار الحركة، واذكر موقعها "
        "بالنسبة للمستخدم (أمامك، يمينك، يسارك، أعلى أو أسفل) ومدى قربها بشكل تقريبي إن أمكن.\n"

        "2. المسارات والحركة: حدّد الممرات أو المساحات المفتوحة التي يمكن المرور من خلالها، "
        "وأوضح اتجاهها إن كان واضحًا، مثل: 'ممر مفتوح أمامك' أو 'يوجد عائق في الجهة اليمنى'.\n"

        "3. الأشخاص: اذكر عدد الأشخاص الظاهرين، وموقعهم، وألوان ملابسهم، وما يفعلونه "
        "إذا كان ذلك واضحًا من الصورة. لا تحاول تحديد هوية الأشخاص أو تخمين معلومات شخصية عنهم.\n"

        "4. الأشياء والأدوات: حدّد الأشياء البارزة والمهمة في المشهد، مثل الهاتف، الكوب، "
        "المقص، الحقيبة، الكرسي أو الطاولة، مع ذكر موقعها عند الحاجة.\n"

        "5. البيئة والمشهد العام: اذكر أهم تفاصيل المكان التي تساعد المستخدم على فهم البيئة "
        "المحيطة، مثل غرفة، شارع، متجر، ممر أو مدخل.\n\n"

        "قواعد مهمة:\n"
        "- رتّب المعلومات حسب الأهمية، وابدأ دائمًا بالعوائق أو المخاطر القريبة.\n"
        "- استخدم أوصافًا مكانية واضحة وبسيطة مثل: أمامك، خلفك، يمينك، يسارك، قريب منك.\n"
        "- كن مباشرًا ومختصرًا، وتجنب التفاصيل غير المفيدة.\n"
        "- لا تخمّن أو تفترض معلومات غير واضحة في الصورة. إذا كانت المعلومة غير مؤكدة، "
        "اذكر أنها غير واضحة بدلًا من اختلاقها.\n"
        "- لا تستخدم مقدمات أو عبارات مثل 'بالتأكيد' أو 'إليك وصف الصورة'.\n"
        "- اجعل الإجابة طبيعية وسهلة الفهم، وكأنك تصف المشهد للمستخدم صوتيًا في الوقت الفعل.\n"
        "- إذا لم توجد عوائق أو مخاطر واضحة، اذكر ذلك باختصار.\n"
        "- إذا كان هناك خطر قريب أو مهم، اجعله أول معلومة في الإجابة وبصياغة واضحة."
    )

    prompt = f"{system_instruction}\n\nسؤال/طلب المستخدم الخصيصاً: {user_prompt}"

    try:
        contents = [prompt] + images_parts
        response = client.models.generate_content(model='gemini-2.5-flash', contents=contents)
        description = response.text.strip()
    except Exception as e:
        print(f"❌ خطأ وصف المشهد: {e}")
        return {"status": "error", "description": "تعذر توليد الوصف حالياً."}

    filename = f"describe_{int(time.time() * 1000)}.mp3"
    audio_path = os.path.join(AUDIO_FOLDER, filename)
    audio_url = ""
    try:
        communicate = edge_tts.Communicate(description[:400], VOICE)
        await communicate.save(audio_path)
        public_base_url = os.environ.get("RENDER_EXTERNAL_URL", "http://localhost:8000")
        audio_url = f"{public_base_url}/audio/{filename}"
    except Exception as e:
        print(f"❌ خطأ توليد صوت الوصف: {e}")

    return {"status": "success", "description": description, "audio_url": audio_url}


@app.post("/api/qa")
@app.post("/chat")
async def ask_question(data: QuestionRequest, session_id: str = Depends(require_session_id)):
    user_query = data.query or data.prompt
    if not user_query:
        return {"status": "error", "response": "لم يتم استقبال أي نص"}

    if not session_manager.check_and_increment_quota(session_id):
        return {
            "status": "error",
            "response": f"تم الوصول للحد الأقصى ({session_manager.max_questions_per_day} سؤال) لهذا اليوم. حاول غداً.",
        }

    print(f"📥 [Server] سؤال جديد من {session_id[:8]}...: {user_query}")

    ai_response = None
    if client:
        try:
            prompt = (
                "أنت مساعد صوّتي ذكي لنظارة كفيف باسم 'مبصر' متخصص في مناسك الحج والعمرة "
                f"والأسئلة الفقهية والإرشاد المكاني. أجب باختصار ووضوح باللغة العربية بغير رموز تنسيق:\n{user_query}"
            )
            response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
            ai_response = response.text.strip()
            print(f"🤖 [Server] رد Gemini: {ai_response}")
        except Exception as e:
            print(f"❌ خطأ Gemini: {e}")

    if not ai_response:
        ai_response = "عذراً، تعذر الوصول للخدمة السحابية حالياً. حاول مرة أخرى أو تأكد من الاتصال بالإنترنت."

    filename = f"voice_{int(time.time() * 1000)}.mp3"
    audio_path = os.path.join(AUDIO_FOLDER, filename)
    audio_url = ""
    try:
        communicate = edge_tts.Communicate(ai_response[:300], VOICE)
        await communicate.save(audio_path)
        public_base_url = os.environ.get("RENDER_EXTERNAL_URL", "http://localhost:8000")
        audio_url = f"{public_base_url}/audio/{filename}"
    except Exception as e:
        print(f"❌ خطأ توليد الصوت: {e}")

    return {
        "status": "success",
        "response": ai_response,
        "reply": ai_response,
        "answer": ai_response,
        "audio_url": audio_url
    }


# ==========================================
# 2. رصد العوائق وفحص السلامة والتزاحم
# ==========================================

@app.post("/api/safety-check")
@app.post("/api/vision/process")
async def process_vision_frame(
    file: UploadFile = File(..., alias="image"),
    image: UploadFile = File(default=None),
    mode: str = Form(default="safety"),
    depth_meters: float | None = Form(default=None),
    phone_depth_reading: float | None = Form(default=None),
    session_id: str = Depends(require_session_id),
):
    try:
        target_file = image or file
        depth_val = depth_meters if depth_meters is not None else phone_depth_reading

        image_bytes = await target_file.read()
        frame = decode_uploaded_frame(image_bytes)
        if frame is None:
            return {"status": "error", "message": "تعذر قراءة الصورة المرسلة."}

        if mode == "safety":
            nav_agent = session_manager.get_nav_agent(session_id)
            if not nav_agent.lidar_active:
                nav_agent.toggle_lidar()

            crowd_info, fused_obstacles = vision_nav_agent.process_frame_fused(
                frame, phone_depth_reading=depth_val
            )
            tactical_action = nav_agent.evaluate_tactical_decision(fused_obstacles, crowd_info)

            depth_source = "ar_sensor" if depth_val is not None else "vision_estimate"

            if not fused_obstacles:
                return {
                    "status": "success",
                    "mode": "safety",
                    "has_obstacle": False,
                    "is_hazard": False,
                    "action": "CLEAR",
                    "tactical_message": tactical_action.text_message or "المسار آمن أمامك",
                    "message": "المسار آمن",
                    "depth_source": depth_source,
                    "obstacle_type": None,
                    "closest_distance_m": None,
                    "direction": None,
                    "vibration_pattern": "NONE",
                }

            primary = fused_obstacles[0]
            direction = "left" if primary.angle_deg < -10 else ("right" if primary.angle_deg > 10 else "center")
            direction_ar = "على اليسار" if direction == "left" else ("على اليمين" if direction == "right" else "أمامك")

            is_critical = tactical_action.action_type == "CRITICAL_STOP"

            return {
                "status": "success",
                "mode": "safety",
                "has_obstacle": True,
                "is_hazard": True,
                "action": "STOP" if is_critical else "AVOID",
                "depth_source": depth_source,
                "distance": round(primary.distance, 2),
                "closest_distance_m": round(primary.distance, 2),
                "direction": direction,
                "direction_ar": direction_ar,
                "tactical_message": tactical_action.text_message,
                "message": f"عائق على بُعد {round(primary.distance, 1)} متر {direction_ar}",
                "obstacle_type": primary.label,
                "vibration_pattern": tactical_action.vibration_pattern,
            }

        elif mode == "landmark":
            landmark_name = vision_agent.detect_landmark(frame)
            return {"status": "success", "mode": "landmark", "landmark_name": landmark_name, "description": landmark_name}

        elif mode == "crowd":
            crowd_info, fused_obstacles = vision_nav_agent.process_frame_fused(
                frame, phone_depth_reading=depth_val
            )

            density = crowd_info.get("density", "LOW") if isinstance(crowd_info, dict) else "LOW"
            people_count = crowd_info.get("people_count", 0) if isinstance(crowd_info, dict) else 0

            is_crowded = density in ["HIGH", "MEDIUM"] or people_count >= 3

            if is_crowded:
                msg = f"تزاحم في المسار (مرصود {people_count} أشخاص)، يرجى التمهل"
            else:
                msg = "المسار آمن، ولا يوجد تزاحم أمامك"

            return {
                "status": "success",
                "mode": "crowd",
                "is_crowded": is_crowded,
                "density": density,
                "people_count": people_count,
                "message": msg,
                "tactical_message": msg
            }

        else:
            return {"status": "error", "message": f"وضع غير مدعوم: {mode}"}

    except Exception as e:
        print(f"❌ خطأ أثناء معالجة الإطار: {e}")
        return {"status": "error", "message": str(e)}


# ==========================================
# 3. عدّاد الطواف والسعي
# ==========================================
@app.post("/api/tawaf/start")
async def tawaf_start(file: UploadFile = File(...), session_id: str = Depends(require_session_id)):
    nav_agent = session_manager.get_nav_agent(session_id)
    image_bytes = await file.read()
    frame = decode_uploaded_frame(image_bytes)
    if frame is None:
        return {"status": "error", "message": "تعذر قراءة الصورة."}
    ok, msg = nav_agent.start_tawaf(frame)
    return {"status": "success" if ok else "error", "message": msg, "current_lap": nav_agent.current_lap}


@app.post("/api/tawaf/update")
async def tawaf_update(file: UploadFile = File(...), session_id: str = Depends(require_session_id)):
    nav_agent = session_manager.get_nav_agent(session_id)
    image_bytes = await file.read()
    frame = decode_uploaded_frame(image_bytes)
    if frame is None:
        return {"status": "error", "message": "تعذر قراءة الصورة."}

    event = nav_agent.update_tawaf(frame)

    if event is None:
        landmark_result = vision_agent.landmark_matcher.detect(frame)
        landmark_event = nav_agent.check_landmark_lap(landmark_result)
        if landmark_event:
            event = landmark_event
            nav_agent.tawaf_counter.current_lap = landmark_event["lap"]

    return {"status": "success", "lap_event": event, "current_lap": nav_agent.current_lap}


@app.post("/api/sai/start")
async def sai_start(session_id: str = Depends(require_session_id)):
    nav_agent = session_manager.get_nav_agent(session_id)
    nav_agent.start_sai()
    return {"status": "success", "message": "بدأ تتبع السعي - توجه إلى المروة أولاً."}


@app.post("/api/sai/update")
async def sai_update(file: UploadFile = File(...), session_id: str = Depends(require_session_id)):
    nav_agent = session_manager.get_nav_agent(session_id)
    image_bytes = await file.read()
    frame = decode_uploaded_frame(image_bytes)
    if frame is None:
        return {"status": "error", "message": "تعذر قراءة الصورة."}

    landmark_result = vision_agent.landmark_matcher.detect(frame)
    event = nav_agent.check_sai_lap(landmark_result)
    return {"status": "success", "lap_event": event, "current_lap": nav_agent.sai_counter.current_lap}


@app.post("/api/tawaf/manual")
async def tawaf_manual(direction: int = 1, session_id: str = Depends(require_session_id)):
    nav_agent = session_manager.get_nav_agent(session_id)
    result = nav_agent.manual_correct_lap(direction)
    return {"status": "success", "result": result}


# ==========================================
# 4. التعرف على المعالم والخريطة
# ==========================================
@app.post("/api/landmark/detect")
async def landmark_detect(file: UploadFile = File(...), session_id: str = Depends(require_session_id)):
    image_bytes = await file.read()
    frame = decode_uploaded_frame(image_bytes)
    if frame is None:
        return {"status": "error", "message": "تعذر قراءة الصورة."}
    landmark_name = vision_agent.detect_landmark(frame)
    detailed = vision_agent.detect_landmark_detailed(frame)
    return {"status": "success", "landmark": landmark_name, "details": detailed}


@app.post("/api/spatial-map")
async def spatial_map(file: UploadFile = File(...), session_id: str = Depends(require_session_id)):
    nav_agent = session_manager.get_nav_agent(session_id)
    image_bytes = await file.read()
    frame = decode_uploaded_frame(image_bytes)
    if frame is None:
        return {"status": "error", "message": "تعذر قراءة الصورة."}

    crowd_info, fused_obstacles = vision_nav_agent.process_frame_fused(frame)
    detection_info = {"fused_obstacles": fused_obstacles}

    if fused_obstacles:
        primary = fused_obstacles[0]
        relative_x = (primary.angle_deg / 30.0) * 2.0
        nav_agent.spatial_map.update_obstacle_position(primary.label, relative_x, primary.distance)

    grid_image = nav_agent.spatial_map.generate_grid_image(detection_info)
    success, encoded_png = cv2.imencode(".png", grid_image)
    if not success:
        return {"status": "error", "message": "تعذر توليد صورة الخريطة."}

    return Response(content=encoded_png.tobytes(), media_type="image/png")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)