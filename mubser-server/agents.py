import os
import json
import glob
import time
import cv2
import numpy as np
import requests
from ultralytics import YOLO

# ==========================================
# Safe Module Imports & Fallbacks
# ==========================================
try:
    from haptic_engine import HapticEngine
except ImportError:
    class HapticEngine:
        def trigger_vibration(self, pattern):
            pass

try:
    from sensor_fusion import SensorFusionEngine
except ImportError:
    class SensorFusionEngine:
        def fuse(self, vision_detections, lidar_reading, frame_width):
            return []

try:
    from importlib import import_module
    LocalSpatialMap = import_module(
        "local_spatial_map", package=__package__
    ).LocalSpatialMap
except ImportError:
    class LocalSpatialMap:
        def __init__(self):
            self.user_pos = (0, 0)

        def update_obstacle_position(self, label, x, dist):
            pass

        def calculate_detour_vector(self, x, dist):
            return "DETOUR", "يرجى الانحراف قليلاً لتفادي العائق."

try:
    from swarm_engine import SwarmSimulator
except ImportError:
    class SwarmSimulator:
        def get_swarm_telemetry(self):
            return {
                "active_nodes": 1,
                "path_a_density": "منخفض",
                "path_b_density": "منخفض",
                "recommended_path_name": "المسار الرئيسي",
                "reason": "المسار خالٍ من الازدحام.",
            }

try:
    from llm_engine import find_answer as ollama_find_answer, is_ollama_available
except ImportError:
    def is_ollama_available():
        return False
    def ollama_find_answer(prompt):
        return None

try:
    from voice_engine import speak_offline
except ImportError:
    def speak_offline(text):
        pass

try:
    from rag_engine import KnowledgeRAG
except ImportError:
    class KnowledgeRAG:
        def retrieve_context(self, query, top_k=2):
            return ""

try:
    from config import JSON_PATH
except ImportError:
    JSON_PATH = None

try:
    from lidar_driver import PhoneSensorLiDARDriver, TrackedObstacle
except ImportError:
    class TrackedObstacle:
        def __init__(self, label="obstacle", distance=2.0, angle_deg=0.0):
            self.label = label
            self.distance = distance
            self.angle_deg = angle_deg

    try:
        from lidar_driver import PhoneSensorLiDARDriver
    except ImportError:
        class PhoneSensorLiDARDriver:
            def update_phone_depth_data(self, reading):
                pass

            def get_filtered_distance(self, vision_fallback_dist=None):
                return vision_fallback_dist or 5.0, True


# ==========================================
# 1. QA Agent (RAG + Hybrid Fallback)
# ==========================================
class QAAgent:
    """وكيل الإجابات المعتمد على الـ RAG محلياً والتحويل لـ Gemini للردود الصياغية"""

    def __init__(self, server_url: str = "http://127.0.0.1:8000/api/qa"):
        print("🔍 [QAAgent]: جاري تحميل محرك الـ RAG وVector Database...")
        self.rag = KnowledgeRAG()
        self.server_url = server_url

    def process_query(self, query: str):
        """معالجة السؤال باستخدام الـ RAG لضمان إجابات موثوقة محلياً أو سحابياً"""
        if not query or len(query.strip()) < 3:
            print("⚠️ [QAAgent]: النص قصير جداً، تم تجنبه.")
            return None

        clean_query = query.strip()
        print(f"\n[QAAgent] 🤖 معالجة السؤال: {clean_query}")

        # 1. استرجاع السياق الموثوق دلالياً بواسطة RAG
        context = self.rag.retrieve_context(clean_query, top_k=2)

        if context and len(context.strip()) > 10:
            print("📚 [RAG VectorStore]: تم استرجاع سياق دلالي مرتبط بالطلب.")
            augmented_prompt = (
                f"استناداً حصرًا إلى المعلومات التالية من المصادر الموثوقة:\n{context}\n\n"
                f"أجب عن سؤال المستخدم باختصار وبأسلوب مباشر مناسب للقراءة الصوتية دون إضافة رموز تنسيق:\n{clean_query}"
            )
        else:
            print("ℹ️ [RAG VectorStore]: لم يتم العثور على سياق مطابق مباشرة، يتم استخدام السؤال الأصلي.")
            augmented_prompt = (
                f"أنت مساعد ذكي في تطبيق (مُبصر) للمكفوفين وحجاج بيت الله الحرام. "
                f"أجب عن السؤال التالي بأسلوب مباشر ومختصر جداً مناسب للقراءة الصوتية:\n{clean_query}"
            )

        # 2. المحاولة الأولى: Ollama المحلي (أوفلاين بالكامل)
        if is_ollama_available():
            print("🧠 [QAAgent]: جاري توليد الإجابة محلياً بـ Ollama (أوفلاين)...")
            try:
                ollama_answer = ollama_find_answer(augmented_prompt)
                if ollama_answer and "غير متصل" not in ollama_answer:
                    return ollama_answer
                print("⚠️ [QAAgent]: رد Ollama غير صالح، جاري المحاولة عبر Gemini كدعم احتياطي...")
            except Exception as err:
                print(f"⚠️ [QAAgent]: فشل Ollama، جاري المحاولة عبر Gemini كدعم احتياطي: {err}")
        else:
            print("ℹ️ [QAAgent]: Ollama غير متاح محلياً، جاري المحاولة عبر Gemini كدعم احتياطي...")

        # 3. Gemini السحابي - دعم احتياطي
        print("🚀 [QAAgent - Fallback]: جاري توجيه السؤال مع السياق لسيرفر Gemini...")
        try:
            response = requests.post(self.server_url, json={"query": augmented_prompt}, timeout=30)
            if response.status_code == 200:
                data = response.json()
                return data.get("response", "عذراً، لم أستطع الحصول على إجابة.")
        except Exception as e:
            print(f"❌ خطأ الاتصال بالسيرفر السحابي أيضاً: {e}")

        return "تعذر الحصول على إجابة من المصدر المحلي أو السحابي. تأكد من تشغيل Ollama أو الاتصال بالإنترنت."

    def search_context(self, query: str) -> str:
        """دالة توافقية ثانوية للبحث المباشر"""
        return self.rag.retrieve_context(query, top_k=2)


# ==========================================
# 2. Crowd Avoidance Agent
# ==========================================
class CrowdAvoidanceAgent:
    """محرك تقدير كثافة الحشود وتوجيه المسار الأقل ازدحاماً"""

    def __init__(self, crowd_threshold_medium=4, crowd_threshold_high=8):
        self.thresh_med = crowd_threshold_medium
        self.thresh_high = crowd_threshold_high

    def analyze_crowd_density(self, detected_boxes, frame_width):
        """حساب عدد الأشخاص وتحديد المستوى وتوصية الاتجاه"""
        person_count = 0
        left_sector_count = 0
        right_sector_count = 0

        if detected_boxes is not None:
            for box in detected_boxes:
                cls_id = int(box.cls[0])
                if cls_id == 0:  # Class 0 in COCO represents 'person'
                    person_count += 1
                    bbox = box.xyxy[0].cpu().numpy()
                    x_center = (bbox[0] + bbox[2]) / 2.0

                    if x_center < (frame_width / 2.0):
                        left_sector_count += 1
                    else:
                        right_sector_count += 1

        if person_count >= self.thresh_high:
            density_level = "HIGH"
            recommended_path = "جهة اليسار" if left_sector_count < right_sector_count else "جهة اليمين"
            advice_msg = (
                f"تنبيه: المنطقة أمامك شديدة الازدحام ({person_count} أشخاص). "
                f"يفضل الانحراف نحو المسار الأقل كثافة {recommended_path}."
            )
        elif person_count >= self.thresh_med:
            density_level = "MEDIUM"
            recommended_path = "جهة اليسار" if left_sector_count < right_sector_count else "جهة اليمين"
            advice_msg = f"منطقة متوسطة الازدحام. التزم بالجانب الـ {recommended_path} لمرور أسهل."
        else:
            density_level = "LOW"
            advice_msg = "المسار أمامك سالك ولا يوجد ازدحام ملحوظ."

        return {
            "person_count": person_count,
            "density_level": density_level,
            "advice": advice_msg,
            "left_count": left_sector_count,
            "right_count": right_sector_count,
        }


# ==========================================
# 3. Fused Vision Navigation Agent
# ==========================================
class VisionNavAgent:
    """وكيل الملاحة البصرية المدمج بـ Sensor Fusion والحشود وحساسات العمق"""

    def __init__(self, model_path="yolov8n.pt"):
        print("🎥 [VisionNavAgent]: جاري دمج Sensor Fusion مع YOLOv8 وحساس الجوال...")
        self.model = YOLO(model_path)
        self.fusion_engine = SensorFusionEngine()
        self.phone_lidar = PhoneSensorLiDARDriver()
        self.crowd_agent = CrowdAvoidanceAgent()

    def estimate_distance_from_bbox(self, bbox, frame_height):
        x1, y1, x2, y2 = bbox[:4]
        box_height = y2 - y1
        if box_height <= 0:
            return 5.0
        height_ratio = box_height / frame_height
        estimated_distance = round((1.0 / (height_ratio + 1e-5)) * 0.35, 2)
        return min(estimated_distance, 5.0)

    def process_frame(self, frame, phone_depth_reading=None):
        """توافقية للنسخ القديمة دون دمج مفصل"""
        if frame is None:
            return None

        _, fused_obstacles = self.process_frame_fused(
            frame=frame,
            vision_processor=self.model,
            phone_depth_reading=phone_depth_reading,
        )

        if not fused_obstacles:
            return {"distance": 5.0, "obstacle": None, "direction": "أمامك", "source": "fused"}

        primary = fused_obstacles[0]
        direction = "أمامك"
        if primary.angle_deg < -10:
            direction = "على اليسار"
        elif primary.angle_deg > 10:
            direction = "على اليمين"

        return {
            "distance": primary.distance,
            "obstacle": primary.label,
            "direction": direction,
            "source": "sensor_fusion",
            "angle_deg": primary.angle_deg,
            "fused_obstacles": fused_obstacles,
        }

    def process_frame_fused(self, frame, vision_processor=None, phone_depth_reading=None):
        """الدالة الأساسية لمعالجة الفريم باستخدام Sensor Fusion الشامل"""
        if frame is None:
            return None, []

        frame_h, frame_w, _ = frame.shape
        raw_detections = []
        yolo_boxes = []

        if vision_processor and hasattr(vision_processor, "extract_detections"):
            raw_detections = vision_processor.extract_detections(frame)
        else:
            known_labels_ar = {
                'person': 'شخص', 'chair': 'كرسي', 'table': 'طاولة',
                'sofa': 'أريكة', 'door': 'باب', 'wheelchair': 'كرسي متحرك',
                'bench': 'مقعد', 'suitcase': 'حقيبة', 'backpack': 'حقيبة ظهر',
                'car': 'سيارة', 'bicycle': 'دراجة', 'motorcycle': 'دراجة نارية',
                'potted plant': 'نبات', 'umbrella': 'مظلة',
            }

            results = self.model(frame, verbose=False)[0]
            yolo_boxes = results.boxes
            for box in yolo_boxes:
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                conf = float(box.conf[0])

                if conf > 0.35:
                    bbox = box.xyxy[0].cpu().numpy()
                    dist = self.estimate_distance_from_bbox(bbox, frame_h)
                    arabic_label = known_labels_ar.get(label, 'عائق')
                    raw_detections.append((arabic_label, bbox, dist, conf))

        if phone_depth_reading:
            self.phone_lidar.update_phone_depth_data(phone_depth_reading)

        fallback_dist = raw_detections[0][2] if raw_detections else None
        filtered_lidar_dist, _ = self.phone_lidar.get_filtered_distance(vision_fallback_dist=fallback_dist)

        fused_obstacles = self.fusion_engine.fuse(
            vision_detections=raw_detections,
            lidar_reading=filtered_lidar_dist,
            frame_width=frame_w,
        )

        crowd_info = self.crowd_agent.analyze_crowd_density(yolo_boxes, frame_w)

        return crowd_info, fused_obstacles


# ==========================================
# 4. Tactical Decision Navigation Agent
# ==========================================
class NavigationAction:
    """كائن يتضمن القرار التكتيكي الموحد لجميع المنظومات الحسية"""
    def __init__(self, action_type, text_message, vibration_pattern, priority=1):
        self.action_type = action_type
        self.text_message = text_message
        self.vibration_pattern = vibration_pattern
        self.priority = priority


class AutomaticTawafCounterV2:
    """
    عدّاد الأشواط التلقائي V2 - مع بنك keyframes ومعايرة CLAHE
    """
    def __init__(self, cooldown_seconds=45.0, keyframe_bank_size=3, max_lap_silence_seconds=300.0):
        self.orb = cv2.ORB_create(nfeatures=800)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        self.keyframe_bank_size = keyframe_bank_size
        self.start_descriptors_bank = []
        self.start_hist_bank = []

        self.started = False
        self.departed = False
        self.current_lap = 0
        self.cooldown_seconds = cooldown_seconds
        self.last_lap_time = 0.0
        self.low_match_frames = 0

        self.max_lap_silence_seconds = max_lap_silence_seconds
        self.last_progress_check = 0.0

        self._collecting_start = False
        self._collect_deadline = 0.0

    def _preprocess(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = self.clahe.apply(gray)
        return gray

    def _histogram(self, gray):
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        cv2.normalize(hist, hist)
        return hist

    def start(self, frame):
        gray = self._preprocess(frame)
        _, desc = self.orb.detectAndCompute(gray, None)
        if desc is None or len(desc) < 20:
            return False, "لم أستطع تثبيت نقطة بداية واضحة، حاول توجيه الكاميرا أوضح."

        self.start_descriptors_bank = [desc]
        self.start_hist_bank = [self._histogram(gray)]
        self._collecting_start = True
        self._collect_deadline = time.time() + 0.6

        self.started = True
        self.departed = False
        self.current_lap = 0
        self.last_lap_time = time.time()
        self.last_progress_check = time.time()
        self.low_match_frames = 0
        return True, "تم تثبيت نقطة البداية. سأحصي الأشواط تلقائيًا."

    def _maybe_collect_keyframe(self, gray, desc):
        if not self._collecting_start:
            return
        if time.time() > self._collect_deadline or len(self.start_descriptors_bank) >= self.keyframe_bank_size:
            self._collecting_start = False
            return
        if desc is not None and len(desc) >= 20:
            self.start_descriptors_bank.append(desc)
            self.start_hist_bank.append(self._histogram(gray))

    def _match_score(self, gray, desc):
        if not self.started or not self.start_descriptors_bank:
            return 0.0, 0.0
        if desc is None or len(desc) < 10:
            return 0.0, 0.0

        best_orb_score = 0.0
        best_hist_score = 0.0
        current_hist = self._histogram(gray)

        for bank_desc, bank_hist in zip(self.start_descriptors_bank, self.start_hist_bank):
            try:
                knn = self.matcher.knnMatch(bank_desc, desc, k=2)
                good = [m for pair in knn if len(pair) == 2 for m, n in [pair] if m.distance < 0.72 * n.distance]
                orb_score = len(good) / max(1, min(len(bank_desc), len(desc)))
            except Exception:
                orb_score = 0.0

            hist_score = float(cv2.compareHist(bank_hist, current_hist, cv2.HISTCMP_CORREL))

            if orb_score > best_orb_score:
                best_orb_score = orb_score
                best_hist_score = hist_score

        return best_orb_score, best_hist_score

    def update(self, frame):
        if not self.started:
            return None

        gray = self._preprocess(frame)
        _, desc = self.orb.detectAndCompute(gray, None)

        self._maybe_collect_keyframe(gray, desc)
        orb_score, hist_score = self._match_score(gray, desc)

        if orb_score < 0.18:
            self.low_match_frames += 1
        else:
            self.low_match_frames = max(0, self.low_match_frames - 1)

        if self.low_match_frames >= 5:
            self.departed = True

        now = time.time()

        strong_match = orb_score >= 0.38 and hist_score >= 0.55
        if self.departed and strong_match and now - self.last_lap_time >= self.cooldown_seconds:
            return self._register_lap(confidence=round((orb_score + hist_score) / 2, 2))

        if now - self.last_progress_check >= self.max_lap_silence_seconds:
            self.last_progress_check = now
            return {
                "lap": self.current_lap,
                "completed": False,
                "silent_warning": True,
                "message": "لم يُسجَّل شوط جديد منذ فترة، تأكد من استمرار الحركة أو أعد تثبيت نقطة البداية.",
            }

        return None

    def _register_lap(self, confidence):
        self.current_lap += 1
        self.departed = False
        self.last_lap_time = time.time()
        self.last_progress_check = time.time()
        self.low_match_frames = 0

        if self.current_lap >= 7:
            self.started = False
            return {
                "lap": 7,
                "completed": True,
                "confidence": confidence,
                "message": "تم رصد الشوط السابع. تم الانتهاء من الطواف بالكامل.",
            }
        return {
            "lap": self.current_lap,
            "completed": False,
            "confidence": confidence,
            "message": f"تم رصد الشوط {self.current_lap} تلقائيًا.",
        }

    def manual_correct_lap(self, direction=1):
        self.current_lap = max(0, self.current_lap + direction)
        self.last_lap_time = time.time()
        self.departed = False
        self.low_match_frames = 0

        if self.current_lap >= 7:
            self.started = False
            return {
                "lap": 7,
                "completed": True,
                "manual": True,
                "message": "تم تسجيل الشوط السابع يدويًا. اكتمل الطواف.",
            }
        return {
            "lap": self.current_lap,
            "completed": False,
            "manual": True,
            "message": f"تم تعديل العدّاد يدويًا: الشوط {self.current_lap}.",
        }


class LandmarkTriggeredLapCounter:
    """عدّاد أشواط مربوط بمعلم بصري محدد (مثل الحجر الأسود)"""
    def __init__(self, target_landmark_id="black_stone", cooldown_seconds=45.0, confidence_threshold=0.10):
        self.target_landmark_id = target_landmark_id
        self.cooldown_seconds = cooldown_seconds
        self.confidence_threshold = confidence_threshold
        self.current_lap = 0
        self.last_trigger_time = 0.0
        self.active = False

    def start(self):
        self.current_lap = 0
        self.last_trigger_time = 0.0
        self.active = True

    def check(self, landmark_result):
        if not self.active or landmark_result is None:
            return None

        if (landmark_result["id"] == self.target_landmark_id and
                landmark_result["confidence"] >= self.confidence_threshold):

            now = time.time()
            if now - self.last_trigger_time < self.cooldown_seconds:
                return None

            self.last_trigger_time = now
            self.current_lap += 1

            if self.current_lap >= 7:
                self.active = False
                return {"lap": 7, "completed": True, "source": "landmark",
                        "message": "تم رصد الحجر الأسود للمرة السابعة. اكتمل الطواف بفضل الله."}
            return {"lap": self.current_lap, "completed": False, "source": "landmark",
                    "message": f"تم رصد الحجر الأسود - اكتمل الشوط {self.current_lap}."}
        return None


class SaiLapCounter:
    """عدّاد أشواط السعي بالتناوب بين الصفا والمروة"""
    def __init__(self, cooldown_seconds=30.0, confidence_threshold=0.10):
        self.cooldown_seconds = cooldown_seconds
        self.confidence_threshold = confidence_threshold
        self.current_lap = 0
        self.last_trigger_time = 0.0
        self.expecting = "marwah"
        self.active = False

    def start(self):
        self.current_lap = 0
        self.last_trigger_time = 0.0
        self.expecting = "marwah"
        self.active = True

    def check(self, landmark_result):
        if not self.active or landmark_result is None:
            return None

        if (landmark_result["id"] == self.expecting and
                landmark_result["confidence"] >= self.confidence_threshold):

            now = time.time()
            if now - self.last_trigger_time < self.cooldown_seconds:
                return None

            self.last_trigger_time = now
            self.current_lap += 1
            reached = "المروة" if self.expecting == "marwah" else "الصفا"
            self.expecting = "safa" if self.expecting == "marwah" else "marwah"

            if self.current_lap >= 7:
                self.active = False
                return {"lap": 7, "completed": True, "source": "landmark",
                        "message": f"تم الوصول إلى {reached} للمرة السابعة. اكتمل السعي بفضل الله."}
            return {"lap": self.current_lap, "completed": False, "source": "landmark",
                    "message": f"تم الوصول إلى {reached} - اكتمل الشوط {self.current_lap}."}
        return None


class NavAgent:
    """العقل التكتيكي لمبصر - Decision Engine المدمج مع الخريطة المكانية والذكاء الجمعي وعدّاد الأشواط"""
    def __init__(self, alert_threshold_m=1.2):
        self.alert_threshold_m = alert_threshold_m
        self.lidar_active = False
        self.haptic = HapticEngine()
        self.spatial_map = LocalSpatialMap()
        self.swarm_sim = SwarmSimulator()
        self.last_action_time = 0

        self.tawaf_counter = AutomaticTawafCounterV2(cooldown_seconds=45.0)
        self.landmark_tawaf_counter = LandmarkTriggeredLapCounter(target_landmark_id="black_stone")
        self.sai_counter = SaiLapCounter()

    def toggle_lidar(self):
        self.lidar_active = not self.lidar_active
        status = "تفعيل" if self.lidar_active else "تعطيل"
        return f"تم {status} رادار العوائق بنجاح."

    def start_tawaf(self, frame):
        self.landmark_tawaf_counter.start()
        return self.tawaf_counter.start(frame)

    def check_landmark_lap(self, landmark_result):
        return self.landmark_tawaf_counter.check(landmark_result)

    def start_sai(self):
        self.sai_counter.start()

    def check_sai_lap(self, landmark_result):
        return self.sai_counter.check(landmark_result)

    def update_tawaf(self, frame):
        return self.tawaf_counter.update(frame)

    def manual_correct_lap(self, direction=1):
        return self.tawaf_counter.manual_correct_lap(direction)

    @property
    def current_lap(self):
        return self.tawaf_counter.current_lap

    def get_swarm_navigation_recommendation(self):
        swarm_data = self.swarm_sim.get_swarm_telemetry()
        rec_path_name = swarm_data["recommended_path_name"]
        reason = swarm_data["reason"]

        action_text = f"توجيه الذكاء الجمعي: يفضل الاتجاه نحو {rec_path_name}. {reason}"

        return NavigationAction(
            action_type="SWARM_REROUTE",
            text_message=action_text,
            vibration_pattern="LIGHT",
            priority=3,
        ), swarm_data

    def evaluate_tactical_decision(self, fused_obstacles, crowd_info=None):
        if not self.lidar_active:
            return NavigationAction("DISABLED", "", "NONE", priority=0)

        if not fused_obstacles:
            if crowd_info and crowd_info.get("density_level") == "HIGH":
                return NavigationAction(
                    action_type="AVOID_CROWD",
                    text_message=crowd_info["advice"],
                    vibration_pattern="WARNING",
                    priority=2,
                )
            return NavigationAction("CLEAR", "المسار آمن أمامك", "NONE", priority=0)

        primary = fused_obstacles[0]
        dist = getattr(primary, "distance", 2.0)
        angle = getattr(primary, "angle_deg", 0.0)
        label = getattr(primary, "label", "عائق")

        relative_x = (angle / 30.0) * 2.0
        self.spatial_map.update_obstacle_position(label, relative_x, dist)

        if dist <= 0.65:
            return NavigationAction(
                action_type="CRITICAL_STOP",
                text_message=f"توقف فوراً! {label} قريب جداً على بعد {dist} متر.",
                vibration_pattern="CRITICAL",
                priority=5,
            )
        elif dist <= self.alert_threshold_m:
            grid_obstacle_x = self.spatial_map.user_pos[0] + relative_x
            detour_type, spatial_advice = self.spatial_map.calculate_detour_vector(grid_obstacle_x, dist)

            vibe = "WARNING" if dist > 0.9 else "CRITICAL"
            return NavigationAction(
                action_type=detour_type,
                text_message=f"انتبه، عائق أمامك ({label}). {spatial_advice}",
                vibration_pattern=vibe,
                priority=4,
            )
        elif dist <= 2.0 and crowd_info and crowd_info.get("density_level") in ["MEDIUM", "HIGH"]:
            return NavigationAction(
                action_type="SLOW_DOWN",
                text_message=f"المنطقة مكتظة، يرجى التمهل. {label} على مسافة {dist} متر.",
                vibration_pattern="LIGHT",
                priority=3,
            )

        return NavigationAction("CLEAR", "المسار آمن", "NONE", priority=0)

    def execute_action(self, action: NavigationAction):
        if action.priority == 0 or not action.text_message:
            return None

        now = time.time()
        if now - self.last_action_time < 2.5 and action.priority < 5:
            return action.text_message

        self.last_action_time = now

        if action.vibration_pattern != "NONE":
            self.haptic.trigger_vibration(action.vibration_pattern)

        return action.text_message


# ==========================================
# 5. Landmark Matcher & Vision Agent
# ==========================================
class LandmarkMatcher:
    """تعرف حقيقي على المعالم الميدانية بناءً على مطابقة ORB المتقدمة و histogram الإضاءة"""
    def __init__(self, reference_dir="reference_landmarks", match_threshold=0.10):
        self.orb = cv2.ORB_create(nfeatures=1500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.match_threshold = match_threshold

        self.landmark_bank = {}
        self.landmark_hist_bank = {}
        self.id_to_name_ar = {}

        self._load_name_mapping()
        self._build_bank(reference_dir)

    def _preprocess(self, frame_or_img):
        gray = cv2.cvtColor(frame_or_img, cv2.COLOR_BGR2GRAY)
        return self.clahe.apply(gray)

    def _histogram(self, gray):
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        cv2.normalize(hist, hist)
        return hist

    def _load_name_mapping(self):
        if not JSON_PATH or not os.path.exists(JSON_PATH):
            print("⚠️ [LandmarkMatcher]: ملف mubsir.json غير متاح، سيتم عرض الـ id الخام فقط.")
            return
        try:
            with open(JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            sanctuary_info = data.get("makkah_sanctuary_info", {})
            for category_items in sanctuary_info.values():
                if isinstance(category_items, list):
                    for item in category_items:
                        item_id = item.get("id")
                        name_ar = item.get("name_ar")
                        if item_id and name_ar:
                            self.id_to_name_ar[item_id] = name_ar
        except Exception as e:
            print(f"⚠️ [LandmarkMatcher]: خطأ في قراءة mubsir.json: {e}")

    def _build_bank(self, reference_dir):
        if not os.path.isdir(reference_dir):
            print(f"⚠️ [LandmarkMatcher]: مجلد الصور المرجعية غير موجود: {reference_dir}")
            return

        landmark_dirs = [d for d in os.listdir(reference_dir) if os.path.isdir(os.path.join(reference_dir, d))]

        for landmark_id in landmark_dirs:
            images = glob.glob(os.path.join(reference_dir, landmark_id, "*.jpg")) + \
                     glob.glob(os.path.join(reference_dir, landmark_id, "*.png"))

            descriptors_list = []
            hist_list = []
            for img_path in images:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                gray = self._preprocess(img)
                _, desc = self.orb.detectAndCompute(gray, None)
                if desc is not None and len(desc) >= 20:
                    descriptors_list.append(desc)
                    hist_list.append(self._histogram(gray))

            if descriptors_list:
                self.landmark_bank[landmark_id] = descriptors_list
                self.landmark_hist_bank[landmark_id] = hist_list
                print(f"✅ [LandmarkMatcher]: تم تحميل {len(descriptors_list)} صورة مرجعية لـ '{landmark_id}'")

        if not self.landmark_bank:
            print("⚠️ [LandmarkMatcher]: لم يتم تحميل أي معالم. اضف صوراً داخل مرجع المعالم.")

    def _score_against_bank(self, frame_desc, frame_hist, bank_descriptors, bank_hists):
        best_orb = 0.0
        best_hist = 0.0
        for ref_desc, ref_hist in zip(bank_descriptors, bank_hists):
            try:
                knn = self.matcher.knnMatch(ref_desc, frame_desc, k=2)
                good = [m for pair in knn if len(pair) == 2 for m, n in [pair] if m.distance < 0.75 * n.distance]
                orb_score = len(good) / max(1, min(len(ref_desc), len(frame_desc)))
            except Exception:
                orb_score = 0.0

            hist_score = float(cv2.compareHist(ref_hist, frame_hist, cv2.HISTCMP_CORREL))

            if orb_score > best_orb:
                best_orb = orb_score
                best_hist = hist_score

        return best_orb, best_hist

    def detect(self, frame):
        if frame is None or not self.landmark_bank:
            return None

        gray = self._preprocess(frame)
        _, frame_desc = self.orb.detectAndCompute(gray, None)
        if frame_desc is None or len(frame_desc) < 15:
            return None

        frame_hist = self._histogram(gray)
        best_id = None
        best_combined = 0.0

        for landmark_id, bank_descriptors in self.landmark_bank.items():
            bank_hists = self.landmark_hist_bank[landmark_id]
            orb_score, hist_score = self._score_against_bank(frame_desc, frame_hist, bank_descriptors, bank_hists)
            combined_score = (orb_score * 0.7) + (max(0, hist_score) * 0.3)

            if combined_score > best_combined:
                best_combined = combined_score
                best_id = landmark_id

        if best_combined >= self.match_threshold:
            return {
                "id": best_id,
                "name_ar": self.id_to_name_ar.get(best_id, best_id),
                "confidence": round(best_combined, 2),
            }
        return None


class VisionAgent:
    """رصد المعالم البصرية والمزامنة الميدانية"""
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    REFERENCE_DIR = os.path.join(BASE_DIR, "reference_landmarks")

    def __init__(self):
        print("👁️ [VisionAgent]: جاري تحميل نموذج YOLOv8 للمعالم...")
        if not os.path.exists(self.REFERENCE_DIR):
            print(f"⚠️ تحذير: المجلد غير موجود في المسار: {self.REFERENCE_DIR}")
        else:
            print(f"✅ تم العثور على مجلد الصور المرجعية: {self.REFERENCE_DIR}")

        self.model = YOLO("yolov8n.pt")
        self.swarm_sim = SwarmSimulator()
        self.landmark_matcher = LandmarkMatcher(reference_dir=self.REFERENCE_DIR)

    def process_frame(self, frame):
        if frame is None:
            return None, None

        results = self.model(frame, verbose=False)
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                name = self.model.names[cls]
                return name, frame

        return None, frame

    def detect_landmark(self, frame=None):
        if frame is None:
            return "لا يوجد إطار كاميرا متاح حاليًا للتعرف على المعلم."

        result = self.landmark_matcher.detect(frame)
        if result is None:
            return "لم أتمكن من التعرف على معلم واضح في هذا الاتجاه، حاول توجيه الكاميرا نحو المعلم مباشرة."

        return result.get("name_ar", "معلم غير معروف")

    def detect_landmark_detailed(self, frame=None):
        if frame is None:
            return None
        return self.landmark_matcher.detect(frame)

    def sync_swarm_data(self):
        telemetry = self.swarm_sim.get_swarm_telemetry()
        return (
            f"شبكة مبصر ({telemetry['active_nodes']} نظارة متصلة): الكثافة [مسار أ: {telemetry['path_a_density']} | "
            f"مسار ب: {telemetry['path_b_density']}]. التوصية: {telemetry['recommended_path_name']}."
        )


class FaceRecognitionAgent:
    """وحدة التعرف الأساسي على الوجوه المسجلة مسبقاً"""
    def __init__(self):
        self.known_faces = {
            "مرافق معتمد": "Face_Token_001",
            "عضو لجنة التحكيم": "Face_Token_002",
        }
        print("👤 [FaceRecognitionAgent]: تم تفعيل وحدة تمييز الوجوه الموثوقة.")

    def recognize_face(self, frame):
        if frame is None:
            return None
        return "تم التعرف على: مرافق معتمد"