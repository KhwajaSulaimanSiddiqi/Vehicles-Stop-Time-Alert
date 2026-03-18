import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
import sys

# --- تنظیمات ---
VIDEO_PATH = "33.mp4"
MODEL_PATH = "traffic_analysis.pt"
COCO_PATH = "coco_for_dron.txt"  # فایل حاوی نام کلاس‌ها (هر خط یک نام)
OUTPUT_VIDEO_PATH = "output_video.mp4"  # مسیر ذخیره ویدئوی خروجی
ZONE_POLYGON = np.array([[1955, 286], [1850, 355], [1721, 396], [1588, 414], [1386, 424], [1212, 412], [1037, 381], [823, 314],
                          [583, 734], [733, 818], [830, 890], [933, 984], [1000, 1086], [1062, 1214], [1120, 1388], [1177, 1582], 
                          [1192, 1731], [1192, 1810], [1489, 1932], [1651, 1815], [1773, 1771], [1940, 1748], 
                          [2100, 1736], [2259, 1751], [2449, 1794], [2730, 1288], [2586, 1161], [2476, 989], 
                          [2439, 877], [2409, 759], [2389, 657], [2389, 555], [2389, 488]])
ZONE_OPACITY = 0.3

def get_color(class_id: int) -> tuple:
    """تولید رنگ یکتا و پایدار بر اساس class_id"""
    np.random.seed(class_id)
    return tuple(int(x) for x in np.random.randint(0, 255, 3))

# بارگذاری کلاس‌ها از فایل COCO
VEHICLE_CLASSES = {}
try:
    with open(COCO_PATH, 'r') as f:
        for idx, line in enumerate(f):
            class_name = line.strip()
            if class_name:
                VEHICLE_CLASSES[idx] = (class_name, get_color(idx))
    print(f"✅ {len(VEHICLE_CLASSES)} کلاس از {COCO_PATH} بارگذاری شد.")
except FileNotFoundError:
    print(f"⚠️ فایل {COCO_PATH} یافت نشد. از کلاس‌های پیش‌فرض استفاده می‌شود.")
    # مقدار پیش‌فرض (در صورت نبود فایل)
    VEHICLE_CLASSES = {
        1: ("bicycle", (255, 0, 0)),
        2: ("car", (0, 255, 255)),
        3: ("motorcycle", (0, 165, 255)),
        4: ("airplane", (147, 20, 255)),
        5: ("bus", (0, 255, 0)),
        6: ("train", (255, 0, 255)),
        7: ("truck", (0, 0, 255)),
        8: ("boat", (255, 255, 0)),
    }

class CustomTrafficAnnotator:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_DUPLEX

    def annotate(self, scene, detections, overstay_data):
        for i in range(len(detections)):
            coords = detections.xyxy[i]
            track_id = detections.tracker_id[i]
            class_id = detections.class_id[i]

            if class_id not in VEHICLE_CLASSES:
                continue
            class_name, color = VEHICLE_CLASSES[class_id]

            cx = int((coords[0] + coords[2]) / 2)
            cy = int((coords[1] + coords[3]) / 2)

            # تشخیص وضعیت هشدار (بیش از 60 فریم در zona)
            is_alert = (track_id in overstay_data and overstay_data[track_id] >= 60)
            line_color = (0, 0, 255) if is_alert else color

            # رسم نشانگر
            line_len1 = 20
            line_len2 = 20
            p1 = (cx, cy)
            p2 = (cx - line_len1, cy - line_len1)
            p3 = (p2[0] - line_len2, p2[1])

            cv2.line(scene, p1, p2, line_color, 2)
            cv2.line(scene, p2, p3, line_color, 2)
            cv2.circle(scene, (cx, cy), 6, line_color, -1)

            if is_alert:
                duration = overstay_data[track_id]
                self.draw_red_alert_box(scene, p3, class_name, duration)
                self.draw_alert_triangle(scene, (cx, cy), duration)
            else:
                # نمایش نام کلاس در حالت عادی
                display_text = class_name
                (tw, th), _ = cv2.getTextSize(display_text, self.font, 0.6, 2)
                text_x = p3[0] - 10
                text_y = p3[1] - 10
                cv2.rectangle(scene,
                              (text_x - 5, text_y - th - 5),
                              (text_x + tw + 5, text_y + 5),
                              color, -1)
                cv2.putText(scene, display_text, (text_x, text_y),
                            self.font, 0.6, (0, 0, 0), 2)

    def draw_red_alert_box(self, scene, pos, class_name, duration):
        x, y = pos
        mins, sec = divmod(int(duration), 60)
        time_str = f"{mins}m {sec}s"

        font_scale_alert1 = 0.6
        font_scale_alert2 = 0.5
        font_scale_time = 0.5
        thickness = 2

        alert_text1 = "alert"
        alert_text2 = class_name
        alert_text3 = f"overstay : {time_str}"
        
        (tw_alert1, th_alert1), _ = cv2.getTextSize(alert_text1, self.font, font_scale_alert1, thickness)
        (tw_alert2, th_alert2), _ = cv2.getTextSize(alert_text2, self.font, font_scale_alert2, thickness)
        (tw_alert3, th_alert3), _ = cv2.getTextSize(alert_text3, self.font, font_scale_time, thickness)

        max_tw = max(tw_alert1, tw_alert2, tw_alert3)
        total_th = th_alert1 + th_alert2 + th_alert3 + 20
        padding = 25
        gap_from_line = 10
        box_h = total_th + 20
        box_w = max_tw + 2 * padding
        box_x = x - padding
        box_y = y - total_th - 20 - gap_from_line

        # رسم باکس قرمز
        cv2.rectangle(scene, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 255), -1)
        cv2.rectangle(scene, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), 2)

        # محاسبه موقعیت مرکزی برای هر خط
        # خط اول: alert
        text_x1 = box_x + (box_w - tw_alert1) // 2
        text_y1 = box_y + th_alert1 + 10
        
        # خط دوم: نوع کلاس
        text_x2 = box_x + (box_w - tw_alert2) // 2
        text_y2 = text_y1 + th_alert2 + 5
        
        # خط سوم: overstay : زمان
        text_x3 = box_x + (box_w - tw_alert3) // 2
        text_y3 = text_y2 + th_alert3 + 5

        # نوشتن متن‌ها در مرکز
        cv2.putText(scene, alert_text1, (text_x1, text_y1), self.font, font_scale_alert1, (0, 0, 0), thickness)
        cv2.putText(scene, alert_text2, (text_x2, text_y2), self.font, font_scale_alert2, (0, 0, 0), thickness)
        cv2.putText(scene, alert_text3, (text_x3, text_y3), self.font, font_scale_time, (0, 0, 0), thickness)

    def draw_alert_triangle(self, scene, center, duration):
        cx, cy = center
        triangle_half_width = 20        # نصف‌پهنا (قابل تنظیم)
        triangle_height = 2 * triangle_half_width   # ارتفاع = 2 * عرض کل (چون عرض کل = 2 * نصف‌پهنا)
        offset_y = 50
        t_y = cy + offset_y

        # رأس‌های مثلث (رأس بالا، چپ پایین، راست پایین)
        t_points = np.array([
            [cx, t_y - triangle_height],                     # رأس بالا
            [cx - triangle_half_width, t_y],                 # پایین چپ
            [cx + triangle_half_width, t_y]                  # پایین راست
        ], dtype=np.int32)

        cv2.fillPoly(scene, [t_points], (0, 0, 255))
        cv2.polylines(scene, [t_points], isClosed=True, color=(0, 0, 0), thickness=2)

        # محاسبه مرکز تقریبی مثلث برای قرار دادن علامت "!"
        centroid_x = cx
        centroid_y = t_y - triangle_height // 2

        text = "!"
        font_scale = 1.0
        thickness = 2
        (tw, th), _ = cv2.getTextSize(text, self.font, font_scale, thickness)
        text_x = centroid_x - tw // 2
        text_y = centroid_y + th // 2
        cv2.putText(scene, text, (text_x, text_y), self.font, font_scale, (0, 0, 0), thickness)


def draw_zone(frame, polygon, color=(30, 144, 255), opacity=0.3, thickness=2):
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon], color)
    cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)
    cv2.polylines(frame, [polygon], isClosed=True, color=color, thickness=thickness)
    return frame


# --- اجرای اصلی ---
model = YOLO(MODEL_PATH)
tracker = sv.ByteTrack()
zone = sv.PolygonZone(polygon=ZONE_POLYGON)

custom_annotator = CustomTrafficAnnotator()

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 30

# گرفتن ابعاد فریم برای تنظیم VideoWriter
ret, first_frame = cap.read()
if not ret:
    print("❌ خطا در خواندن ویدئو")
    sys.exit(1)
height, width = first_frame.shape[:2]
# برگرداندن فریم اول به ابتدای ویدئو
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# تعریف VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # کدک برای MP4
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

track_history = {}

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)

        # فقط کلاس‌های موجود در VEHICLE_CLASSES نگه داشته می‌شوند
        vehicle_class_ids = list(VEHICLE_CLASSES.keys())
        detections = detections[np.isin(detections.class_id, vehicle_class_ids)]

        detections = tracker.update_with_detections(detections)
        is_in_zone = zone.trigger(detections=detections)

        current_overstays = {}
        for i, in_zone in enumerate(is_in_zone):
            if detections.tracker_id is None:
                continue
            tid = detections.tracker_id[i]
            if in_zone:
                if tid not in track_history:
                    track_history[tid] = 0
                track_history[tid] += 1 / fps
                current_overstays[tid] = track_history[tid]
            else:
                if tid in track_history:
                    del track_history[tid]

        frame = draw_zone(frame, ZONE_POLYGON, color=(255, 144, 30), opacity=ZONE_OPACITY)
        custom_annotator.annotate(frame, detections, current_overstays)

        # ذخیره فریم در فایل خروجی
        out.write(frame)

        cv2.imshow("ByteTrack - Smart City", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as e:
    print(f"⚠️ خطا در حین پردازش: {e}")
finally:
    # آزادسازی منابع
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ ویدئوی خروجی تا آخرین فریم پردازش‌شده در {OUTPUT_VIDEO_PATH} ذخیره شد.")