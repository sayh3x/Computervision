# rppg_local.py
import cv2
import numpy as np
import time
from collections import deque
from scipy import signal

# ---------- پارامترها ----------
FPS = 30  # فریم ریت هدف (دقت کنید دوربین شما این فریم ریت را تحویل دهد)
BUFFER_SECONDS = 8  # طول بافر به ثانیه (برای FFT و پایداری)
BUFFER_SIZE = int(FPS * BUFFER_SECONDS)
PROCESS_INTERVAL = 2.0  # هر چند ثانیه سیگنال پردازش شود
MIN_HR = 40
MAX_HR = 200

# متدهای قابل انتخاب: 'GREEN', 'PCA', 'CHROM'
current_method = 'PCA'

# ---------- ابزار تشخیص صورت (Haar cascade از OpenCV) ----------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ---------- بافرها ----------
r_buffer = deque(maxlen=BUFFER_SIZE)
g_buffer = deque(maxlen=BUFFER_SIZE)
b_buffer = deque(maxlen=BUFFER_SIZE)
time_buffer = deque(maxlen=BUFFER_SIZE)
last_process_time = 0
hr_history = deque(maxlen=10)

# ---------- توابع کمکی ----------
def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(x, fs, lowcut=0.7, highcut=4.0, order=3):
    if len(x) < 3:
        return x
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    # استفاده از filtfilt برای جلوگیری از تأخیر فاز
    return signal.filtfilt(b, a, x)

def detrend_signal(x):
    return signal.detrend(x, type='linear')

def estimate_hr_from_signal(sig, fps):
    """محاسبه ضربان قلب از سیگنال تک‌بعدی با FFT"""
    sig = np.asarray(sig)
    N = len(sig)
    if N < 4:
        return None
    # حذف میانگین / DC
    sig = sig - np.mean(sig)
    # پنجره هنینگ
    win = np.hanning(N)
    sig_win = sig * win
    # FFT
    fft = np.fft.rfft(sig_win)
    fft_mag = np.abs(fft)
    freqs = np.fft.rfftfreq(N, d=1.0/fps)
    # بازه فرکانسی ضربان قلب
    mask = (freqs >= 0.7) & (freqs <= 4.0)
    if not np.any(mask):
        return None
    freqs_hr = freqs[mask]
    mag_hr = fft_mag[mask]
    if len(mag_hr) == 0:
        return None
    peak_idx = np.argmax(mag_hr)
    peak_freq = freqs_hr[peak_idx]
    hr_bpm = peak_freq * 60.0
    if MIN_HR <= hr_bpm <= MAX_HR:
        return hr_bpm
    return None

# ---------- الگوریتم‌های استخراج پالس ----------
def method_green(r, g, b):
    """سیگنال بر اساس کانال سبز"""
    return np.array(g)

def method_pca(r, g, b):
    """استخراج مؤلفه اصلی (PCA) از سه کانال رنگ"""
    X = np.vstack([r, g, b])  # شکل 3 x N
    # نرمال‌سازی کانال‌ها (نسبی)
    mean_channels = np.mean(X, axis=1, keepdims=True)
    # جلوگیری از تقسیم بر صفر
    mean_channels[mean_channels == 0] = 1.0
    Xn = X / mean_channels - 1.0  # تغییرات نسبی
    # حذف ترند زمانی
    Xn = signal.detrend(Xn, axis=1)
    # SVD برای گرفتن مؤلفه‌های زمانی
    try:
        U, S, Vt = np.linalg.svd(Xn, full_matrices=False)
        # Vt[0] مؤلفه زمانی اول (به ترتیب انرژی)
        pc = Vt[0]
        # سیگنال مؤلفه اول (زمانی)
        return pc
    except np.linalg.LinAlgError:
        # اگر SVD شکست خورد، از کانال سبز استفاده کن
        return Xn[1]

def method_chrom(r, g, b):
    """
    پیاده‌سازی ساده‌شدهِ کرومینانس (فرمول متداول):
    X = 3R - 2G
    Y = 1.5R + G - 1.5B
    S = X - (std(X)/std(Y)) * Y
    (این روش در منابع متعددی برای rPPG دیده می‌شود)
    """
    R = np.array(r)
    G = np.array(g)
    B = np.array(b)
    X = 3.0 * R - 2.0 * G
    Y = 1.5 * R + 1.0 * G - 1.5 * B
    stdY = np.std(Y) if np.std(Y) != 0 else 1.0
    stdX = np.std(X) if np.std(X) != 0 else 1.0
    S = X - (stdX / stdY) * Y
    return S

# ---------- تابع پردازش بافر ----------
def process_buffer(fps):
    """پردازش بافر و برگرداندن HR (در صورت امکان)"""
    if len(time_buffer) < 4:
        return None

    r = np.array(r_buffer)
    g = np.array(g_buffer)
    b = np.array(b_buffer)

    # زمان واقعی براساس timestampها (برای نمونه‌برداری دقیق)
    times = np.array(time_buffer)
    # تقریب fps واقعی
    if len(times) >= 2:
        dt = np.median(np.diff(times))
        if dt <= 0:
            real_fps = fps
        else:
            real_fps = 1.0 / dt
    else:
        real_fps = fps

    # انتخاب متد
    if current_method == 'GREEN':
        sig = method_green(r, g, b)
        sig = detrend_signal(sig)
        sig = bandpass_filter(sig, real_fps)
    elif current_method == 'PCA':
        sig = method_pca(r, g, b)
        sig = detrend_signal(sig)
        sig = bandpass_filter(sig, real_fps)
    elif current_method == 'CHROM':
        sig = method_chrom(r, g, b)
        sig = detrend_signal(sig)
        sig = bandpass_filter(sig, real_fps)
    else:
        # fallback
        sig = method_green(r, g, b)
        sig = detrend_signal(sig)
        sig = bandpass_filter(sig, real_fps)

    hr = estimate_hr_from_signal(sig, real_fps)
    return hr

# ---------- حلقه اصلی ----------

def main():
    global current_method, last_process_time

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # تنظیم FPS تلاش می‌کند ولی موفقیت تضمینی نیست
    cap.set(cv2.CAP_PROP_FPS, FPS)

    if not cap.isOpened():
        print("خطا: دوربین باز نشد.")
        return

    last_process_time = time.time()
    face_roi = None  # نگهداری آخرین ROI پیدا شده

    print("شروع rPPG (LOCAL). روش فعلی:", current_method)
    print("کلیدها: 1=GREEN, 2=PCA, 3=CHROM, Q=Quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("فریم دریافت نشد — پایان.")
            break

        t = time.time()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(80, 80))

        # اگر صورت یافت شد، آخرین ROI را به‌روزرسانی کن
        if len(faces) > 0:
            # انتخاب بزرگترین صورت (معمولاً نزدیک‌ترین)
            x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
            face_roi = (x, y, w, h)
        # اگر صورت نبود، face_roi قبلی را نگه می‌داریم (در صورت وجود)
        if face_roi is not None:
            x, y, w, h = face_roi
            # کاهش کمی نواحی حاشیه‌ای تا ROI بیشتر شامل پوست شود
            pad_w = int(0.1 * w)
            pad_h = int(0.06 * h)
            x1 = max(x + pad_w, 0)
            y1 = max(y + pad_h, 0)
            x2 = min(x + w - pad_w, frame.shape[1])
            y2 = min(y + h - pad_h, frame.shape[0])
            roi = frame[y1:y2, x1:x2]
        else:
            # اگر دیگر ROI نداریم، از تمام فریم استفاده کن (بدترین حالت)
            roi = frame

        # اگر ROI خالی شد نپرداز
        if roi.size == 0:
            continue

        # میانگین کانال‌ها داخل ROI
        roi_small = cv2.resize(roi, (0, 0), fx=0.5, fy=0.5)  # کاهش محاسبات
        b_mean = float(np.mean(roi_small[:, :, 0]))
        g_mean = float(np.mean(roi_small[:, :, 1]))
        r_mean = float(np.mean(roi_small[:, :, 2]))

        # ذخیره در بافر
        b_buffer.append(b_mean)
        g_buffer.append(g_mean)
        r_buffer.append(r_mean)
        time_buffer.append(t)

        # پردازش دوره‌ای
        if (t - last_process_time) >= PROCESS_INTERVAL and len(time_buffer) >= int(FPS * 2):
            hr = None
            try:
                hr = process_buffer(FPS)
            except Exception as e:
                print("خطا در پردازش بافر:", e)
                hr = None

            if hr is not None:
                hr_history.append(hr)
                avg_hr = np.median(list(hr_history))
                print(f"HR: {hr:.1f} BPM  —  median_history: {avg_hr:.1f} BPM")
            else:
                # اگر مقدار جدید بهتر نبود؛ می‌توانیم مقدار قبلی را نگه داریم
                avg_hr = np.median(list(hr_history)) if hr_history else None
                print("HR: (not detected)")

            last_process_time = t

        # نمایش روی فریم
        display = frame.copy()
        if face_roi is not None:
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        y_pos = 30
        status_text = "ACTIVE" if len(hr_history) > 0 else "INITIALIZING"
        cv2.putText(display, f"Status: {status_text}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        y_pos += 30

        if hr_history:
            avg_hr = int(round(np.median(list(hr_history))))
            cv2.putText(display, f"Heart Rate: {avg_hr} BPM", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            y_pos += 40
        else:
            cv2.putText(display, "Adjust lighting and hold still...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            y_pos += 30

        cv2.putText(display, f"Method: {current_method}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(display, "Keys: 1=GREEN, 2=PCA, 3=CHROM, Q=Quit", (10, display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("rPPG (local)", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('1'):
            current_method = 'GREEN'
            print("Method -> GREEN")
        elif key == ord('2'):
            current_method = 'PCA'
            print("Method -> PCA")
        elif key == ord('3'):
            current_method = 'CHROM'
            print("Method -> CHROM")

    cap.release()
    cv2.destroyAllWindows()
    if hr_history:
        print(f"\nمیانگین ضربان قلب در جلسه: {np.mean(list(hr_history)):.1f} BPM")
    print("پایان.")

if __name__ == "__main__":
    main()
