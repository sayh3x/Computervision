# rppg_pose_merged.py
"""
Merged code:
- Keypoint detection using YOLO pose (ultralytics)
- Face ROI extraction based on keypoints (nose/eyes/ears)
- Fallback to Haar cascade if keypoints are unavailable
- rPPG computation using GREEN / PCA / CHROM methods
- Display annotated frame and heart rate
"""

import cv2
import numpy as np
import time
from collections import deque
from scipy import signal
from ultralytics import YOLO

# ---------- Configuration ----------
CAM_ID = 0
MODEL_PATH = "yolo11s-pose.pt"  # Path to your pose model
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 30  # Target FPS; actual sampling is computed from timestamps
BUFFER_SECONDS = 8
BUFFER_SIZE = int(FPS * BUFFER_SECONDS)
PROCESS_INTERVAL = 2.0
MIN_HR = 40
MAX_HR = 200
KP_CONF_THRESH = 0.3  # Confidence threshold for keypoints

# Available methods: 'GREEN', 'PCA', 'CHROM'
current_method = 'CHROM'

# ---------- Load pose model ----------
model = YOLO(MODEL_PATH)

# ---------- Haar cascade fallback ----------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ---------- Buffers ----------
r_buffer = deque(maxlen=BUFFER_SIZE)
g_buffer = deque(maxlen=BUFFER_SIZE)
b_buffer = deque(maxlen=BUFFER_SIZE)
time_buffer = deque(maxlen=BUFFER_SIZE)
last_process_time = 0
hr_history = deque(maxlen=10)

# Body keypoint names according to your input (indices)
keypoint_names = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# ---------- Filter and FFT helper functions ----------
def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(x, fs, lowcut=0.7, highcut=4.0, order=3):
    x = np.asarray(x)
    if len(x) < 3:
        return x
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    try:
        return signal.filtfilt(b, a, x)
    except Exception:
        # If filtfilt fails, fall back to lfilter (no zero-phase filtering)
        return signal.lfilter(b, a, x)

def detrend_signal(x):
    return signal.detrend(x, type='linear')

def estimate_hr_from_signal(sig, fps):
    sig = np.asarray(sig)
    N = len(sig)
    if N < 4:
        return None
    sig = sig - np.mean(sig)
    win = np.hanning(N)
    sig_win = sig * win
    fft = np.fft.rfft(sig_win)
    fft_mag = np.abs(fft)
    freqs = np.fft.rfftfreq(N, d=1.0 / fps)
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

# ---------- Pulse extraction algorithms ----------
def method_green(r, g, b):
    return np.array(g)

def method_pca(r, g, b):
    X = np.vstack([r, g, b])
    mean_channels = np.mean(X, axis=1, keepdims=True)
    mean_channels[mean_channels == 0] = 1.0
    Xn = X / mean_channels - 1.0
    Xn = signal.detrend(Xn, axis=1)
    try:
        U, S, Vt = np.linalg.svd(Xn, full_matrices=False)
        pc = Vt[0]
        return pc
    except np.linalg.LinAlgError:
        return Xn[1]

def method_chrom(r, g, b):
    R = np.array(r)
    G = np.array(g)
    B = np.array(b)
    X = 3.0 * R - 2.0 * G
    Y = 1.5 * R + 1.0 * G - 1.5 * B
    stdY = np.std(Y) if np.std(Y) != 0 else 1.0
    stdX = np.std(X) if np.std(X) != 0 else 1.0
    S = X - (stdX / stdY) * Y
    return S

# ---------- Buffer processing ----------
def process_buffer(fps):
    if len(time_buffer) < 4:
        return None

    r = np.array(r_buffer)
    g = np.array(g_buffer)
    b = np.array(b_buffer)
    times = np.array(time_buffer)

    if len(times) >= 2:
        dt = np.median(np.diff(times))
        real_fps = fps if dt <= 0 else 1.0 / dt
    else:
        real_fps = fps

    if current_method == 'GREEN':
        sig = method_green(r, g, b)
    elif current_method == 'PCA':
        sig = method_pca(r, g, b)
    elif current_method == 'CHROM':
        sig = method_chrom(r, g, b)
    else:
        sig = method_green(r, g, b)

    sig = detrend_signal(sig)
    sig = bandpass_filter(sig, real_fps)
    hr = estimate_hr_from_signal(sig, real_fps)
    return hr

# ---------- ROI extraction from keypoints ----------
def roi_from_keypoints_array(keypoints_xy, keypoints_conf, frame_shape,
                             conf_thresh=KP_CONF_THRESH):
    """
    keypoints_xy: numpy (num_people, num_kp, 2)
    keypoints_conf: numpy (num_people, num_kp)

    Returns: (x1, y1, x2, y2, person_idx) or None

    Method:
    For each person, if multiple facial keypoints have high confidence,
    create a bounding box around them.
    """
    best = None
    H, W = frame_shape[:2]
    num_people = keypoints_xy.shape[0]

    for p in range(num_people):
        kp_xy = keypoints_xy[p]
        kp_conf = keypoints_conf[p]

        # Facial keypoint indices: nose, eyes, ears (0..4)
        face_idxs = [0, 1, 2, 3, 4]
        pts = []

        for i in face_idxs:
            if i < kp_xy.shape[0] and kp_conf[i] >= conf_thresh:
                x, y = kp_xy[i]
                pts.append((x, y))

        if len(pts) >= 2:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]

            x1 = max(int(min(xs) - 0.25 * (max(xs) - min(xs) + 10)), 0)
            y1 = max(int(min(ys) - 0.6 * (max(ys) - min(ys) + 10)), 0)
            x2 = min(int(max(xs) + 0.25 * (max(xs) - min(xs) + 10)), W)
            y2 = min(int(max(ys) + 0.25 * (max(ys) - min(ys) + 10)), H)

            area = (x2 - x1) * (y2 - y1)

            # Select the largest area (usually the closest person)
            if best is None or area > best[0]:
                best = (area, (x1, y1, x2, y2, p))

    if best is None:
        return None

    return best[1]

# ---------- Main loop ----------
def main():
    global current_method, last_process_time

    cap = cv2.VideoCapture(CAM_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    if not cap.isOpened():
        print("Error: camera could not be opened.")
        return

    last_process_time = time.time()
    face_roi = None

    print("Starting rPPG with Pose + Haar fallback. Current method:", current_method)
    print("Keys: 1=GREEN, 2=PCA, 3=CHROM, Q=Quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No frame received — exiting.")
            break

        t = time.time()
        annotated = frame.copy()

        # --- Pose model inference ---
        roi_from_pose = None
        try:
            results = model(frame, conf=0.5)
            r0 = results[0]

            # Draw default model annotations
            annotated = r0.plot() if hasattr(r0, 'plot') else annotated

            # Access keypoints if available
            if getattr(r0, 'keypoints', None) is not None:
                try:
                    kp_xy = r0.keypoints.xy.numpy()
                    kp_conf = r0.keypoints.conf.numpy()

                    kp_xy = np.asarray(kp_xy)
                    kp_conf = np.asarray(kp_conf)

                    roi_data = roi_from_keypoints_array(
                        kp_xy, kp_conf, frame.shape,
                        conf_thresh=KP_CONF_THRESH
                    )

                    if roi_data is not None:
                        x1, y1, x2, y2, person_idx = roi_data
                        roi_from_pose = (x1, y1, x2, y2)

                        cv2.rectangle(
                            annotated, (x1, y1), (x2, y2),
                            (0, 200, 0), 2
                        )

                        # Draw keypoints for selected person
                        num_kp = kp_xy.shape[1]
                        for k_idx in range(num_kp):
                            conf = kp_conf[person_idx, k_idx]
                            if conf >= KP_CONF_THRESH:
                                xk, yk = kp_xy[person_idx, k_idx].astype(int)
                                cv2.circle(
                                    annotated, (xk, yk),
                                    3, (0, 0, 255), -1
                                )
                                # Label only facial keypoints to avoid clutter
                                if k_idx <= 4:
                                    cv2.putText(
                                        annotated,
                                        keypoint_names[k_idx],
                                        (xk + 4, yk - 4),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.4, (0, 255, 0), 1
                                    )
                except Exception:
                    pass
        except Exception:
            roi_from_pose = None

        # --- Use pose ROI if available, otherwise Haar fallback ---
        if roi_from_pose is not None:
            x1, y1, x2, y2 = roi_from_pose
            face_roi = (x1, y1, x2 - x1, y2 - y1)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1,
                minNeighbors=4, minSize=(80, 80)
            )
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
                face_roi = (x, y, w, h)
            # Otherwise, keep the previous ROI for a short time

        # Extract ROI and compute channel means
        if face_roi is not None:
            x, y, w, h = face_roi

            # Crop margins to focus more on skin
            pad_w = int(0.1 * w)
            pad_h = int(0.06 * h)

            x1 = max(x + pad_w, 0)
            y1 = max(y + pad_h, 0)
            x2 = min(x + w - pad_w, frame.shape[1])
            y2 = min(y + h - pad_h, frame.shape[0])

            roi = frame[y1:y2, x1:x2]

            if roi.size != 0:
                roi_small = cv2.resize(roi, (0, 0), fx=0.5, fy=0.5)
                b_mean = float(np.mean(roi_small[:, :, 0]))
                g_mean = float(np.mean(roi_small[:, :, 1]))
                r_mean = float(np.mean(roi_small[:, :, 2]))

                b_buffer.append(b_mean)
                g_buffer.append(g_mean)
                r_buffer.append(r_mean)
                time_buffer.append(t)

                cv2.rectangle(
                    annotated, (x1, y1), (x2, y2),
                    (255, 255, 0), 2
                )
        else:
            # If no ROI is available, sample the entire frame (low quality)
            roi_small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            b_buffer.append(float(np.mean(roi_small[:, :, 0])))
            g_buffer.append(float(np.mean(roi_small[:, :, 1])))
            r_buffer.append(float(np.mean(roi_small[:, :, 2])))
            time_buffer.append(t)

        # Periodic HR processing
        if (t - last_process_time) >= PROCESS_INTERVAL and \
           len(time_buffer) >= int(FPS * 2):

            try:
                hr = process_buffer(FPS)
            except Exception as e:
                print("Buffer processing error:", e)
                hr = None

            if hr is not None:
                hr_history.append(hr)
                avg_hr = np.median(list(hr_history))
                print(f"HR: {hr:.1f} BPM — median_history: {avg_hr:.1f} BPM")
            else:
                print("HR: (not detected)")

            last_process_time = t

        # Display overlay
        display = annotated
        y_pos = 30

        status_text = "ACTIVE" if len(hr_history) > 0 else "INITIALIZING"
        cv2.putText(
            display, f"Status: {status_text}",
            (10, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 165, 255), 2
        )

        y_pos += 30
        if hr_history:
            avg_hr = int(round(np.median(list(hr_history))))
            cv2.putText(
                display, f"Heart Rate: {avg_hr} BPM",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 0), 2
            )
        else:
            cv2.putText(
                display, "Adjust lighting and hold still...",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (200, 200, 200), 2
            )

        y_pos += 40
        cv2.putText(
            display, f"Method: {current_method}",
            (10, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 255, 0), 2
        )

        cv2.putText(
            display,
            "Keys: 1=GREEN, 2=PCA, 3=CHROM, Q=Quit",
            (10, display.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (200, 200, 200), 1
        )

        cv2.imshow("rPPG Pose Merged", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
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
        print(f"\nSession average heart rate: {np.mean(list(hr_history)):.1f} BPM")

    print("Finished.")

if __name__ == "__main__":
    main()
