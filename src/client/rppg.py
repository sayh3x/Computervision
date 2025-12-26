import cv2
import numpy as np
import time
from collections import deque
from scipy import signal

# ---------- Parameters ----------
FPS = 30  # Target frame rate (camera must be able to deliver this FPS)
BUFFER_SECONDS = 8  # Buffer length in seconds (for FFT and stability)
BUFFER_SIZE = int(FPS * BUFFER_SECONDS)
PROCESS_INTERVAL = 2.0  # Signal processing interval (seconds)
MIN_HR = 40
MAX_HR = 200

# Available methods: 'GREEN', 'PCA', 'CHROM'
current_method = 'PCA'

# ---------- Face detection tool (OpenCV Haar Cascade) ----------
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

# ---------- Helper functions ----------
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
    # Use filtfilt to avoid phase delay
    return signal.filtfilt(b, a, x)

def detrend_signal(x):
    return signal.detrend(x, type='linear')

def estimate_hr_from_signal(sig, fps):
    """Estimate heart rate from a 1D signal using FFT"""
    sig = np.asarray(sig)
    N = len(sig)
    if N < 4:
        return None

    # Remove DC component
    sig = sig - np.mean(sig)

    # Hanning window
    win = np.hanning(N)
    sig_win = sig * win

    # FFT
    fft = np.fft.rfft(sig_win)
    fft_mag = np.abs(fft)
    freqs = np.fft.rfftfreq(N, d=1.0 / fps)

    # Heart rate frequency range
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

# ---------- Pulse extraction methods ----------
def method_green(r, g, b):
    """Pulse signal based on the green channel"""
    return np.array(g)

def method_pca(r, g, b):
    """Extract pulse using PCA over RGB channels"""
    X = np.vstack([r, g, b])  # Shape: 3 x N

    # Normalize channels (relative variation)
    mean_channels = np.mean(X, axis=1, keepdims=True)
    mean_channels[mean_channels == 0] = 1.0
    Xn = X / mean_channels - 1.0

    # Remove temporal trend
    Xn = signal.detrend(Xn, axis=1)

    # SVD for principal component extraction
    try:
        _, _, Vt = np.linalg.svd(Xn, full_matrices=False)
        return Vt[0]  # First temporal principal component
    except np.linalg.LinAlgError:
        # Fallback to green channel
        return Xn[1]

def method_chrom(r, g, b):
    """
    Simplified CHROM method:
    X = 3R - 2G
    Y = 1.5R + G - 1.5B
    S = X - (std(X) / std(Y)) * Y
    """
    R = np.array(r)
    G = np.array(g)
    B = np.array(b)

    X = 3.0 * R - 2.0 * G
    Y = 1.5 * R + G - 1.5 * B

    stdX = np.std(X) if np.std(X) != 0 else 1.0
    stdY = np.std(Y) if np.std(Y) != 0 else 1.0

    return X - (stdX / stdY) * Y

# ---------- Buffer processing ----------
def process_buffer(fps):
    """Process signal buffer and return estimated heart rate"""
    if len(time_buffer) < 4:
        return None

    r = np.array(r_buffer)
    g = np.array(g_buffer)
    b = np.array(b_buffer)

    # Estimate real FPS from timestamps
    times = np.array(time_buffer)
    if len(times) >= 2:
        dt = np.median(np.diff(times))
        real_fps = 1.0 / dt if dt > 0 else fps
    else:
        real_fps = fps

    # Method selection
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

    return estimate_hr_from_signal(sig, real_fps)

# ---------- Main loop ----------
def main():
    global current_method, last_process_time

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        return

    last_process_time = time.time()
    face_roi = None

    print("Starting rPPG (LOCAL). Current method:", current_method)
    print("Keys: 1=GREEN, 2=PCA, 3=CHROM, Q=Quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame capture failed. Exiting.")
            break

        t = time.time()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(80, 80)
        )

        # Update ROI if a face is detected
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
            face_roi = (x, y, w, h)

        # Use previous ROI if no face detected
        if face_roi is not None:
            x, y, w, h = face_roi
            pad_w = int(0.1 * w)
            pad_h = int(0.06 * h)
            x1 = max(x + pad_w, 0)
            y1 = max(y + pad_h, 0)
            x2 = min(x + w - pad_w, frame.shape[1])
            y2 = min(y + h - pad_h, frame.shape[0])
            roi = frame[y1:y2, x1:x2]
        else:
            roi = frame

        if roi.size == 0:
            continue

        # Mean RGB values in ROI
        roi_small = cv2.resize(roi, (0, 0), fx=0.5, fy=0.5)
        b_buffer.append(float(np.mean(roi_small[:, :, 0])))
        g_buffer.append(float(np.mean(roi_small[:, :, 1])))
        r_buffer.append(float(np.mean(roi_small[:, :, 2])))
        time_buffer.append(t)

        # Periodic processing
        if (t - last_process_time) >= PROCESS_INTERVAL and len(time_buffer) >= int(FPS * 2):
            try:
                hr = process_buffer(FPS)
            except Exception as e:
                print("Buffer processing error:", e)
                hr = None

            if hr is not None:
                hr_history.append(hr)
                median_hr = np.median(list(hr_history))
                print(f"HR: {hr:.1f} BPM — Median history: {median_hr:.1f} BPM")
            else:
                print("HR: not detected")

            last_process_time = t

        # Display overlay
        display = frame.copy()
        if face_roi is not None:
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        y = 30
        status = "ACTIVE" if hr_history else "INITIALIZING"
        cv2.putText(display, f"Status: {status}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        y += 30

        if hr_history:
            bpm = int(round(np.median(list(hr_history))))
            cv2.putText(display, f"Heart Rate: {bpm} BPM", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        else:
            cv2.putText(display, "Adjust lighting and hold still...",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (200, 200, 200), 2)

        cv2.putText(display, f"Method: {current_method}",
                    (10, display.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.putText(display,
                    "Keys: 1=GREEN, 2=PCA, 3=CHROM, Q=Quit",
                    (10, display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("rPPG (local)", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            break
        elif key == ord('1'):
            current_method = 'GREEN'
            print("Method switched to GREEN")
        elif key == ord('2'):
            current_method = 'PCA'
            print("Method switched to PCA")
        elif key == ord('3'):
            current_method = 'CHROM'
            print("Method switched to CHROM")

    cap.release()
    cv2.destroyAllWindows()

    if hr_history:
        print(f"\nSession average heart rate: {np.mean(list(hr_history)):.1f} BPM")

    print("Finished.")

if __name__ == "__main__":
    main()
