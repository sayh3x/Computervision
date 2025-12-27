"""
Advanced camera test - version 2
With longer warm-up time and additional settings
"""
import cv2
import time

print("=" * 60)
print("🔧 ADVANCED CAMERA TEST")
print("=" * 60)

# ═══════════════════════════════════════════
# Stage 1: Close any previous connections
# ═══════════════════════════════════════════
print("\n[1/5] Releasing any previous connections...")
for i in range(3):
    temp = cv2.VideoCapture(i)
    temp.release()
time.sleep(1)

# ═══════════════════════════════════════════
# Stage 2: Open with specific settings
# ═══════════════════════════════════════════
print("[2/5] Opening camera with DirectShow...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Cannot open camera!")
    exit()

# ═══════════════════════════════════════════
# Stage 3: Set camera properties
# ═══════════════════════════════════════════
print("[3/5] Setting camera properties...")

# Disable auto settings for faster start
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 1 = manual mode on some cameras

# Set low resolution for speed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# Set exposure and brightness manually
cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)
cap.set(cv2.CAP_PROP_CONTRAST, 128)
cap.set(cv2.CAP_PROP_EXPOSURE, -4)  # Negative value = brighter

# Show actual settings
print(f"   Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
print(f"   FPS: {int(cap.get(cv2.CAP_PROP_FPS))}")
print(f"   Brightness: {cap.get(cv2.CAP_PROP_BRIGHTNESS)}")
print(f"   Exposure: {cap.get(cv2.CAP_PROP_EXPOSURE)}")

# ═══════════════════════════════════════════
# Stage 4: Longer warm-up
# ═══════════════════════════════════════════
print("[4/5] Warming up camera (this may take 5-10 seconds)...")

max_warmup = 100  # Maximum 100 frames
for i in range(max_warmup):
    ret, frame = cap.read()
    
    if ret and frame is not None:
        mean_val = frame.mean()
        
        # Show progress
        if i % 10 == 0:
            print(f"   Frame {i}: mean = {mean_val:.1f}")

        # If image became bright
        if mean_val > 10:
            print(f"   ✅ Camera ready at frame {i}! (mean = {mean_val:.1f})")
            break
    
    time.sleep(0.1)  # Wait between frames
else:
    print("   ⚠️ Camera still dark after warmup")

# ═══════════════════════════════════════════
# Stage 5: Live display
# ═══════════════════════════════════════════
print("[5/5] Showing live feed...")
print("\n" + "=" * 60)
print("👀 LIVE CAMERA FEED")
print("   Press 'Q' to quit")
print("   Press 'B' to increase brightness")
print("   Press 'D' to decrease brightness")
print("   Press 'R' to reset camera")
print("=" * 60)

cv2.namedWindow("Camera Test", cv2.WINDOW_NORMAL)

brightness = 128
frame_count = 0

while True:
    ret, frame = cap.read()
    frame_count += 1
    
    if not ret or frame is None:
        # Retry
        print(f"⚠️ Frame {frame_count}: Failed to read")
        time.sleep(0.1)
        continue
    
    mean_val = frame.mean()
    
    # Add information on image
    info_text = f"Frame: {frame_count} | Mean: {mean_val:.1f} | Brightness: {brightness}"

    # Background for text
    cv2.rectangle(frame, (0, 0), (500, 40), (0, 0, 0), -1)
    cv2.putText(frame, info_text, (10, 28), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # If still black, show message
    if mean_val < 5:
        cv2.rectangle(frame, (150, 200), (490, 280), (0, 0, 100), -1)
        cv2.putText(frame, "CAMERA IS BLACK!", (160, 230), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, "Check privacy shutter", (160, 260), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    cv2.imshow("Camera Test", frame)
    
    key = cv2.waitKey(30) & 0xFF
    
    if key == ord('q') or key == ord('Q'):
        break
    
    elif key == ord('b') or key == ord('B'):
        brightness = min(255, brightness + 20)
        cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
        print(f"🔆 Brightness: {brightness}")
    
    elif key == ord('d') or key == ord('D'):
        brightness = max(0, brightness - 20)
        cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
        print(f"🔅 Brightness: {brightness}")
    
    elif key == ord('r') or key == ord('R'):
        print("🔄 Resetting camera...")
        cap.release()
        time.sleep(1)
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        brightness = 128
        cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)

cap.release()
cv2.destroyAllWindows()

print("\n✅ Test completed!")
