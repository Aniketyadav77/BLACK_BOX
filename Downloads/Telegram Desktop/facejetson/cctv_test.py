import cv2

url = "rtsp://admin:mits%251957@192.168.1.245:554/cam/realmonitor?channel=30&subtype=0"

cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("❌ Cannot open stream")
    exit()

print("✅ Connected!")

while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ Frame failed")
        break

    cv2.imshow("CCTV Feed", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()