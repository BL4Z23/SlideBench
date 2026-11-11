import cv2
import numpy as np
import subprocess

WINDOW = "ADB Cropper"
OUTPUT = "crop.png"

zoom = 1.0
MIN_ZOOM = 0.2
MAX_ZOOM = 8.0

cam_x = 0
cam_y = 0

mouse_x = 0
mouse_y = 0

panning = False
drawing = False

pan_start = (0, 0)
start_img = (0, 0)
end_img = (0, 0)


def adb_screenshot():
    print("Taking screenshot...")

    result = subprocess.run(
        ["adb", "exec-out", "screencap", "-p"],
        capture_output=True,
        check=True
    )

    img = cv2.imdecode(
        np.frombuffer(result.stdout, np.uint8),
        cv2.IMREAD_COLOR
    )

    if img is None:
        raise RuntimeError("Couldn't decode screenshot.")

    return img


img = adb_screenshot()

h, w = img.shape[:2]

zoom = min(1600 / w, 900 / h)
zoom = min(zoom, 1.0)

cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW, int(w * zoom), int(h * zoom))


def screen_to_image(x, y):
    ix = int((x + cam_x) / zoom)
    iy = int((y + cam_y) / zoom)

    ix = max(0, min(ix, img.shape[1] - 1))
    iy = max(0, min(iy, img.shape[0] - 1))

    return ix, iy


def mouse(event, x, y, flags, param):
    global mouse_x, mouse_y
    global zoom, cam_x, cam_y
    global panning, drawing
    global pan_start
    global start_img, end_img

    mouse_x = x
    mouse_y = y

    if event == cv2.EVENT_MOUSEWHEEL:

        old = zoom

        if flags > 0:
            zoom *= 1.15
        else:
            zoom /= 1.15

        zoom = max(MIN_ZOOM, min(MAX_ZOOM, zoom))

        ix = (x + cam_x) / old
        iy = (y + cam_y) / old

        cam_x = int(ix * zoom - x)
        cam_y = int(iy * zoom - y)

    elif event == cv2.EVENT_MBUTTONDOWN:
        panning = True
        pan_start = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and panning:

        dx = x - pan_start[0]
        dy = y - pan_start[1]

        cam_x -= dx
        cam_y -= dy

        pan_start = (x, y)

    elif event == cv2.EVENT_MBUTTONUP:
        panning = False

    elif event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_img = screen_to_image(x, y)
        end_img = start_img

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        end_img = screen_to_image(x, y)

    elif event == cv2.EVENT_LBUTTONUP:

        drawing = False
        end_img = screen_to_image(x, y)

        x1 = min(start_img[0], end_img[0])
        y1 = min(start_img[1], end_img[1])
        x2 = max(start_img[0], end_img[0])
        y2 = max(start_img[1], end_img[1])

        crop = img[y1:y2, x1:x2]

        if crop.size:
            cv2.imwrite(OUTPUT, crop)
            print(f"Saved {OUTPUT}")


cv2.setMouseCallback(WINDOW, mouse)

while True:

    scaled = cv2.resize(
        img,
        None,
        fx=zoom,
        fy=zoom,
        interpolation=cv2.INTER_NEAREST
    )

    H, W = scaled.shape[:2]

    win_w, win_h = cv2.getWindowImageRect(WINDOW)[2:]

    cam_x = max(0, min(cam_x, max(0, W - win_w)))
    cam_y = max(0, min(cam_y, max(0, H - win_h)))

    view = scaled[
        cam_y:cam_y + win_h,
        cam_x:cam_x + win_w
    ].copy()

    cv2.line(view, (mouse_x, 0), (mouse_x, view.shape[0]), (0, 255, 0), 1)
    cv2.line(view, (0, mouse_y), (view.shape[1], mouse_y), (0, 255, 0), 1)

    if drawing:

        x1 = int(start_img[0] * zoom) - cam_x
        y1 = int(start_img[1] * zoom) - cam_y

        x2 = int(end_img[0] * zoom) - cam_x
        y2 = int(end_img[1] * zoom) - cam_y

        cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 255), 2)

    cv2.putText(
        view,
        f"{zoom:.2f}x",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.imshow(WINDOW, view)

    key = cv2.waitKey(10) & 0xFF

    if key == 27:
        break

    elif key == ord(" "):
        img = adb_screenshot()

    elif key == ord("r"):
        drawing = False

cv2.destroyAllWindows()