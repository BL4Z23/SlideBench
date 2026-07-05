import cv2
import json
import subprocess

WINDOW = "Calibrator"
IMAGE = "screenshot.png"
OUTPUT = "calibration.json"

MAX_WINDOW_W = 1600
MAX_WINDOW_H = 900

zoom = 1.0
MIN_ZOOM = 0.2
MAX_ZOOM = 10.0

cam_x = 0
cam_y = 0

mouse_x = 0
mouse_y = 0

dragging = False
last_drag = (0, 0)

points = []


def adb_screenshot():
    print("Taking screenshot...")
    with open(IMAGE, "wb") as f:
        subprocess.run(
            ["adb", "exec-out", "screencap", "-p"],
            stdout=f,
            check=True
        )

    img = cv2.imread(IMAGE)
    if img is None:
        raise RuntimeError("Couldn't load screenshot.")
    return img


img = adb_screenshot()

h, w = img.shape[:2]

zoom = min(MAX_WINDOW_W / w, MAX_WINDOW_H / h)
zoom = min(zoom, 1.0)

cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW, int(w * zoom), int(h * zoom))


def screen_to_image(x, y):
    return (
        int((x + cam_x) / zoom),
        int((y + cam_y) / zoom),
    )


def mouse(event, x, y, flags, param):
    global mouse_x, mouse_y
    global zoom, cam_x, cam_y
    global dragging, last_drag

    mouse_x = x
    mouse_y = y

    if event == cv2.EVENT_MOUSEWHEEL:

        old_zoom = zoom

        if flags > 0:
            zoom *= 1.15
        else:
            zoom /= 1.15

        zoom = max(MIN_ZOOM, min(MAX_ZOOM, zoom))

        ix = (x + cam_x) / old_zoom
        iy = (y + cam_y) / old_zoom

        cam_x = int(ix * zoom - x)
        cam_y = int(iy * zoom - y)

    elif event == cv2.EVENT_MBUTTONDOWN:
        dragging = True
        last_drag = (x, y)

    elif event == cv2.EVENT_MBUTTONUP:
        dragging = False

    elif event == cv2.EVENT_MOUSEMOVE and dragging:

        dx = x - last_drag[0]
        dy = y - last_drag[1]

        cam_x -= dx
        cam_y -= dy

        last_drag = (x, y)

    elif event == cv2.EVENT_LBUTTONDOWN:

        ix, iy = screen_to_image(x, y)

        ix = max(0, min(ix, img.shape[1]-1))
        iy = max(0, min(iy, img.shape[0]-1))

        if len(points) < 2:
            points.append((ix, iy))
            print(points[-1])

        if len(points) == 2:

            with open(OUTPUT, "w") as f:
                json.dump({
                    "top_left": {
                        "x": points[0][0],
                        "y": points[0][1]
                    },
                    "bottom_right": {
                        "x": points[1][0],
                        "y": points[1][1]
                    }
                }, f, indent=4)

            print("Saved.")


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

    if win_w <= 0:
        win_w = MAX_WINDOW_W

    if win_h <= 0:
        win_h = MAX_WINDOW_H

    cam_x = max(0, min(cam_x, max(0, W - win_w)))
    cam_y = max(0, min(cam_y, max(0, H - win_h)))

    view = scaled[
        cam_y:cam_y + win_h,
        cam_x:cam_x + win_w
    ].copy()

    mx = mouse_x
    my = mouse_y

    cv2.line(view, (mx, 0), (mx, view.shape[0]), (0,255,0), 1)
    cv2.line(view, (0,my), (view.shape[1], my), (0,255,0), 1)

    if len(points):

        px = int(points[0][0]*zoom) - cam_x
        py = int(points[0][1]*zoom) - cam_y

        cv2.circle(view, (px, py), 5, (0,0,255), -1)
        cv2.rectangle(view, (px, py), (mx, my), (255,0,0), 2)

    if len(points) == 2:

        x1 = int(points[0][0]*zoom) - cam_x
        y1 = int(points[0][1]*zoom) - cam_y

        x2 = int(points[1][0]*zoom) - cam_x
        y2 = int(points[1][1]*zoom) - cam_y

        cv2.rectangle(view, (x1,y1), (x2,y2), (0,255,255), 2)

    cv2.putText(
        view,
        f"{zoom:.2f}x",
        (10,30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,255),
        2
    )

    cv2.imshow(WINDOW, view)

    key = cv2.waitKey(10) & 0xFF

    if key == 27:
        break

    elif key == ord("r"):
        points.clear()

    elif key == ord(" "):
        img = adb_screenshot()
        h, w = img.shape[:2]

cv2.destroyAllWindows()