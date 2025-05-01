# final_version.py
import os
from importlib import resources
import cv2
import numpy as np
import tkinter as tk
from tkinter import Label, messagebox
from PIL import Image, ImageTk
import pandas as pd
import math
import re
import webbrowser
from sklearn.cluster import KMeans

#dataset
with resources.path("foundation_matcher.data", "Final_Foundation_dataset.csv") as p:
    DATASET_PATH = str(p)

# Utility Functions for Image Processing

def adjust_brightness(image, factor=1.35):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 2] = np.clip(hsv[..., 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def extract_dominant_color(image, k=3):

    # reshape to (N,3)
    data = image.reshape(-1, 3).astype(np.float32)
    # cluster
    kmeans = KMeans(n_clusters=k, random_state=42).fit(data)
    labels = kmeans.labels_
    # find dominant cluster
    counts = np.bincount(labels)
    dominant = np.argmax(counts)
    # get all pixels in that cluster
    cluster_pixels = data[labels == dominant]
    # median per channel
    med = np.median(cluster_pixels, axis=0)
    return np.uint8(med)


def bgr_to_hex(color_bgr):
    b, g, r = color_bgr
    return f'#{r:02x}{g:02x}{b:02x}'

#Makeup Shade Matching

def hex_to_rgb(h):
    h = h.strip()
    if not h.startswith("#"):
        h = "#" + h
    if len(h) != 7:
        raise ValueError(f"Invalid hex: {h}")
    return tuple(int(h[i:i+2], 16) for i in (1, 3, 5))


def color_distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def find_closest_shades(target_hex, dataset, n=5):
    tgt = hex_to_rgb(target_hex)
    out = []
    for item in dataset:
        try:
            d = color_distance(tgt, hex_to_rgb(item.get("hex_code", "#000000")))
            out.append({**item, "distance": d})
        except:
            continue
    return sorted(out, key=lambda x: x["distance"])[:n]

#MediaPipe Face Detector Initialization
_face_detector = None

def get_face_detector():
    global _face_detector
    if _face_detector is None:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import logging
        logging.getLogger('absl').setLevel(logging.ERROR)
        import mediapipe as mp
        _face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )
    return _face_detector

# Load & Clean Dataset

if not os.path.isfile(DATASET_PATH):
    messagebox.showerror("Error", f"Dataset not found:\n{DATASET_PATH}")
    raise SystemExit("Missing dataset.")

df = pd.read_csv(DATASET_PATH, dtype={'hex_code': str})
df["hex_code"] = df["hex_code"].astype(str).str.strip().replace(
    {r'^(?!#)([0-9A-Fa-f]{6})$': r'#\1'}, regex=True
)
df = df[df["hex_code"].str.match(r'^#[0-9A-Fa-f]{6}$', na=False)]
if 'url' in df.columns:
    df["url"] = df["url"].astype(str).str.strip().replace({"": "N/A", "nan": "N/A"})
makeup_dataset = df.to_dict(orient='records')

#GUI & Camera

PRIMARY, SECONDARY, ACCENT = "#FF69B4", "#FFB6C1", "#FF1493"
BG, TEXT = "#FFF0F5", "#333333"

#Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    messagebox.showerror("Error", "Cannot open camera.")
    raise SystemExit("Camera unavailable.")

#Main window
root = tk.Tk()
root.title("Foundation Perfect Match")
root.geometry("800x700")
root.configure(bg=BG)

#Layout
tk.Label(root, text="Perfect Foundation Match", font=("Arial", 24, "bold"), fg=ACCENT, bg=BG).pack(pady=(15,5))
tk.Label(root, text="Find your ideal foundation shade with just one scan", font=("Arial", 14), fg=TEXT, bg=BG).pack(pady=(0,15))

video_frame = tk.Frame(root, bg=PRIMARY, bd=5)
video_frame.pack(padx=20, pady=10)
video_panel = Label(video_frame)
video_panel.pack()

#Label for the instructions
instr = tk.Label(root, text="Center your face and press the button", font=("Arial",12), fg=TEXT, bg=BG)
instr.pack(pady=10)

#Update and Capture

def update_frame():
    global update_id
    ret, frame = cap.read()
    if ret:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(img))
        video_panel.imgtk = imgtk
        video_panel.config(image=imgtk)
    update_id = video_panel.after(10, update_frame)


def capture_image():
    global update_id
    if update_id:
        video_panel.after_cancel(update_id)
    ret, frame = cap.read()
    if not ret:
        messagebox.showerror("Error", "Failed to capture image.")
        return

    bright = adjust_brightness(frame)
    rgb = cv2.cvtColor(bright, cv2.COLOR_BGR2RGB)
    detector = get_face_detector()
    results = detector.process(rgb)

    if results.detections:
        bbox = results.detections[0].location_data.relative_bounding_box
        h, w = bright.shape[:2]
        x1, y1 = int(bbox.xmin * w), int(bbox.ymin * h)
        x2 = x1 + int(bbox.width * w)
        y2 = y1 + int(bbox.height * h)
        crop = bright[y1:y2, x1:x2]
    else:
        crop = bright

    skin_color_bgr = extract_dominant_color(crop)
    skin_hex = bgr_to_hex(skin_color_bgr)

    cap.release()
    root.destroy()
    show_preview(crop, skin_hex)

#Capture Button & preview
capture_btn = tk.Button(root, text="Find My Perfect Match", font=("Arial",14,"bold"), bg=PRIMARY, fg="white", activebackground=ACCENT, command=capture_image)
capture_btn.pack(pady=20)


def show_preview(crop_img, skin_hex):
    p = tk.Tk()
    p.title("Your Skin Tone Analysis")
    p.configure(bg=BG)
    tk.Label(p, text="Processing Your Skin Tone", font=("Arial",18,"bold"), fg=ACCENT, bg=BG).pack(pady=15)
    frame = tk.Frame(p, bg=PRIMARY, bd=5)
    frame.pack(padx=20,pady=10)
    rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    im = ImageTk.PhotoImage(image=Image.fromarray(rgb))
    lbl = Label(frame, image=im)
    lbl.image = im
    lbl.pack()
    tk.Label(p, text=f"Detected Tone: {skin_hex}", font=("Arial",12,"bold"), fg=TEXT, bg=BG).pack(pady=10)
    p.after(3000, lambda: (p.destroy(), open_results(skin_hex)))
    p.mainloop()

#Final results layout for user

def open_results(skin_hex):
    matches = find_closest_shades(skin_hex, makeup_dataset)
    w = tk.Tk()
    w.geometry("800x800")
    tk.Label(w, text="Your Perfect Foundation Matches", font=("Arial",22,"bold"), fg=ACCENT, bg=BG).pack(pady=15)
    frame = tk.Frame(w, bg=BG)
    frame.pack(pady=10)
    tk.Label(frame, text="Your Skin Tone:", font=("Arial",12,"bold"), fg=TEXT, bg=BG).pack(side=tk.LEFT)
    tk.Frame(frame, width=30, height=30, bg=skin_hex, bd=1, relief="solid").pack(side=tk.LEFT,padx=5)
    tk.Label(frame, text=skin_hex, font=("Arial",12), fg=TEXT, bg=BG).pack(side=tk.LEFT)
    mf = tk.Frame(w, bg=BG)
    mf.pack(fill="both",expand=True,padx=20,pady=10)
    c = tk.Canvas(mf, bg=BG)
    sb = tk.Scrollbar(mf,orient="vertical",command=c.yview)
    sf = tk.Frame(c, bg=BG)
    sf.bind("<Configure>", lambda e: c.configure(scrollregion=c.bbox("all")))
    c.create_window((0,0),window=sf,anchor="nw")
    c.configure(yscrollcommand=sb.set)
    c.pack(side="left",fill="both",expand=True)
    sb.pack(side="right",fill="y")
    for i, shade in enumerate(matches):
        bgc = SECONDARY if i%2==0 else BG
        f=tk.Frame(sf,bg=bgc,bd=1,relief="solid");f.pack(fill="x",pady=5,ipady=10)
        tk.Frame(f,width=50,height=50,bg=shade['hex_code'],bd=1,relief="solid").pack(side="left",padx=10)
        info=tk.Frame(f,bg=bgc,padx=10);info.pack(side="left",fill="x",expand=True)
        tk.Label(info,text=f"{shade.get('brand','N/A')} – {shade.get('product','N/A')}",font=("Arial",14,"bold"),fg=ACCENT,bg=bgc).pack(anchor="w")
        tk.Label(info,text=f"Shade: {shade.get('shade_name','N/A')}",font=("Arial",12),fg=TEXT,bg=bgc).pack(anchor="w")
        tk.Label(info,text=f"Available at: {shade.get('retailer','N/A')}",font=("Arial",12),fg=TEXT,bg=bgc).pack(anchor="w")
        url=shade.get('url','N/A')
        if url!='N/A': l=tk.Label(info,text="Shop Online",font=("Arial",12,"underline"),fg="blue",bg=bgc,cursor="hand2");l.pack(anchor="w",pady=5);l.bind("<Button-1>",lambda e,u=url:webbrowser.open(u))
        pct=max(0,min(100,100-(shade['distance']*5)))
        tk.Label(info,text=f"Match Quality: {pct:.1f}%",font=("Arial",12),fg=TEXT,bg=bgc).pack(anchor="w",pady=5)
    tk.Label(w,text="Results based on image analysis; may vary with lighting.",font=("Arial",10),fg=TEXT,bg=BG).pack(pady=10)
    w.mainloop()


root.bind('<Return>',lambda e:capture_image())
update_frame()
root.mainloop()
