# -*- coding: utf-8 -*-
import os
import threading
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict
from scipy.ndimage import gaussian_filter
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import customtkinter as ctk
import tkinter.filedialog as fd
import tkinter.messagebox as mb

# ----------------------------
# GLOBAL DEĞİŞKENLER
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
video_path = ""
model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=50)
person_frames = defaultdict(int)
output_image_path = os.path.join(BASE_DIR, "heatmap_output.png")
report_txt_path = os.path.join(BASE_DIR, "analiz_raporu.txt")
analysis_done = False
fps_global = 30  # varsayılan değer

# ----------------------------
# TXT RAPOR OLUŞTURMA

def generate_txt_report(fps, person_frames, txt_path):
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Is Etudu Analizi Raporu\n\n")
        f.write("Calisan Sureleri:\n")
        for pid, frame_total in sorted(person_frames.items()):
            saniye = int((frame_total / fps) * 3)
            f.write(f"Calisan {pid}: {saniye} saniye\n")


# ----------------------------
# ANALİZ FONKSİYONU

def process_video():
    global video_path, analysis_done, fps_global, heatmap_data

    if not video_path:
        mb.showerror("Hata", "Lütfen önce bir video seçin.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        mb.showerror("Hata", "Video açılamadı.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps_global = fps
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    heatmap_data = np.zeros((frame_height, frame_width))
    person_frames.clear()

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 3 != 0:
            continue

        results = model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes.data:
            x1, y1, x2, y2, conf, cls = box
            if int(cls) == 0 and float(conf) > 0.4:
                detections.append(([int(x1), int(y1), int(x2 - x1), int(y2 - y1)], float(conf), 'person'))

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l, t, w, h = track.to_ltrb()

            person_frames[track_id] += 1

            label = f"Calisan {track_id}"
            cv2.rectangle(frame, (int(l), int(t)), (int(l + w), int(t + h)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(l), int(t - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cx = int(l + w / 2)
            cy = int(t + h / 2)
            if 0 <= cy < frame_height and 0 <= cx < frame_width:
                heatmap_data[cy, cx] += 1

        cv2.imshow("Analiz", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    heatmap_blurred = gaussian_filter(heatmap_data, sigma=30)
    heatmap_normalized = heatmap_blurred / np.max(heatmap_blurred)

    plt.figure(figsize=(10, 6))
    plt.imshow(heatmap_normalized, cmap='jet', interpolation='bilinear')
    plt.axis('off')
    plt.title("Isi Haritasi")
    plt.show()  # KAYDETME YOK, SADECE GÖSTERİLİYOR

    analysis_done = True
    btn_download.configure(state="normal")
    mb.showinfo("Tamamlandı", "Analiz tamamlandı! Isı haritası gösterildi. Raporu kaydetmek için butona basın.")


# ----------------------------
# THREAD BAŞLAT

def start_analysis():
    threading.Thread(target=process_video, daemon=True).start()

# ----------------------------
# VİDEO SEÇME

def select_video():
    global video_path
    file = fd.askopenfilename(filetypes=[("MP4 Dosyaları", "*.mp4")])
    if file:
        video_path = os.path.normpath(file)
        label_video.configure(text=f"Seçilen Video: {os.path.basename(file)}")

# ----------------------------
# RAPOR İNDİRME

def download_report():
    if not analysis_done:
        mb.showwarning("Uyari", "Analiz yapilmadi.")
        return

    dest_txt = fd.asksaveasfilename(defaultextension=".txt", filetypes=[("Metin Dosyasi", "*.txt")])
    if dest_txt:
        generate_txt_report(fps_global, person_frames, dest_txt)
        mb.showinfo("Basarili", "Rapor metin olarak kaydedildi.")

        dest_img = fd.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Resmi", "*.png")])
        if dest_img:
            heatmap_blurred = gaussian_filter(heatmap_data, sigma=30)
            heatmap_normalized = heatmap_blurred / np.max(heatmap_blurred)
            plt.imsave(dest_img, heatmap_normalized, cmap='jet')
            mb.showinfo("Basarili", "Isı haritası PNG olarak kaydedildi.")


# ----------------------------
# UI TASARIMI

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("Görsel İşleme ile İş Etüdü Analizi")
app.geometry("1000x550")
app.minsize(800, 500)

try:
    from PIL import Image
    icon_folder_img = ctk.CTkImage(Image.open(os.path.join(BASE_DIR, "folder_icon.png")), size=(32, 32))
    icon_download_img = ctk.CTkImage(Image.open(os.path.join(BASE_DIR, "download_icon.png")), size=(32, 32))
except:
    icon_folder_img = None
    icon_download_img = None

frame = ctk.CTkFrame(app, fg_color="transparent")
frame.pack(expand=True, fill="both")

try:
    logo_img = ctk.CTkImage(Image.open(os.path.join(BASE_DIR, "header_logo_resized.png")), size=(300, 150))
    logo_label = ctk.CTkLabel(frame, image=logo_img, text="")
    logo_label.pack(pady=(20, 5))
except:
    pass

label_title = ctk.CTkLabel(frame, text="Görsel İşleme ile İş Etüdü Analizi", font=("Arial", 28, "bold"))
label_title.pack(pady=(0, 20))

btn_select = ctk.CTkButton(frame, text=" Video Seçin", image=icon_folder_img, compound="left", command=select_video, width=350, height=60)
btn_select.pack(pady=10)

label_video = ctk.CTkLabel(frame, text="Seçilen Video: ---", font=("Arial", 14))
label_video.pack(pady=5)

btn_visual = ctk.CTkButton(frame, text="🎥 Analizi Başlat", command=start_analysis, width=350, height=60, fg_color="#2196F3")
btn_visual.pack(pady=10)

btn_download = ctk.CTkButton(frame, text=" Raporu İndir (TXT)", image=icon_download_img, compound="left", command=download_report, width=350, height=60, state="disabled")
btn_download.pack(pady=10)

btn_exit = ctk.CTkButton(frame, text="Çıkış", command=app.quit, width=200, height=50, fg_color="white", text_color="black", border_width=1, border_color="black")
btn_exit.pack(pady=20)

app.mainloop()