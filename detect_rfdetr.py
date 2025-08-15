import numpy as np
import supervision as sv
from rfdetr import RFDETRNano
from tqdm import tqdm
import json

# Tentukan jalur video input dan output
SOURCE_VIDEO_PATH = "D:/project-computer-vision/exavator-load-detection/data/raw/exa_day.mp4"
TARGET_VIDEO_PATH = "D:/project-computer-vision/exavator-load-detection/data/processed/exavator_processed.mp4"
TARGET_CSV_PATH = "D:/project-computer-vision/exavator-load-detection/data/processed/bucket_tracking.csv"

# Load class mapping dari file JSON
with open("D:/project-computer-vision/exavator-load-detection/src/classes.json", "r") as f:
    class_mapping = json.load(f)

# Inisialisasi model RFDETRSmall / RFDETRNano
model = RFDETRNano(pretrain_weights="D:/project-computer-vision/exavator-load-detection/model/RF-DETR_25epo_Nano/checkpoint_best_ema.pth")
model.optimize_for_inference()

# Buat generator untuk frame video
frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

# Dapatkan informasi video
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
tracker = sv.ByteTrack()

# BoxAnnotator untuk truck
box_annotator_truck = sv.BoxAnnotator()

# BoxAnnotator untuk bucket
box_annotator_bucket = sv.BoxCornerAnnotator()

# LabelAnnotator untuk truck (label di TOP_RIGHT, warna hitam)
label_annotator_truck = sv.LabelAnnotator(
    text_thickness=2,
    text_scale=1.0,
    text_position=sv.Position.TOP_RIGHT,
    smart_position=True,
    text_color=sv.Color.BLACK
)

# LabelAnnotator untuk bucket (label di CENTER, warna hitam)
label_annotator_bucket = sv.LabelAnnotator(
    text_thickness=2,
    text_scale=1.0,
    text_position=sv.Position.CENTER,
    smart_position=True,
    text_color=sv.Color.BLACK
)

with sv.CSVSink(TARGET_CSV_PATH) as csv_sink, sv.VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
    for frame_index, frame in enumerate(tqdm(frame_generator, desc="Memproses video")):
        detections = model.predict(frame, threshold=0.5)

        # Filter truck (class_id == 5)
        truck_mask = detections.class_id == 5
        truck_detections = detections[truck_mask]
        bucket_detections = detections[~truck_mask]

        # Tracking untuk bucket saja
        tracked_bucket = tracker.update_with_detections(bucket_detections)

        # Simpan hasil tracking bucket ke CSV
        csv_sink.append(tracked_bucket, {})

        # Label untuk truck
        truck_labels = [
            class_mapping.get(str(class_id))
            for class_id in truck_detections.class_id
        ]

        # Label untuk bucket
        bucket_labels = [
            class_mapping.get(str(class_id))
            for class_id in tracked_bucket.class_id
        ]

        annotated_frame = frame.copy()
        # Anotasi truck (label di TOP_RIGHT)
        annotated_frame = box_annotator_truck.annotate(annotated_frame, detections=truck_detections)
        annotated_frame = label_annotator_truck.annotate(annotated_frame, detections=truck_detections, labels=truck_labels)
        # Anotasi bucket (label di CENTER)
        annotated_frame = box_annotator_bucket.annotate(annotated_frame, detections=tracked_bucket)
        annotated_frame = label_annotator_bucket.annotate(annotated_frame, detections=tracked_bucket, labels=bucket_labels)

        sink.write_frame(annotated_frame)

print(f"Pemrosesan video selesai. Output disimpan ke {TARGET_VIDEO_PATH}")
print(f"Hasil tracking bucket disimpan ke {TARGET_CSV_PATH}")