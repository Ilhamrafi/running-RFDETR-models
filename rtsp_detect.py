import json
import cv2
import os
import time
import threading
from queue import Queue
from rfdetr import RFDETRNano
import supervision as sv
import logging
import numpy as np

# Setup logging - Konfigurasi sistem pencatatan log
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RTSPImageCapture:
    """
    Kelas untuk mengakses kamera RTSP, melakukan deteksi objek,
    dan menampilkan hasil deteksi secara real-time
    """
    def __init__(self, rtsp_url, output_dir="output"):
        # Inisialisasi parameter dasar
        self.rtsp_url = rtsp_url
        self.output_dir = output_dir
        self.cap = None
        self.image_count = 0
        self.model = None
        self.class_mapping = None
        
        # Batasi laju pembacaan frame untuk performa yang stabil
        self.target_fps = 10

        # Konfigurasi thread dan antrian untuk pemrosesan paralel
        self.frame_queue = Queue(maxsize=5)
        self.result_queue = Queue(maxsize=5)
        self.is_running = False
        
        # Variabel untuk mencatat statistik performa
        self.frame_count = 0
        self.detection_count = 0
        self.last_log_time = time.time()
        self.start_time = time.time()
        
        # Muat model AI dan pemetaan kelas
        self._load_model()
        self._load_class_mapping()

    def _load_model(self):
        """Memuat model deteksi objek RF-DETR"""
        try:
            model_path = "D:/project-computer-vision/exavator-load-detection/model/RF-DETR_25epo_Nano/checkpoint_best_ema.pth"
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"File model tidak ditemukan: {model_path}")
            
            logger.info("Memuat model RF-DETR...")
            self.model = RFDETRNano(pretrain_weights=model_path)
            self.model.optimize_for_inference()
            
            # Pemanasan model untuk mengurangi latensi awal
            logger.info("Melakukan warm-up model...")
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            for _ in range(3):
                _ = self.model.predict(dummy_frame)
            
            logger.info("Model berhasil dimuat dan dioptimasi")
        except Exception as e:
            logger.error(f"Error saat memuat model: {e}")
            raise

    def _load_class_mapping(self):
        """Memuat pemetaan ID kelas ke nama kelas"""
        try:
            classes_file = "D:/project-computer-vision/exavator-load-detection/src/classes.json"
            if not os.path.exists(classes_file):
                raise FileNotFoundError(f"File classes tidak ditemukan: {classes_file}")
            
            with open(classes_file, "r") as f:
                self.class_mapping = json.load(f)
            logger.info(f"Pemetaan kelas berhasil dimuat: {len(self.class_mapping)} kelas")
        except Exception as e:
            logger.error(f"Error saat memuat pemetaan kelas: {e}")
            raise

    def _preprocess_frame(self, frame):
        """Preprocessing minimal sebelum inference"""
        return frame

    def open_stream(self):
        """Membuka koneksi ke stream RTSP"""
        try:
            logger.info(f"Membuka RTSP stream: {self.rtsp_url}")
            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            # Konfigurasi parameter stream untuk performa optimal
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
            self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            
            if not self.cap.isOpened():
                raise ConnectionError("Tidak dapat membuka RTSP stream")

            ret, frame = self.cap.read()
            if not ret:
                raise ConnectionError("Tidak dapat membaca frame dari stream")

            # Buat folder output jika belum ada
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Dapatkan informasi stream
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            logger.info(f"Stream opened - Resolution: {actual_width}x{actual_height}, FPS: {actual_fps}")
        except Exception as e:
            logger.error(f"Error saat membuka RTSP stream: {e}")
            raise

    def _process_detections(self, frame):
        """Melakukan deteksi objek pada frame dan menambahkan anotasi"""
        try:
            processed_frame = self._preprocess_frame(frame)
            # Lakukan prediksi dengan model
            detections = self.model.predict(processed_frame)
            
            # Persiapkan label untuk anotasi
            labels = []
            if hasattr(detections, 'class_id') and detections.class_id is not None:
                for class_id in detections.class_id:
                    class_id_str = str(int(class_id))
                    class_name = self.class_mapping.get(class_id_str, f"class_{class_id}")
                    labels.append(class_name)
            
            # Anotasi frame dengan hasil deteksi
            annotated_frame = frame.copy()
            detection_count = len(detections) if detections is not None else 0
            
            if detection_count > 0:
                try:
                    # Tambahkan bounding box dan label menggunakan supervision
                    box_annotator = sv.BoxAnnotator(thickness=2)
                    label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.6)
                    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
                    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
                except Exception as e:
                    logger.warning(f"Annotation error: {e}")
                    return frame, detection_count
            
            return annotated_frame, detection_count
        except Exception as e:
            logger.error(f"Error saat memproses deteksi: {e}")
            return frame, 0

    def _log_performance_stats(self):
        """Mencatat statistik performa setiap 5 detik"""
        current_time = time.time()
        if current_time - self.last_log_time >= 5.0:
            elapsed = current_time - self.start_time
            avg_fps = self.frame_count / elapsed if elapsed > 0 else 0
            avg_detection_per_frame = self.detection_count / max(1, self.frame_count)
            logger.info(f"Performance Stats - Frames: {self.frame_count}, Avg FPS: {avg_fps:.2f}, "
                        f"Total Detections: {self.detection_count}, Avg Det/Frame: {avg_detection_per_frame:.2f}")
            self.last_log_time = current_time

    def capture_images(self):
        """Loop utama untuk membaca frame dan menjalankan deteksi objek"""
        last_frame_time = time.time()
        try:
            while True:
                # Kontrol FPS agar sesuai dengan target
                current_time = time.time()
                frame_interval = 1.0 / self.target_fps
                time_since_last = current_time - last_frame_time
                if time_since_last < frame_interval:
                    time.sleep(frame_interval - time_since_last)
                last_frame_time = time.time()
                
                # Baca frame dari kamera
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Frame capture failed, attempting reconnect...")
                    self.cap.release()
                    self.open_stream()
                    continue
                
                # Proses frame dan tampilkan hasil
                self.frame_count += 1
                annotated_frame, detection_count = self._process_detections(frame)
                self.detection_count += detection_count
                
                # Catat statistik dan tampilkan frame
                self._log_performance_stats()
                cv2.imshow('RTSP Object Detection', annotated_frame)
                
                # Cek input keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Exit requested by user")
                    break
                elif key == ord('s'):
                    self._save_frame(annotated_frame, self.frame_count)
                    
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error in capture loop: {e}")

    def _save_frame(self, frame, frame_number):
        """Menyimpan frame sebagai file gambar"""
        try:
            filename = f"frame_{frame_number:06d}_{int(time.time())}.jpg"
            filepath = os.path.join(self.output_dir, filename)
            cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            logger.info(f"Frame saved: {filepath}")
        except Exception as e:
            logger.error(f"Error saving frame: {e}")

    def close_stream(self):
        """Membersihkan resource saat aplikasi ditutup"""
        try:
            self.is_running = False
            if self.cap is not None:
                self.cap.release()
                logger.info("Video capture released")
            cv2.destroyAllWindows()
            logger.info("OpenCV windows closed")
        except Exception as e:
            logger.error(f"Error closing stream: {e}")

    def main(self):
        """Metode utama untuk menjalankan aplikasi"""
        try:
            logger.info("=== Starting RTSP Object Detection ===")
            logger.info(f"Target FPS: {self.target_fps}")
            logger.info("Controls: 'q' quit, 's' save frame")
            self.open_stream()
            self.capture_images()
        except Exception as e:
            logger.error(f"Error in main execution: {e}")
        finally:
            # Tutup koneksi dan tampilkan statistik akhir
            self.close_stream()
            total_time = time.time() - self.start_time
            avg_fps = self.frame_count / total_time if total_time > 0 else 0
            logger.info(f"=== Final Statistics ===")
            logger.info(f"Total runtime: {total_time:.2f} seconds")
            logger.info(f"Total frames: {self.frame_count}")
            logger.info(f"Average FPS: {avg_fps:.2f}")
            logger.info(f"Total detections: {self.detection_count}")
            logger.info("Application terminated")

def main():
    """Fungsi main untuk menginisialisasi dan menjalankan aplikasi"""
    rtsp_url = 'rtsp://admin:Bengawanai_2024@192.168.1.101:554/Streaming/Channels/101'
    output_dir = "output"
    try:
        image_capture = RTSPImageCapture(rtsp_url, output_dir)
        image_capture.main()
        return 0
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        return 1

if __name__ == "__main__":
    exit(main())