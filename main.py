import cv2
import numpy as np
from ultralytics import YOLO
import time

class YOLOStreamProcessor:
    def __init__(self, stream_url):
        self.stream_url = stream_url
        self.running = False
        self.model = YOLO('yolov8n.pt')
        print("YOLO модель загружена")
        
    def process_stream_window(self):
        print(f"Подключение к потоку: {self.stream_url}")
        
        cap = cv2.VideoCapture(self.stream_url)
        if not cap.isOpened():
            print("Ошибка подключения к потоку")
            return
        
        print("Подключено. Нажмите 'q' для выхода")
        self.running = True
        frame_count = 0
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("Нет сигнала...")
                time.sleep(0.1)
                continue
            
            frame_count += 1
            
            # Детекция каждый второй кадр
            if frame_count % 2 == 0:
                processed_frame = self.detect_people(frame)
            else:
                processed_frame = frame.copy()
            
            cv2.putText(processed_frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('YOLO Detection', processed_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_name = f"detection_{int(time.time())}.jpg"
                cv2.imwrite(screenshot_name, processed_frame)
                print(f"Скриншот: {screenshot_name}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("Окно закрыто")
    
    def process_stream_http(self, http_port=8000):
        from flask import Flask, Response
        
        print(f"Запуск HTTP сервера на порту {http_port}")
        print(f"HTTP поток: http://localhost:{http_port}/stream")
        
        app = Flask(__name__)
        
        def generate_frames():
            cap = cv2.VideoCapture(self.stream_url)
            if not cap.isOpened():
                print("Ошибка подключения к потоку")
                return
            
            frame_count = 0
            print("Генерация кадров запущена")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                frame_count += 1
                processed_frame = self.detect_people(frame)
                
                cv2.putText(processed_frame, f"HTTP Stream | Frame: {frame_count}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                ret, buffer = cv2.imencode('.jpg', processed_frame, 
                                         [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        @app.route('/stream')
        def video_feed():
            return Response(generate_frames(),
                          mimetype='multipart/x-mixed-replace; boundary=frame')
        
        @app.route('/')
        def index():
            return f'''
            <html>
            <body style="margin:0; background:#000; color:white; font-family:Arial;">
                <h2 style="text-align:center; margin:20px;">YOLO Detection Stream</h2>
                <div style="text-align:center;">
                    <img src="/stream" style="max-width:100%; border:2px solid #0f0;">
                </div>
                <p style="text-align:center; margin:20px;">
                    Прямая ссылка: <a href="http://localhost:{http_port}/stream" style="color:#0f0;">
                    http://localhost:{http_port}/stream</a>
                </p>
            </body>
            </html>
            '''
        
        print("HTTP сервер запущен")
        print("Для просмотра:")
        print(f"   Браузер: http://localhost:{http_port}")
        print(f"   VLC: http://localhost:{http_port}/stream")
        print("   Нажмите Ctrl+C для остановки")
        
        try:
            app.run(host='0.0.0.0', port=http_port, debug=False, threaded=True)
        except KeyboardInterrupt:
            print("HTTP сервер остановлен")
        except Exception as e:
            print(f"Ошибка HTTP: {e}")
    
    def detect_people(self, frame):
        # Детекция с настройками качества
        results = self.model(frame, conf=0.4, iou=0.5, verbose=False)
        
        person_count = 0
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Только люди (class_id = 0)
                    if class_id == 0:
                        person_count += 1
                        
                        # Цвет квадрата по уверенности
                        if confidence > 0.8:
                            color = (0, 255, 0)      # Зеленый
                        elif confidence > 0.6:
                            color = (0, 255, 255)    # Желтый
                        else:
                            color = (0, 165, 255)    # Оранжевый
                        
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                                    color, 2)
                        
                        # Фон для текста
                        label = f"Person {confidence:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(frame, (int(x1), int(y1-25)), 
                                    (int(x1 + label_size[0]), int(y1)), color, -1)
                        
                        cv2.putText(frame, label, (int(x1), int(y1-5)), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Счетчик людей
        cv2.putText(frame, f"People: {person_count}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        return frame


if __name__ == "__main__":
    STREAM_URL = "http://13.61.178.32/12_38/stream.m3u8"
    
    print("YOLO Stream Processor")
    print("1. Окно с обработкой")
    print("2. HTTP поток")
    
    choice = input("Ваш выбор (1 или 2): ").strip()
    
    try:
        processor = YOLOStreamProcessor(STREAM_URL)
        
        if choice == "1":
            processor.process_stream_window()
        elif choice == "2":
            processor.process_stream_http()
        else:
            print("Выберите 1 или 2")
            
    except KeyboardInterrupt:
        print("Остановлено")
    except Exception as e:
        print(f"Ошибка: {e}")
        print("Установите: pip install ultralytics opencv-python flask")