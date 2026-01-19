import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from typing import List, Tuple, Optional
import os
from config import MONITOR_MODEL_PATH, FACE_MODEL_PATH, USE_DEFAULT_YOLO
from head_pose import HeadPoseEstimator
from deep_pixbis import DeepPixBisDetector
from anti_spoofing import AntiSpoofing


class DetectionEngine:
    """Класс для детекции монитора и лица с помощью YOLO"""
    
    def __init__(self):
        """Инициализация моделей YOLO"""
        self.monitor_model = None
        self.face_model = None
        self.head_pose_estimator = None
        self.deep_pixbis_detector = None
        self.anti_spoofing = None  # Будет создан при первом использовании
        self._load_models()
        # Head pose estimator НЕ загружается по умолчанию (очень медленный)
        # Загружается только если явно нужен (use_head_pose=True или use_anti_spoofing=True)
        self._load_deep_pixbis()
    
    def _load_models(self):
        """Загрузка моделей YOLO"""
        try:
            # Загружаем модель для детекции монитора
            if os.path.exists(MONITOR_MODEL_PATH):
                self.monitor_model = YOLO(MONITOR_MODEL_PATH)
                print(f"✅ Загружена модель монитора: {MONITOR_MODEL_PATH}")
            else:
                # Используем стандартную YOLO модель если кастомная не найдена
                if USE_DEFAULT_YOLO:
                    self.monitor_model = YOLO('yolo11n.pt')
                    print("⚠️ Используется стандартная YOLO модель для детекции монитора")
                else:
                    raise FileNotFoundError(f"Модель монитора не найдена: {MONITOR_MODEL_PATH}")
            
            # Загружаем модель для детекции лица
            if os.path.exists(FACE_MODEL_PATH):
                self.face_model = YOLO(FACE_MODEL_PATH)
                print(f"✅ Загружена модель лица: {FACE_MODEL_PATH}")
            else:
                # Используем стандартную YOLO модель если кастомная не найдена
                if USE_DEFAULT_YOLO:
                    self.face_model = YOLO('yolo11n.pt')
                    print("⚠️ Используется стандартная YOLO модель для детекции лица")
                    print("ВНИМАНИЕ: Для лучшей детекции лица рекомендуется использовать специализированную модель")
                else:
                    raise FileNotFoundError(f"Модель лица не найдена: {FACE_MODEL_PATH}")
        
        except Exception as e:
            print(f"❌ Ошибка при загрузке моделей: {e}")
            raise
    
    def _load_head_pose_estimator(self):
        """Загрузка модуля для определения позы головы"""
        try:
            self.head_pose_estimator = HeadPoseEstimator(device='cpu')
        except Exception as e:
            print(f"⚠️ Не удалось загрузить head pose estimator: {e}")
            self.head_pose_estimator = None
    
    def _load_deep_pixbis(self):
        """Загрузка DeepPixBis детектора"""
        try:
            # Путь к модели DeepPixBis (если есть)
            deeppixbis_model_path = os.getenv('DEEPPIXBIS_MODEL_PATH', None)
            if deeppixbis_model_path and os.path.exists(deeppixbis_model_path):
                self.deep_pixbis_detector = DeepPixBisDetector(
                    model_path=deeppixbis_model_path,
                    device='cpu'  # Можно изменить на 'cuda' если есть GPU
                )
                print("✅ DeepPixBis детектор загружен")
            else:
                # Создаем детектор без предобученной модели
                self.deep_pixbis_detector = DeepPixBisDetector(device='cpu')
                print("⚠️ DeepPixBis детектор создан без предобученной модели")
        except Exception as e:
            print(f"⚠️ Не удалось загрузить DeepPixBis: {e}")
            self.deep_pixbis_detector = None
    
    def _load_anti_spoofing(self):
        """Загрузка AntiSpoofing модуля (требует head_pose_estimator для landmarks)"""
        try:
            # Загружаем head pose estimator если еще не загружен
            if self.head_pose_estimator is None:
                self._load_head_pose_estimator()
            
            # Создаем AntiSpoofing экземпляр
            self.anti_spoofing = AntiSpoofing(self.head_pose_estimator)
        except Exception as e:
            print(f"⚠️ Не удалось загрузить AntiSpoofing: {e}")
            self.anti_spoofing = None
    
    def detect_monitors(self, image_path: str) -> List[dict]:
        """
        Детекция мониторов на изображении
        
        Args:
            image_path: Путь к изображению
            
        Returns:
            Список словарей с координатами и уверенностью детекции мониторов
            Формат: [{'bbox': [x1, y1, x2, y2], 'confidence': float}, ...]
        """
        try:
            results = self.monitor_model(image_path)
            monitors = []
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Получаем координаты bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Если используем стандартную модель, фильтруем по классу "monitor" или "screen"
                    # В стандартной YOLO модели может не быть класса монитор, поэтому берем все объекты
                    # или можно использовать класс "laptop" (класс 63) или другие похожие
                    monitors.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': confidence,
                        'class_id': class_id
                    })
            
            return monitors
        
        except Exception as e:
            print(f"Ошибка при детекции мониторов: {e}")
            return []
    
    def detect_faces(self, image_path: str) -> List[dict]:
        """
        Детекция лиц на изображении
        
        Args:
            image_path: Путь к изображению
            
        Returns:
            Список словарей с координатами и уверенностью детекции лиц
            Формат: [{'bbox': [x1, y1, x2, y2], 'confidence': float}, ...]
        """
        try:
            results = self.face_model(image_path)
            faces = []
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    faces.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': confidence,
                        'class_id': class_id
                    })
            
            return faces
        
        except Exception as e:
            print(f"Ошибка при детекции лиц: {e}")
            return []
    
    def is_face_inside_monitor(self, face_bbox: List[float], monitor_bbox: List[float]) -> bool:
        """
        Проверка находится ли лицо внутри монитора
        
        Args:
            face_bbox: [x1, y1, x2, y2] координаты лица
            monitor_bbox: [x1, y1, x2, y2] координаты монитора
            
        Returns:
            True если лицо находится внутри монитора
        """
        face_x1, face_y1, face_x2, face_y2 = face_bbox
        monitor_x1, monitor_y1, monitor_x2, monitor_y2 = monitor_bbox
        
        # Проверяем что центр лица находится внутри монитора
        face_center_x = (face_x1 + face_x2) / 2
        face_center_y = (face_y1 + face_y2) / 2
        
        # Проверяем что центр лица находится внутри bounding box монитора
        is_inside = (monitor_x1 <= face_center_x <= monitor_x2 and 
                    monitor_y1 <= face_center_y <= monitor_y2)
        
        return is_inside
    
    def is_face_too_large(self, face_bbox: List[float], image_width: int, image_height: int, 
                         threshold: float = 0.5) -> bool:
        """
        Проверка занимает ли лицо слишком много места на изображении
        (подозрительно для лиц на экране при приближении)
        
        Args:
            face_bbox: [x1, y1, x2, y2] координаты лица
            image_width: Ширина изображения
            image_height: Высота изображения
            threshold: Порог (0.5 = 50% площади изображения)
            
        Returns:
            True если лицо занимает больше threshold площади изображения
        """
        face_x1, face_y1, face_x2, face_y2 = face_bbox
        
        # Вычисляем площадь лица
        face_width = face_x2 - face_x1
        face_height = face_y2 - face_y1
        face_area = face_width * face_height
        
        # Вычисляем площадь изображения
        image_area = image_width * image_height
        
        # Проверяем соотношение
        face_ratio = face_area / image_area if image_area > 0 else 0
        
        return face_ratio > threshold
    
    def analyze_image(self, image_path: str, use_head_pose: bool = False, use_anti_spoofing: bool = True) -> dict:
        """
        Анализ изображения: детекция мониторов и лиц, проверка фейка
        
        Args:
            image_path: Путь к изображению
            
        Returns:
            Словарь с результатами анализа:
            {
                'is_fake': bool,
                'monitors': List[dict],
                'faces': List[dict],
                'face_in_monitor': List[dict] - список лиц внутри мониторов,
                'head_pose': dict - углы поворота головы (если определены),
                'head_movement_check': dict - результаты проверки движения головы,
                'suspicious_large_face': bool - подозрительно большое лицо без монитора
            }
        """
        import cv2
        
        # Получаем размеры изображения
        img = cv2.imread(image_path)
        if img is None:
            return {'error': 'Не удалось загрузить изображение'}
        
        image_height, image_width = img.shape[:2]
        
        monitors = self.detect_monitors(image_path)
        faces = self.detect_faces(image_path)
        
        face_in_monitor = []
        is_fake = False
        head_pose = None
        head_movement_check = None
        suspicious_large_face = False
        block_artifacts_analysis = None
        all_artifacts = []
        anti_spoofing_results = {}  # Результаты антиспуфинга для каждого лица
        
        # Если найден монитор в кадре - это сразу фейк
        if len(monitors) > 0:
            is_fake = True
        
        # Загружаем AntiSpoofing если нужно и еще не загружен
        if use_anti_spoofing and self.anti_spoofing is None:
            self._load_anti_spoofing()
        
        # Проверяем каждое лицо относительно каждого монитора
        for face in faces:
            face_found_in_monitor = False
            
            # Сначала проверяем размер лица - если большое, это всегда подозрительно
            if self.is_face_too_large(face['bbox'], image_width, image_height, threshold=0.4):
                suspicious_large_face = True
                face['suspicious_large'] = True
                face['face_area_ratio'] = self._calculate_face_area_ratio(
                    face['bbox'], image_width, image_height
                )
                # Если лицо большое - это фейк, независимо от детекции монитора
                is_fake = True
            
            # Анализ блочных артефактов отключен
            # try:
            #     artifacts = self.analyze_block_artifacts(image_path, face['bbox'])
            #     if 'error' not in artifacts:
            #         face['block_artifacts'] = artifacts
            #         all_artifacts.append(artifacts)
            #         
            #         # Если обнаружены артефакты сжатия - это подозрительно (возможно экран)
            #         if artifacts.get('has_artifacts', False):
            #             is_fake = True
            # except Exception as e:
            #     print(f"Ошибка при анализе артефактов: {e}")
            
            for monitor in monitors:
                if self.is_face_inside_monitor(face['bbox'], monitor['bbox']):
                    face_in_monitor.append({
                        'face': face,
                        'monitor': monitor
                    })
                    is_fake = True
                    face_found_in_monitor = True
                    
                    # Head pose estimation отключен по умолчанию для ускорения
                    # Можно включить передав use_head_pose=True
                    if use_head_pose:
                        # Загружаем head pose estimator только если нужен
                        if self.head_pose_estimator is None:
                            self._load_head_pose_estimator()
                        
                        if self.head_pose_estimator is not None:
                            try:
                                head_pose = self.head_pose_estimator.get_head_pose(
                                    image_path, 
                                    face['bbox']
                                )
                                
                                if head_pose:
                                    head_movement_check = {
                                        'pose_detected': True,
                                        'pitch': head_pose['pitch'],
                                        'yaw': head_pose['yaw'],
                                        'roll': head_pose['roll'],
                                        'note': 'Для проверки движения нужны несколько кадров'
                                    }
                            except Exception as e:
                                print(f"Ошибка при определении позы головы: {e}")
            
            # Если лицо большое, но не найдено в мониторе - дополнительно подозрительно
            if not face_found_in_monitor and face.get('suspicious_large', False):
                # Лицо большое и не в мониторе - очень подозрительно
                pass  # Уже помечено как фейк выше
            
            # Антиспуфинг анализ (блики в глазах, освещение)
            if use_anti_spoofing and self.anti_spoofing is not None:
                try:
                    # Получаем landmarks если head_pose_estimator доступен
                    landmarks = None
                    if self.head_pose_estimator is not None:
                        try:
                            landmarks = self.head_pose_estimator.get_face_landmarks(image_path)
                        except:
                            pass
                    
                    # 1. Детекция бликов в глазах
                    eye_reflections = self.anti_spoofing.detect_eye_reflections(
                        image_path, face['bbox'], landmarks
                    )
                    if 'error' not in eye_reflections:
                        face['eye_reflections'] = eye_reflections
                        # Если нет бликов в глазах - подозрительно (возможно изображение на экране)
                        if not eye_reflections.get('has_reflections', False):
                            # Отсутствие бликов - слабый признак фейка, но учитываем
                            # Не устанавливаем is_fake = True сразу, но добавляем в score
                            face['no_eye_reflections'] = True
                    
                    # 2. Анализ освещения
                    lighting = self.anti_spoofing.analyze_lighting_consistency(
                        image_path, face['bbox'], landmarks
                    )
                    if 'error' not in lighting:
                        face['lighting'] = lighting
                        # Если освещение неестественное - подозрительно
                        if not lighting.get('is_consistent', True):
                            # Неестественное освещение - признак фейка
                            face['unnatural_lighting'] = True
                            # Не устанавливаем is_fake = True сразу, но учитываем
                    
                    # Сохраняем результаты антиспуфинга для этого лица
                    anti_spoofing_results[f"face_{len(anti_spoofing_results)}"] = {
                        'eye_reflections': eye_reflections if 'error' not in eye_reflections else None,
                        'lighting': lighting if 'error' not in lighting else None
                    }
                    
                except Exception as e:
                    print(f"⚠️ Ошибка при антиспуфинг анализе: {e}")
                    # Продолжаем работу даже если антиспуфинг не сработал
        
        # Анализ артефактов отключен
        # if all_artifacts:
        #     block_artifacts_analysis = max(all_artifacts, key=lambda x: x.get('artifact_score', 0))
        
        return {
            'is_fake': is_fake,
            'monitors': monitors,
            'faces': faces,
            'face_in_monitor': face_in_monitor,
            'head_pose': head_pose,
            'head_movement_check': head_movement_check,
            'suspicious_large_face': suspicious_large_face,
            'block_artifacts_analysis': block_artifacts_analysis,
            'anti_spoofing_results': anti_spoofing_results
        }
    
    def analyze_image_with_deeppixbis(self, image_path: str) -> dict:
        """
        Анализ изображения используя DeepPixBis
        
        Args:
            image_path: Путь к изображению
            
        Returns:
            Словарь с результатами анализа
        """
        import cv2
        
        # Получаем размеры изображения
        img = cv2.imread(image_path)
        if img is None:
            return {'error': 'Не удалось загрузить изображение'}
        
        image_height, image_width = img.shape[:2]
        
        # Детектируем лица
        faces = self.detect_faces(image_path)
        
        if len(faces) == 0:
            return {
                'is_fake': False,
                'monitors': [],
                'faces': [],
                'face_in_monitor': [],
                'deeppixbis_results': None,
                'method': 'deeppixbis'
            }
        
        # Анализируем каждое лицо с помощью DeepPixBis
        deeppixbis_results = []
        is_fake = False
        
        if self.deep_pixbis_detector is None:
            return {
                'error': 'DeepPixBis детектор не загружен',
                'is_fake': False,
                'monitors': [],
                'faces': faces,
                'face_in_monitor': [],
                'method': 'deeppixbis'
            }
        
        for face in faces:
            result = self.deep_pixbis_detector.predict_from_image_path(
                image_path, 
                face['bbox']
            )
            
            if 'error' not in result:
                face['deeppixbis'] = result
                deeppixbis_results.append(result)
                
                # Если DeepPixBis определил как fake
                if result.get('is_fake', False):
                    is_fake = True
        
        return {
            'is_fake': is_fake,
            'monitors': [],  # DeepPixBis не использует детекцию мониторов
            'faces': faces,
            'face_in_monitor': [],
            'deeppixbis_results': deeppixbis_results,
            'method': 'deeppixbis'
        }
    
    def _calculate_face_area_ratio(self, face_bbox: List[float], image_width: int, image_height: int) -> float:
        """Вычисляет соотношение площади лица к площади изображения"""
        face_x1, face_y1, face_x2, face_y2 = face_bbox
        face_area = (face_x2 - face_x1) * (face_y2 - face_y1)
        image_area = image_width * image_height
        return face_area / image_area if image_area > 0 else 0
    
    def analyze_block_artifacts(self, image_path: str, face_bbox: List[float], 
                               block_size: int = 8, variance_threshold: float = 50.0) -> dict:
        """
        Анализирует блочные артефакты на лице (характерно для JPEG сжатия и экранов)
        
        Args:
            image_path: Путь к изображению
            face_bbox: [x1, y1, x2, y2] координаты лица
            block_size: Размер блока для анализа (обычно 8x8 для JPEG)
            variance_threshold: Порог дисперсии для определения артефактов
            
        Returns:
            Словарь с результатами анализа:
            {
                'has_artifacts': bool,
                'artifact_score': float (0-1),
                'low_variance_blocks_ratio': float,
                'total_blocks': int,
                'low_variance_blocks': int
            }
        """
        import cv2
        
        # Загружаем изображение
        img = cv2.imread(image_path)
        if img is None:
            return {'error': 'Не удалось загрузить изображение'}
        
        # Конвертируем в grayscale для анализа
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Извлекаем область лица
        x1, y1, x2, y2 = map(int, face_bbox)
        # Убеждаемся что координаты в пределах изображения
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(gray.shape[1], x2)
        y2 = min(gray.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return {'error': 'Некорректные координаты лица'}
        
        face_region = gray[y1:y2, x1:x2]
        
        if face_region.size == 0:
            return {'error': 'Область лица пуста'}
        
        # Разбиваем на блоки 8x8
        height, width = face_region.shape
        total_blocks = 0
        low_variance_blocks = 0
        block_variances = []
        
        # Проходим по блокам
        for y in range(0, height - block_size + 1, block_size):
            for x in range(0, width - block_size + 1, block_size):
                block = face_region[y:y+block_size, x:x+block_size]
                
                # Вычисляем дисперсию блока
                variance = np.var(block)
                block_variances.append(variance)
                total_blocks += 1
                
                # Если дисперсия низкая - это может быть артефакт сжатия
                if variance < variance_threshold:
                    low_variance_blocks += 1
        
        if total_blocks == 0:
            return {'error': 'Не удалось создать блоки'}
        
        # Вычисляем соотношение блоков с низкой дисперсией
        low_variance_ratio = low_variance_blocks / total_blocks
        
        # Score артефактов (0-1, где 1 = много артефактов)
        # Если >30% блоков имеют низкую дисперсию - подозрительно
        artifact_score = min(1.0, low_variance_ratio / 0.3)
        
        # Считаем что есть артефакты если >20% блоков с низкой дисперсией
        has_artifacts = low_variance_ratio > 0.2
        
        return {
            'has_artifacts': has_artifacts,
            'artifact_score': artifact_score,
            'low_variance_blocks_ratio': low_variance_ratio,
            'total_blocks': total_blocks,
            'low_variance_blocks': low_variance_blocks,
            'average_variance': np.mean(block_variances) if block_variances else 0,
            'min_variance': np.min(block_variances) if block_variances else 0,
            'max_variance': np.max(block_variances) if block_variances else 0
        }
    
    def analyze_video(self, video_path: str, sample_rate: int = 5) -> dict:
        """
        Анализ видео: детекция мониторов, лиц и проверка движения головы
        
        Args:
            video_path: Путь к видео файлу
            sample_rate: Каждый N-й кадр для анализа (для ускорения)
            
        Returns:
            Словарь с результатами анализа видео
        """
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'error': 'Не удалось открыть видео'}
        
        frames_analyzed = []
        head_poses = []
        eye_history = []  # История состояний глаз для анализа моргания
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Загружаем AntiSpoofing для анализа моргания
        if self.anti_spoofing is None:
            self._load_anti_spoofing()
        
        frame_count = 0
        analyzed_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Анализируем каждый N-й кадр
            if frame_count % sample_rate == 0:
                # Сохраняем временный кадр
                temp_frame_path = f"temp_frame_{frame_count}.jpg"
                cv2.imwrite(temp_frame_path, frame)
                
                # Анализируем кадр (с антиспуфингом)
                frame_analysis = self.analyze_image(temp_frame_path, use_head_pose=True, use_anti_spoofing=True)
                current_time = frame_count / fps if fps > 0 else 0
                
                frames_analyzed.append({
                    'frame': frame_count,
                    'time': current_time,
                    'analysis': frame_analysis
                })
                
                # Собираем позы головы
                if frame_analysis.get('head_pose'):
                    head_poses.append(frame_analysis['head_pose'])
                
                # Собираем историю состояний глаз для анализа моргания
                if self.anti_spoofing is not None and len(frame_analysis.get('faces', [])) > 0:
                    # Берем первое лицо для анализа
                    face = frame_analysis['faces'][0]
                    if 'eye_reflections' in face:
                        # Проверяем открытость глаз
                        try:
                            landmarks = None
                            if self.head_pose_estimator is not None:
                                landmarks = self.head_pose_estimator.get_face_landmarks(temp_frame_path)
                            
                            if landmarks is not None:
                                eye_openness = self.anti_spoofing.check_eye_openness(temp_frame_path, landmarks)
                                if 'error' not in eye_openness:
                                    eye_history.append({
                                        'left_eye_open': eye_openness.get('left_eye_open', True),
                                        'right_eye_open': eye_openness.get('right_eye_open', True),
                                        'timestamp': current_time,
                                        'frame': frame_count
                                    })
                        except Exception as e:
                            print(f"⚠️ Ошибка при проверке открытости глаз: {e}")
                
                # Удаляем временный файл
                try:
                    os.remove(temp_frame_path)
                except:
                    pass
                
                analyzed_count += 1
            
            frame_count += 1
        
        cap.release()
        
        # Анализируем движение головы
        head_movement_analysis = None
        if self.head_pose_estimator is not None and len(head_poses) >= 2:
            head_movement_analysis = self.head_pose_estimator.is_natural_head_movement(
                head_poses,
                min_movement=3.0,
                max_static_frames=3
            )
        
        # Анализируем моргание
        blinking_analysis = None
        if self.anti_spoofing is not None and len(eye_history) >= 3:
            try:
                blinking_analysis = self.anti_spoofing.detect_blinking(eye_history)
            except Exception as e:
                print(f"⚠️ Ошибка при анализе моргания: {e}")
        
        # Определяем общий результат
        is_fake = any(frame['analysis']['is_fake'] for frame in frames_analyzed)
        unnatural_movement = False
        
        if head_movement_analysis and not head_movement_analysis.get('is_natural', True):
            unnatural_movement = True
            is_fake = True  # Если движение неестественное - это фейк
        
        # Если нет моргания или неестественное моргание - подозрительно
        if blinking_analysis:
            if not blinking_analysis.get('is_blinking', False):
                # Нет моргания - подозрительно (возможно статичное изображение)
                is_fake = True
            elif not blinking_analysis.get('is_natural', True):
                # Неестественное моргание - подозрительно
                is_fake = True
        
        return {
            'is_fake': is_fake,
            'unnatural_movement': unnatural_movement,
            'total_frames': total_frames,
            'analyzed_frames': analyzed_count,
            'frames_analysis': frames_analyzed,
            'head_poses': head_poses,
            'head_movement_analysis': head_movement_analysis,
            'blinking_analysis': blinking_analysis,
            'eye_history': eye_history,
            'fps': fps
        }
