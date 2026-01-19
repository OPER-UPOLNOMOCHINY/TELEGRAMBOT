# План улучшения антиспуфинга

## Текущее состояние

### Реализованные методы:
1. ✅ Детекция монитора в кадре (YOLO)
2. ✅ Детекция лица внутри монитора
3. ✅ Проверка размера лица (>40% площади = фейк)
4. ✅ Head pose estimation (отключен для производительности)
5. ✅ DeepPixBis (пиксельная детекция)
6. ⚠️ Анализ блочных артефактов (отключен)
7. ⚠️ Методы из `anti_spoofing.py` (не интегрированы в основной анализ)

### Проблемы:
- Методы из `anti_spoofing.py` не используются в основном анализе
- Head pose отключен из-за производительности
- Анализ артефактов отключен
- Нет комбинированного подхода с весами для разных методов

---

## Предлагаемые улучшения

### 1. Интеграция методов из `anti_spoofing.py` ⭐ ВЫСОКИЙ ПРИОРИТЕТ

**Методы для интеграции:**
- **Детекция бликов в глазах** (`detect_eye_reflections`)
  - Реальные глаза имеют блики от источника света
  - Изображения на экране обычно не имеют естественных бликов
  - Производительность: быстрая
  
- **Анализ согласованности освещения** (`analyze_lighting_consistency`)
  - Реальное лицо имеет естественные тени и блики
  - Изображение на экране может иметь неестественное освещение
  - Производительность: средняя

- **Детекция моргания** (`detect_blinking`) - для видео
  - Реальное лицо моргает естественным образом
  - Статичное изображение не моргает
  - Производительность: быстрая (требует несколько кадров)

**Реализация:**
```python
# В detection.py добавить интеграцию с AntiSpoofing
def analyze_image(self, image_path: str, use_anti_spoofing: bool = True):
    # ... существующий код ...
    
    if use_anti_spoofing:
        anti_spoofing = AntiSpoofing(self.head_pose_estimator)
        for face in faces:
            # Проверка бликов в глазах
            eye_reflections = anti_spoofing.detect_eye_reflections(
                image_path, face['bbox']
            )
            # Проверка освещения
            lighting = anti_spoofing.analyze_lighting_consistency(
                image_path, face['bbox']
            )
            # Комбинируем результаты
```

---

### 2. Система взвешенных оценок (Ensemble) ⭐ ВЫСОКИЙ ПРИОРИТЕТ

**Идея:** Комбинировать результаты разных методов с весами

**Методы и их веса:**
- Детекция монитора: 0.3 (высокая надежность)
- Лицо в мониторе: 0.25 (высокая надежность)
- Большое лицо: 0.15 (средняя надежность)
- Блики в глазах: 0.1 (средняя надежность)
- Освещение: 0.1 (средняя надежность)
- DeepPixBis: 0.1 (если используется)

**Реализация:**
```python
def calculate_spoof_score(self, analysis_results: dict) -> float:
    """Вычисляет общий spoof score (0-1, где 1 = точно фейк)"""
    weights = {
        'monitor_detected': 0.3,
        'face_in_monitor': 0.25,
        'large_face': 0.15,
        'eye_reflections': 0.1,
        'lighting': 0.1,
        'deeppixbis': 0.1
    }
    
    score = 0.0
    if analysis_results.get('monitors'):
        score += weights['monitor_detected']
    if analysis_results.get('face_in_monitor'):
        score += weights['face_in_monitor']
    # ... и т.д.
    
    return min(1.0, score)
```

---

### 3. Улучшение детекции монитора

**Проблема:** Большой монитор при приближении может не детектироваться

**Решения:**
- Использовать несколько порогов уверенности (conf_threshold)
- Анализировать края изображения (монитор может занимать весь кадр)
- Проверять соотношение сторон (мониторы обычно 16:9, 4:3)
- Детектировать рамки монитора (четкие границы)

**Реализация:**
```python
def detect_monitor_edges(self, image_path: str) -> dict:
    """Детектирует края монитора через анализ границ"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Детекция краев (Canny)
    edges = cv2.Canny(gray, 50, 150)
    
    # Поиск прямоугольников (рамка монитора)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Фильтруем по размеру и форме
    for contour in contours:
        # Проверяем что это прямоугольник
        # Проверяем размер (занимает большую часть кадра)
        pass
```

---

### 4. Анализ глубины (Depth Analysis)

**Идея:** Реальное лицо имеет 3D структуру, изображение на экране - плоское

**Методы:**
- Анализ градиентов яркости (реальное лицо имеет плавные переходы)
- Анализ текстур (реальное лицо имеет более сложные текстуры)
- Анализ отражений (реальное лицо отражает свет по-другому)

**Реализация:**
```python
def analyze_depth_cues(self, image_path: str, face_bbox: List[float]) -> dict:
    """Анализирует признаки глубины на лице"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Извлекаем область лица
    x1, y1, x2, y2 = map(int, face_bbox)
    face_region = gray[y1:y2, x1:x2]
    
    # Анализ градиентов (Sobel)
    grad_x = cv2.Sobel(face_region, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(face_region, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Реальное лицо имеет более плавные градиенты
    gradient_variance = np.var(gradient_magnitude)
    
    # Анализ текстур (Local Binary Pattern)
    from skimage import feature
    lbp = feature.local_binary_pattern(face_region, 8, 1, method='uniform')
    texture_complexity = np.var(lbp)
    
    return {
        'gradient_variance': gradient_variance,
        'texture_complexity': texture_complexity,
        'is_3d': gradient_variance > threshold and texture_complexity > threshold
    }
```

---

### 5. Анализ цветовых пространств

**Идея:** Изображения на экране могут иметь неестественные цвета

**Методы:**
- Анализ цветового баланса (реальное лицо имеет естественный баланс)
- Анализ насыщенности (изображения на экране могут быть пере/недо-насыщенными)
- Анализ цветовых переходов (реальное лицо имеет плавные переходы)

**Реализация:**
```python
def analyze_color_space(self, image_path: str, face_bbox: List[float]) -> dict:
    """Анализирует цветовое пространство лица"""
    img = cv2.imread(image_path)
    
    # Конвертируем в разные цветовые пространства
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Извлекаем область лица
    x1, y1, x2, y2 = map(int, face_bbox)
    face_hsv = hsv[y1:y2, x1:x2]
    face_lab = lab[y1:y2, x1:x2]
    
    # Анализ насыщенности (S в HSV)
    saturation = face_hsv[:, :, 1]
    saturation_mean = np.mean(saturation)
    saturation_std = np.std(saturation)
    
    # Реальное лицо имеет умеренную насыщенность (20-80)
    is_natural_saturation = 20 <= saturation_mean <= 80
    
    # Анализ цветового баланса (a и b в LAB)
    a_channel = face_lab[:, :, 1]
    b_channel = face_lab[:, :, 2]
    
    # Реальное лицо имеет определенный диапазон цветов
    a_mean = np.mean(a_channel)
    b_mean = np.mean(b_channel)
    
    # Нормальные значения для лица: a ~128, b ~128
    is_natural_color = (100 <= a_mean <= 150) and (100 <= b_mean <= 150)
    
    return {
        'saturation_mean': saturation_mean,
        'saturation_std': saturation_std,
        'is_natural_saturation': is_natural_saturation,
        'is_natural_color': is_natural_color,
        'is_real': is_natural_saturation and is_natural_color
    }
```

---

### 6. Оптимизация производительности

**Проблема:** Head pose estimation очень медленный

**Решения:**
- Использовать более легкую модель для landmarks (например, MediaPipe Face Mesh)
- Кэшировать результаты для одинаковых изображений
- Использовать GPU если доступно
- Параллельная обработка нескольких лиц

**Реализация:**
```python
# Использовать MediaPipe вместо face-alignment (быстрее)
import mediapipe as mp

class FastHeadPoseEstimator:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True
        )
    
    def get_landmarks(self, image_path: str):
        # MediaPipe быстрее чем face-alignment
        pass
```

---

### 7. Анализ движения для видео

**Улучшения:**
- Более точный анализ движения головы
- Анализ микродвижений (реальное лицо имеет естественные микродвижения)
- Анализ стабильности (статичное изображение слишком стабильно)

**Реализация:**
```python
def analyze_micro_movements(self, video_path: str) -> dict:
    """Анализирует микродвижения лица"""
    cap = cv2.VideoCapture(video_path)
    
    face_positions = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Детектируем лицо
        faces = self.detect_faces(frame)
        if faces:
            face_center = self._get_face_center(faces[0]['bbox'])
            face_positions.append(face_center)
    
    # Анализируем вариацию позиций
    if len(face_positions) > 1:
        positions_array = np.array(face_positions)
        position_variance = np.var(positions_array, axis=0)
        
        # Реальное лицо имеет небольшие микродвижения
        has_micro_movements = np.mean(position_variance) > threshold
    
    return {
        'has_micro_movements': has_micro_movements,
        'position_variance': position_variance
    }
```

---

### 8. Машинное обучение для улучшения

**Идея:** Обучить модель на собранных данных

**Подход:**
- Использовать собранные данные из `retrain_data/`
- Обучить классификатор (например, Random Forest или XGBoost) на признаках:
  - Размер лица
  - Наличие монитора
  - Блики в глазах
  - Освещение
  - И т.д.
- Использовать обученную модель для финального решения

**Реализация:**
```python
from sklearn.ensemble import RandomForestClassifier
import joblib

class SpoofClassifier:
    def __init__(self, model_path: str = None):
        if model_path and os.path.exists(model_path):
            self.model = joblib.load(model_path)
        else:
            self.model = RandomForestClassifier(n_estimators=100)
    
    def extract_features(self, analysis_results: dict) -> np.ndarray:
        """Извлекает признаки из результатов анализа"""
        features = [
            len(analysis_results.get('monitors', [])),
            len(analysis_results.get('face_in_monitor', [])),
            analysis_results.get('suspicious_large_face', False),
            analysis_results.get('eye_reflections', {}).get('reflection_score', 0),
            analysis_results.get('lighting', {}).get('lighting_score', 0),
            # ... и т.д.
        ]
        return np.array(features).reshape(1, -1)
    
    def predict(self, analysis_results: dict) -> dict:
        features = self.extract_features(analysis_results)
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0]
        
        return {
            'is_fake': prediction == 1,
            'confidence': max(probability),
            'fake_probability': probability[1] if len(probability) > 1 else 0
        }
```

---

## Приоритеты реализации

### Фаза 1 (Быстрые улучшения):
1. ✅ Интеграция методов из `anti_spoofing.py`
2. ✅ Система взвешенных оценок
3. ✅ Улучшение детекции монитора

### Фаза 2 (Средние улучшения):
4. ✅ Анализ глубины
5. ✅ Анализ цветовых пространств
6. ✅ Оптимизация производительности

### Фаза 3 (Долгосрочные):
7. ✅ Улучшенный анализ движения
8. ✅ Машинное обучение

---

## Метрики для оценки

- **Точность (Accuracy)**: % правильных определений
- **Precision**: % фейков среди определенных как фейк
- **Recall**: % найденных фейков среди всех фейков
- **F1-Score**: Гармоническое среднее Precision и Recall
- **Время обработки**: Среднее время анализа одного изображения

---

## Тестирование

1. Создать тестовый набор данных:
   - Реальные лица (100+ изображений)
   - Фейки (лица на экране, 100+ изображений)
   
2. Оценить каждый метод отдельно
3. Оценить комбинированный подход
4. Оптимизировать веса методов
