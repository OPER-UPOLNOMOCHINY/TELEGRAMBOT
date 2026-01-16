import os
import logging
import shutil
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, WebAppInfo
from telegram.error import Conflict
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters
)
from PIL import Image
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path

from config import (
    TELEGRAM_BOT_TOKEN, RETRAIN_IMAGES_DIR, RETRAIN_LABELS_DIR, 
    ALL_PHOTOS_DIR, RETRAIN_METADATA_DIR, TEMP_VIDEOS_DIR, FACEID_WEBAPP_URL
)
import json
from detection import DetectionEngine

# Настройка логирования
# Создаем директорию для логов
LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)

# Настройка логирования с записью в файл и консоль
log_file = os.path.join(LOG_DIR, f'bot_{datetime.now().strftime("%Y%m%d")}.log')
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),  # Запись в файл
        logging.StreamHandler()  # Вывод в консоль
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Логи записываются в файл: {log_file}")

# Глобальный объект для детекции
detection_engine = DetectionEngine()

# Хранилище данных о текущих анализах пользователей
# Формат: {user_id: {'image_path': str, 'analysis_result': dict, 'message_id': int, 'photo_history': list}}
user_sessions = {}

# Хранилище истории фото для анализа движения (последние N фото)
# Формат: {user_id: [{'image_path': str, 'head_pose': dict, 'timestamp': datetime}, ...]}
user_photo_history = {}


def draw_detections(image_path: str, monitors: list, faces: list, face_in_monitor: list) -> str:
    """
    Рисует детекции на изображении и сохраняет результат
    
    Args:
        image_path: Путь к исходному изображению
        monitors: Список детектированных мониторов
        faces: Список детектированных лиц
        face_in_monitor: Список лиц внутри мониторов
        
    Returns:
        Путь к обработанному изображению
    """
    img = cv2.imread(image_path)
    if img is None:
        return image_path
    
    # Рисуем мониторы (зеленым)
    for monitor in monitors:
        x1, y1, x2, y2 = map(int, monitor['bbox'])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"Monitor {monitor['confidence']:.2f}", 
                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Рисуем лица (синим)
    for face in faces:
        x1, y1, x2, y2 = map(int, face['bbox'])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, f"Face {face['confidence']:.2f}", 
                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Рисуем лица внутри мониторов (красным)
    for item in face_in_monitor:
        face = item['face']
        x1, y1, x2, y2 = map(int, face['bbox'])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(img, "FAKE!", (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)
    
    # Сохраняем обработанное изображение
    output_path = image_path.replace('.jpg', '_detected.jpg').replace('.png', '_detected.png')
    if output_path == image_path:
        output_path = image_path.rsplit('.', 1)[0] + '_detected.' + image_path.rsplit('.', 1)[1]
    
    cv2.imwrite(output_path, img)
    return output_path


# Хранилище выбранного метода детекции для каждого пользователя
user_detection_method = {}  # {user_id: 'standard' или 'deeppixbis'}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    user_id = update.effective_user.id
    
    # Устанавливаем метод по умолчанию
    if user_id not in user_detection_method:
        user_detection_method[user_id] = 'standard'
    
    welcome_message = (
        "👋 Привет! Я бот для детекции фейков на фото и видео.\n\n"
        "📸 Отправь мне фото или видео, и я проверю:\n"
        "• Есть ли на фото монитор\n"
        "• Есть ли лицо внутри монитора\n"
        "• Размер лица (большое лицо = фейк)\n"
        "• Если лицо внутри монитора - это ФЕЙК! 🚨\n\n"
        "🔧 Выберите метод детекции:\n"
        "• /method_standard - Стандартный метод (монитор + движение)\n"
        "• /method_deeppixbis - DeepPixBis (пиксельная детекция)\n\n"
        "После анализа я спрошу правильно ли я определил, "
        "и сохраню данные для обучения модели (включая информацию о позе головы)."
    )
    
    # Создаем кнопки для выбора метода и Face ID
    keyboard = [
        [
            InlineKeyboardButton("🔧 Стандартный метод", callback_data=f"method_standard_{user_id}"),
            InlineKeyboardButton("🎯 DeepPixBis", callback_data=f"method_deeppixbis_{user_id}")
        ]
    ]
    
    # Добавляем кнопку Face ID только если URL настроен
    if FACEID_WEBAPP_URL:
        keyboard.append([
            InlineKeyboardButton("🔐 Face ID Verification", web_app=WebAppInfo(url=FACEID_WEBAPP_URL))
        ])
    else:
        # Если URL не настроен, добавляем информационную кнопку
        keyboard.append([
            InlineKeyboardButton("🔐 Face ID (требует настройки)", callback_data=f"faceid_info_{user_id}")
        ])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(welcome_message, reply_markup=reply_markup)


async def webcam_info(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /webcam"""
    info_message = (
        "📹 Работа с веб-камерой в реальном времени\n\n"
        "Для запуска анализа с веб-камеры выполните на вашем компьютере:\n\n"
        "```\n"
        "py -3.10 webcam_detection.py\n"
        "```\n\n"
        "Скрипт будет:\n"
        "• Анализировать кадры с веб-камеры в реальном времени\n"
        "• Показывать результаты на экране\n"
        "• Определять фейки (монитор, большое лицо, артефакты)\n"
        "• Анализировать движение головы\n\n"
        "Управление:\n"
        "• 'q' - выход\n"
        "• 's' - сохранить текущий кадр\n"
    )
    await update.message.reply_text(info_message, parse_mode='Markdown')


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик получения фото"""
    user_id = update.effective_user.id
    
    try:
        # Получаем фото
        photo = update.message.photo[-1]  # Берем фото наибольшего размера
        file = await context.bot.get_file(photo.file_id)
        
        # Сохраняем временный файл
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_image_path = f"temp_{user_id}_{timestamp}.jpg"
        await file.download_to_drive(temp_image_path)
        
        # Сохранение всех фото отключено для ускорения (можно включить обратно если нужно)
        # saved_photo_path = os.path.join(ALL_PHOTOS_DIR, f"{user_id}_{timestamp}.jpg")
        # shutil.copy(temp_image_path, saved_photo_path)
        
        # Анализируем изображение
        await update.message.reply_text("🔍 Анализирую изображение...")
        
        # Определяем метод детекции
        detection_method = user_detection_method.get(user_id, 'standard')
        
        if detection_method == 'deeppixbis':
            # Используем DeepPixBis
            analysis_result = detection_engine.analyze_image_with_deeppixbis(temp_image_path)
        else:
            # Используем стандартный метод (без head pose для ускорения)
            analysis_result = detection_engine.analyze_image(temp_image_path, use_head_pose=False)
        
        # Анализ движения головы отключен для ускорения
        head_movement_analysis = None
        
        # Рисуем детекции на изображении
        detected_image_path = draw_detections(
            temp_image_path,
            analysis_result['monitors'],
            analysis_result['faces'],
            analysis_result['face_in_monitor']
        )
        
        # Формируем сообщение с результатами
        detection_method = user_detection_method.get(user_id, 'standard')
        
        if detection_method == 'deeppixbis' and analysis_result.get('method') == 'deeppixbis':
            # Форматируем результаты для DeepPixBis
            result_message = "📊 Результаты анализа (DeepPixBis):\n\n"
            
            if 'error' in analysis_result:
                result_message += f"⚠️ {analysis_result['error']}\n\n"
                result_message += "💡 DeepPixBis модель не загружена. Используйте стандартный метод или загрузите предобученную модель.\n"
            else:
                result_message += f"👤 Лиц найдено: {len(analysis_result['faces'])}\n\n"
                
                deeppixbis_results = analysis_result.get('deeppixbis_results', [])
                if deeppixbis_results:
                    for i, result in enumerate(deeppixbis_results):
                        if 'error' not in result:
                            confidence = result.get('confidence', 0) * 100
                            binary_score = result.get('binary_score', 0) * 100
                            pixel_score = result.get('pixel_score', 0) * 100
                            
                            result_message += f"👤 Лицо {i+1}:\n"
                            result_message += f"• Confidence: {confidence:.1f}%\n"
                            result_message += f"• Binary Score: {binary_score:.1f}%\n"
                            result_message += f"• Pixel Score: {pixel_score:.1f}%\n"
                            result_message += f"• Результат: {'🚨 FAKE' if result.get('is_fake') else '✅ REAL'}\n\n"
                elif len(analysis_result['faces']) == 0:
                    result_message += "❌ Лица не найдены\n"
        else:
            # Стандартный формат
            result_message = "📊 Результаты анализа:\n\n"
            result_message += f"🖥️ Мониторов найдено: {len(analysis_result['monitors'])}\n"
            result_message += f"👤 Лиц найдено: {len(analysis_result['faces'])}\n\n"
        
        is_fake = analysis_result['is_fake']
        unnatural_movement = False
        
        # Проверяем подозрительно большое лицо
        if analysis_result.get('suspicious_large_face'):
            large_faces = [f for f in analysis_result['faces'] if f.get('suspicious_large', False)]
            if large_faces:
                for face in large_faces:
                    ratio = face.get('face_area_ratio', 0) * 100
                    result_message += f"⚠️ ПОДОЗРИТЕЛЬНО БОЛЬШОЕ ЛИЦО!\n"
                    result_message += f"• Лицо занимает {ratio:.1f}% площади изображения\n"
                    result_message += f"• Это может быть лицо на экране монитора!\n"
                    if len(analysis_result['monitors']) == 0:
                        result_message += f"• Монитор не обнаружен - очень подозрительно!\n"
                    elif len(analysis_result['face_in_monitor']) == 0:
                        result_message += f"• Лицо не находится внутри детектированного монитора - подозрительно!\n"
                    result_message += "\n"
        
        # Анализ блочных артефактов отключен
        # if analysis_result.get('block_artifacts_analysis'):
        #     artifacts = analysis_result['block_artifacts_analysis']
        #     if artifacts.get('has_artifacts', False):
        #         ratio = artifacts.get('low_variance_blocks_ratio', 0) * 100
        #         score = artifacts.get('artifact_score', 0) * 100
        #         result_message += f"🔍 ОБНАРУЖЕНЫ АРТЕФАКТЫ СЖАТИЯ!\n"
        #         result_message += f"• Блоков с низкой дисперсией: {ratio:.1f}%\n"
        #         result_message += f"• Score артефактов: {score:.1f}%\n"
        #         result_message += f"• Это может указывать на JPEG сжатие или изображение на экране!\n\n"
        
        if analysis_result['is_fake']:
            result_message += "🚨 ВНИМАНИЕ! Обнаружен ФЕЙК!\n"
            if len(analysis_result['monitors']) > 0:
                result_message += f"• Обнаружен монитор в кадре - это фейк!\n"
            if len(analysis_result['face_in_monitor']) > 0:
                result_message += f"• Найдено лиц внутри монитора: {len(analysis_result['face_in_monitor'])}\n"
            if analysis_result.get('suspicious_large_face'):
                result_message += "• Подозрительно большое лицо (возможно на экране)\n"
            # Анализ артефактов отключен
            # if analysis_result.get('block_artifacts_analysis', {}).get('has_artifacts', False):
            #     result_message += "• Обнаружены артефакты сжатия (возможно изображение на экране)\n"
            result_message += "\n"
            
            # Анализ позы головы отключен для ускорения
            # if analysis_result.get('head_pose'):
            #     head_pose = analysis_result['head_pose']
            #     result_message += "📐 Анализ позы головы:\n"
            #     result_message += f"• Pitch (X): {head_pose['pitch']:.1f}°\n"
            #     result_message += f"• Yaw (Y): {head_pose['yaw']:.1f}°\n"
            #     result_message += f"• Roll (Z): {head_pose['roll']:.1f}°\n\n"
            
            # Анализ движения головы отключен для ускорения
            # if head_movement_analysis:
            #     if not head_movement_analysis.get('is_natural', True):
            #         unnatural_movement = True
            #         is_fake = True
            #         result_message += "⚠️ ПОДОЗРИТЕЛЬНОЕ ДВИЖЕНИЕ!\n"
            #         result_message += f"• Статичных кадров подряд: {head_movement_analysis.get('static_frames_count', 0)}\n"
            #         result_message += f"• Движение обнаружено: {'Да' if head_movement_analysis.get('has_movement') else 'Нет'}\n"
            #         result_message += "• Голова не двигается естественно - возможно статичное изображение на экране!\n\n"
            #     else:
            #         result_message += "✅ Движение головы выглядит естественным\n\n"
            # else:
            #     result_message += "💡 Отправьте еще несколько фото для проверки движения головы\n\n"
        else:
            result_message += "✅ Фейк не обнаружен. Лицо не находится внутри монитора.\n"
        
        # Отправляем обработанное изображение
        with open(detected_image_path, 'rb') as photo_file:
            sent_message = await update.message.reply_photo(
                photo=photo_file,
                caption=result_message
            )
        
        # Сохраняем сессию пользователя
        user_sessions[user_id] = {
            'image_path': temp_image_path,
            'detected_image_path': detected_image_path,
            'analysis_result': analysis_result,
            'head_movement_analysis': head_movement_analysis,
            'message_id': sent_message.message_id,
            'is_fake': is_fake,
            'unnatural_movement': unnatural_movement
        }
        
        # Создаем кнопки для подтверждения/опровержения результата
        keyboard = [
            [
                InlineKeyboardButton("✅ Правильно", callback_data=f"correct_{user_id}"),
                InlineKeyboardButton("❌ Неправильно", callback_data=f"incorrect_{user_id}")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "❓ Правильно ли я определил?",
            reply_markup=reply_markup
        )
    
    except Exception as e:
        logger.error(f"Ошибка при обработке фото: {e}")
        await update.message.reply_text(
            f"❌ Произошла ошибка при обработке фото: {str(e)}"
        )


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик callback кнопок"""
    query = update.callback_query
    await query.answer()
    
    callback_data = query.data
    user_id = update.effective_user.id
    
    # Обработка информации о Face ID
    if callback_data.startswith("faceid_info_"):
        await query.edit_message_text(
            "🔐 Face ID Verification\n\n"
            "⚠️ Мини-приложение не настроено!\n\n"
            "📋 Для настройки:\n"
            "1. Запустите: setup_faceid.bat\n"
            "2. Или разместите faceid_app.html на GitHub Pages\n"
            "3. Установите FACEID_WEBAPP_URL в config.py\n"
            "4. Перезапустите бота\n\n"
            "📖 Подробнее: README_FACEID.md"
        )
        return
    
    # Обработка выбора метода детекции
    if callback_data.startswith("method_"):
        if callback_data.startswith("method_standard_"):
            user_detection_method[user_id] = 'standard'
            await query.edit_message_text(
                "✅ Выбран стандартный метод детекции\n\n"
                "📋 Используется:\n"
                "• Детекция мониторов\n"
                "• Детекция лиц\n"
                "• Проверка размера лица\n\n"
                "Отправьте фото для анализа."
            )
            return
        elif callback_data.startswith("method_deeppixbis_"):
            user_detection_method[user_id] = 'deeppixbis'
            await query.edit_message_text(
                "✅ Выбран метод DeepPixBis\n\n"
                "📋 Используется:\n"
                "• Пиксельная детекция артефактов\n"
                "• Глубокое обучение на уровне пикселей\n"
                "• Более точная детекция фейков\n\n"
                "⚠️ Требуется предобученная модель DeepPixBis\n\n"
                "Отправьте фото для анализа."
            )
            return
    
    if callback_data.startswith("correct_"):
        # Пользователь подтвердил правильность - сохраняем для обучения
        if user_id in user_sessions:
            session = user_sessions[user_id]
            analysis_result = session.get('analysis_result', {})
            
            # Сохраняем фото с аннотациями для обучения (подтвержденные результаты)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            retrain_image_path = os.path.join(RETRAIN_IMAGES_DIR, f"{user_id}_{timestamp}_confirmed.jpg")
            
            # Копируем изображение
            shutil.copy(session['image_path'], retrain_image_path)
            
            # Создаем аннотации
            img = cv2.imread(retrain_image_path)
            img_height, img_width = img.shape[:2]
            
            annotation_path = os.path.join(RETRAIN_LABELS_DIR, f"{user_id}_{timestamp}_confirmed.txt")
            with open(annotation_path, 'w') as f:
                for monitor in analysis_result.get('monitors', []):
                    x1, y1, x2, y2 = monitor['bbox']
                    center_x = ((x1 + x2) / 2) / img_width
                    center_y = ((y1 + y2) / 2) / img_height
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height
                    f.write(f"0 {center_x} {center_y} {width} {height}\n")
                
                for face in analysis_result.get('faces', []):
                    x1, y1, x2, y2 = face['bbox']
                    center_x = ((x1 + x2) / 2) / img_width
                    center_y = ((y1 + y2) / 2) / img_height
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height
                    f.write(f"1 {center_x} {center_y} {width} {height}\n")
            
            # Сохраняем метаданные о позе головы
            metadata_path = os.path.join(RETRAIN_METADATA_DIR, f"{user_id}_{timestamp}_confirmed.json")
            metadata = {
                'image_path': retrain_image_path,
                'annotation_path': annotation_path,
                'is_fake': session.get('is_fake', False),
                'unnatural_movement': session.get('unnatural_movement', False),
                'head_pose': analysis_result.get('head_pose'),
                'head_movement_analysis': session.get('head_movement_analysis'),
                'monitors_count': len(analysis_result.get('monitors', [])),
                'faces_count': len(analysis_result.get('faces', [])),
                'face_in_monitor_count': len(analysis_result.get('face_in_monitor', [])),
                'confirmed_by_user': True,
                'timestamp': timestamp
            }
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            await query.edit_message_text(
                "✅ Спасибо за подтверждение!\n"
                "💾 Данные сохранены для обучения модели (включая информацию о позе головы)."
            )
            
            # Очищаем временные файлы
            try:
                if os.path.exists(session['image_path']):
                    os.remove(session['image_path'])
                if os.path.exists(session['detected_image_path']):
                    os.remove(session['detected_image_path'])
            except:
                pass
            del user_sessions[user_id]
        else:
            await query.edit_message_text("✅ Спасибо за подтверждение!")
    
    elif callback_data.startswith("incorrect_"):
        # Пользователь указал что результат неправильный
        if user_id not in user_sessions:
            await query.edit_message_text("❌ Сессия не найдена. Пожалуйста, отправьте фото заново.")
            return
        
        session = user_sessions[user_id]
        analysis_result = session['analysis_result']
        
        # Сохраняем фото для переобучения
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        retrain_image_path = os.path.join(RETRAIN_IMAGES_DIR, f"{user_id}_{timestamp}.jpg")
        
        # Копируем изображение в папку для переобучения
        shutil.copy(session['image_path'], retrain_image_path)
        
        # Создаем файл аннотации в формате YOLO
        # Формат: class_id center_x center_y width height (нормализованные координаты)
        img = cv2.imread(retrain_image_path)
        img_height, img_width = img.shape[:2]
        
        annotation_path = os.path.join(RETRAIN_LABELS_DIR, f"{user_id}_{timestamp}.txt")
        with open(annotation_path, 'w') as f:
            # Записываем мониторы (класс 0)
            for monitor in analysis_result['monitors']:
                x1, y1, x2, y2 = monitor['bbox']
                center_x = ((x1 + x2) / 2) / img_width
                center_y = ((y1 + y2) / 2) / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height
                f.write(f"0 {center_x} {center_y} {width} {height}\n")
            
            # Записываем лица (класс 1)
            for face in analysis_result['faces']:
                x1, y1, x2, y2 = face['bbox']
                center_x = ((x1 + x2) / 2) / img_width
                center_y = ((y1 + y2) / 2) / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height
                f.write(f"1 {center_x} {center_y} {width} {height}\n")
        
        # Сохраняем метаданные о позе головы
        metadata_path = os.path.join(RETRAIN_METADATA_DIR, f"{user_id}_{timestamp}.json")
        metadata = {
            'image_path': retrain_image_path,
            'annotation_path': annotation_path,
            'is_fake': session.get('is_fake', False),
            'unnatural_movement': session.get('unnatural_movement', False),
            'head_pose': analysis_result.get('head_pose'),
            'head_movement_analysis': session.get('head_movement_analysis'),
            'monitors_count': len(analysis_result.get('monitors', [])),
            'faces_count': len(analysis_result.get('faces', [])),
            'face_in_monitor_count': len(analysis_result.get('face_in_monitor', [])),
            'confirmed_by_user': False,  # Пользователь сказал что неправильно
            'timestamp': timestamp
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        await query.edit_message_text(
            f"💾 Фото сохранено для переобучения модели!\n\n"
            f"📁 Изображение: {retrain_image_path}\n"
            f"📄 Аннотация: {annotation_path}\n"
            f"📋 Метаданные: {metadata_path}\n\n"
            f"Спасибо за помощь в улучшении модели!"
        )
        
        # Очищаем временные файлы
        try:
            if os.path.exists(session['detected_image_path']):
                os.remove(session['detected_image_path'])
        except:
            pass
        
        del user_sessions[user_id]


def main():
    """Главная функция запуска бота"""
    if not TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN не установлен! Создайте файл .env и укажите токен.")
        return
    
    # Создаем приложение
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Обработчик видео
    async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик получения видео"""
        user_id = update.effective_user.id
        
        try:
            video = update.message.video
            if not video:
                await update.message.reply_text("❌ Не удалось получить видео")
                return
            
            file = await context.bot.get_file(video.file_id)
            
            # Сохраняем временный файл
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_video_path = os.path.join(TEMP_VIDEOS_DIR, f"{user_id}_{timestamp}.mp4")
            await file.download_to_drive(temp_video_path)
            
            await update.message.reply_text("🔍 Анализирую видео... Это может занять некоторое время.")
            
            # Анализируем видео
            video_analysis = detection_engine.analyze_video(temp_video_path, sample_rate=5)
            
            if 'error' in video_analysis:
                await update.message.reply_text(f"❌ Ошибка: {video_analysis['error']}")
                return
            
            # Формируем сообщение с результатами
            result_message = "📹 Результаты анализа видео:\n\n"
            result_message += f"📊 Всего кадров: {video_analysis.get('total_frames', 0)}\n"
            result_message += f"🔍 Проанализировано: {video_analysis.get('analyzed_frames', 0)}\n"
            result_message += f"🖥️ FPS: {video_analysis.get('fps', 0):.1f}\n\n"
            
            if video_analysis.get('is_fake'):
                result_message += "🚨 ВНИМАНИЕ! Обнаружен ФЕЙК!\n\n"
                
                if video_analysis.get('unnatural_movement'):
                    result_message += "⚠️ ПОДОЗРИТЕЛЬНОЕ ДВИЖЕНИЕ ГОЛОВЫ!\n"
                    movement_info = video_analysis.get('head_movement_analysis', {})
                    result_message += f"• Статичных кадров: {movement_info.get('static_frames_count', 0)}\n"
                    result_message += f"• Движение обнаружено: {'Да' if movement_info.get('has_movement') else 'Нет'}\n"
                    result_message += "• Голова не двигается естественно!\n\n"
                else:
                    result_message += "• Лицо обнаружено внутри монитора\n\n"
            else:
                result_message += "✅ Фейк не обнаружен\n"
            
            await update.message.reply_text(result_message)
            
            # Удаляем временный файл
            try:
                os.remove(temp_video_path)
            except:
                pass
        
        except Exception as e:
            logger.error(f"Ошибка при обработке видео: {e}")
            await update.message.reply_text(f"❌ Ошибка при обработке видео: {str(e)}")
    
    # Регистрируем обработчики
    application.add_handler(CommandHandler("start", start))
    # application.add_handler(CommandHandler("webcam", webcam_info))  # Отключено
    async def handle_webapp_data(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик данных от мини-приложения Face ID"""
        user_id = update.effective_user.id
        
        try:
            # Получаем данные от мини-приложения
            if not update.message or not update.message.web_app_data:
                await update.message.reply_text("❌ Данные от мини-приложения не получены")
                return
                
            web_app_data = update.message.web_app_data.data
            
            if not web_app_data:
                await update.message.reply_text("❌ Данные от мини-приложения пусты")
                return
            
            # Парсим JSON данные
            try:
                data = json.loads(web_app_data)
            except json.JSONDecodeError as e:
                logger.error(f"Ошибка парсинга JSON: {e}, данные: {web_app_data[:100]}")
                await update.message.reply_text("❌ Ошибка формата данных от мини-приложения")
                return
            
            if data.get('type') == 'faceid_video':
                await update.message.reply_text("🔍 Получено видео от Face ID. Анализирую...")
                
                # Декодируем base64 видео
                import base64
                video_base64 = data.get('video')
                if not video_base64:
                    await update.message.reply_text("❌ Видео не получено")
                    return
                
                # Сохраняем временный файл
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                temp_video_path = os.path.join(TEMP_VIDEOS_DIR, f"faceid_{user_id}_{timestamp}.webm")
                
                try:
                    # Декодируем и сохраняем
                    video_bytes = base64.b64decode(video_base64, validate=True)
                    if len(video_bytes) == 0:
                        await update.message.reply_text("❌ Получено пустое видео")
                        return
                    
                    with open(temp_video_path, 'wb') as f:
                        f.write(video_bytes)
                    logger.info(f"Видео сохранено: {temp_video_path}, размер: {len(video_bytes)} байт")
                except Exception as e:
                    logger.error(f"Ошибка декодирования base64: {e}")
                    await update.message.reply_text(f"❌ Ошибка декодирования видео: {str(e)}")
                    return
                
                # Конвертируем webm в mp4 если нужно (для лучшей совместимости)
                try:
                    import subprocess
                    mp4_path = temp_video_path.replace('.webm', '.mp4')
                    # Используем ffmpeg для конвертации (если установлен)
                    subprocess.run([
                        'ffmpeg', '-i', temp_video_path, '-c', 'copy', mp4_path, '-y'
                    ], check=False, capture_output=True)
                    if os.path.exists(mp4_path):
                        os.remove(temp_video_path)
                        temp_video_path = mp4_path
                except:
                    # Если ffmpeg не установлен, используем webm как есть
                    pass
                
                # Анализируем видео
                video_analysis = detection_engine.analyze_video(temp_video_path, sample_rate=2)
                
                if 'error' in video_analysis:
                    await update.message.reply_text(f"❌ Ошибка анализа: {video_analysis['error']}")
                    return
                
                # Формируем сообщение с результатами
                result_message = "🔐 Результаты Face ID проверки:\n\n"
                result_message += f"📹 Видео: 1 секунда\n"
                result_message += f"📊 Проанализировано кадров: {video_analysis.get('analyzed_frames', 0)}\n\n"
                
                if video_analysis.get('is_fake'):
                    result_message += "🚨 ВНИМАНИЕ! Обнаружен ФЕЙК!\n\n"
                    
                    if video_analysis.get('unnatural_movement'):
                        result_message += "⚠️ ПОДОЗРИТЕЛЬНОЕ ДВИЖЕНИЕ ГОЛОВЫ!\n"
                        movement_info = video_analysis.get('head_movement_analysis', {})
                        result_message += f"• Статичных кадров: {movement_info.get('static_frames_count', 0)}\n"
                        result_message += f"• Движение обнаружено: {'Да' if movement_info.get('has_movement') else 'Нет'}\n"
                        result_message += "• Голова не двигается естественно - возможно статичное изображение!\n\n"
                    else:
                        result_message += "• Лицо обнаружено внутри монитора\n"
                        result_message += "• Или обнаружено подозрительно большое лицо\n\n"
                else:
                    result_message += "✅ Фейк не обнаружен\n"
                    result_message += "✅ Лицо выглядит реальным\n"
                
                await update.message.reply_text(result_message)
                
                # Удаляем временный файл
                try:
                    if os.path.exists(temp_video_path):
                        os.remove(temp_video_path)
                except:
                    pass
            else:
                await update.message.reply_text("❌ Неизвестный тип данных от мини-приложения")
                
        except json.JSONDecodeError as e:
            logger.error(f"Ошибка декодирования JSON от мини-приложения: {e}")
            logger.error(f"Данные: {web_app_data[:200] if web_app_data else 'None'}")
            await update.message.reply_text("❌ Ошибка формата данных от мини-приложения")
        except Exception as e:
            logger.error(f"Ошибка при обработке данных от мини-приложения: {e}", exc_info=True)
            error_msg = str(e)
            if len(error_msg) > 200:
                error_msg = error_msg[:200] + "..."
            await update.message.reply_text(f"❌ Ошибка при обработке: {error_msg}")
    
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.VIDEO, handle_video))
    application.add_handler(MessageHandler(filters.StatusUpdate.WEB_APP_DATA, handle_webapp_data))
    application.add_handler(CallbackQueryHandler(handle_callback))
    
    # Сначала удаляем webhook если он есть
    bot = application.bot
    try:
        webhook_info = bot.get_webhook_info()
        if webhook_info.url:
            logger.info(f"Найден webhook: {webhook_info.url}, удаляем...")
            bot.delete_webhook(drop_pending_updates=True)
            logger.info("Webhook удален, ждем 2 секунды...")
            import time
            time.sleep(2)  # Даем время API обработать удаление
    except Exception as e:
        logger.warning(f"Ошибка при проверке webhook: {e}")
        # Все равно пытаемся удалить
        try:
            bot.delete_webhook(drop_pending_updates=True)
            import time
            time.sleep(2)
        except:
            pass
    
    # Запускаем бота
    logger.info("Бот запущен...")
    
    try:
        application.run_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True,  # Игнорируем старые обновления
            close_loop=False
        )
    except Conflict as e:
        logger.error(f"Конфликт: другой экземпляр бота запущен или активен webhook. {e}")
        logger.info("Попробуйте:")
        logger.info("1. Остановить все другие экземпляры бота")
        logger.info("2. Запустить: py -3.10 delete_webhook.py")
        logger.info("3. Подождать 5 секунд и запустить бота снова")
        # Пытаемся удалить webhook еще раз
        try:
            bot.delete_webhook(drop_pending_updates=True)
            logger.info("Webhook удален, перезапустите бота через 5 секунд")
        except:
            pass
    except KeyboardInterrupt:
        logger.info("Бот остановлен пользователем")
    except Exception as e:
        logger.error(f"Ошибка при работе бота: {e}")
        raise


if __name__ == '__main__':
    main()
