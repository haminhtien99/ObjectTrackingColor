# ObjectTrackingColor - Трекирование автомобилей и классификация по цвету

## Описание
Система для детекции, трекинга и классификации автомобилей по цвету с использованием YOLO и компьютерного зрения.

## Основной процесс
1. **Детекция** - обнаружение объектов с помощью YOLO
2. **Фильтрация** - отсев объектов за пределами зоны интереса
3. **Классификация цвета** - определение цвета в HSV-пространстве с использованием цветовых масок
4. **Ассоциация** - трекинг объектов между кадрами


## Установка и запуск
```bash
git clone https://github.com/haminhtien99/ObjectTrackingColor
cd ObjectTrackingColor/
git lfs pull

pip install -r requirements.txt
python main.py  # Изменить track_cfg.yaml для конфигурации трекинга
```
## Структура
```bash
ObjectTrackingColor/
├── README.md                          # Основное описание проекта
├── requirements.txt                   # Зависимости Python
├── ultralytics/                       # Модуль ultralytics для детекции
├── detectors/                         # YOLO детектор
├── trackers/                          # Модуль трекинга
├── custom_detector.py                 # Детектор кастом
├── color_classifier.py                # Классификация цвета
├── data/                              # Видео тест
├── output.avi                         # Видео трекинга
└── outputs/
    ├── name_video.txt                 # Результат трекинга в MOT-формате
    └── name_video/                    # рамка объекта
```