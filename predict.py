from config import *
from OCR import Predictor


def load_json(file: str) -> dict[str, any]:
    """
    Загружает данные из JSON-файла.
    
    Args:
        file (str): Путь к JSON-файлу
    
    Returns:
        dict: Загруженные данные
    """
    with open(file, 'r') as f:
        return json.load(f)


def open_img(img_path: str) -> np.ndarray:
    """
    Загружает и конвертирует изображение.
    
    Args:
        img_path (str): Путь к изображению
    
    Returns:
        np.ndarray: Изображение в формате RGB
    
    Raises:
        ValueError: Если изображение не удалось загрузить
    """
    carplate_img = cv2.imread(img_path)
    if carplate_img is None:
        raise ValueError(f"Не удалось загрузить изображение: {img_path}")
    carplate_img = cv2.cvtColor(carplate_img, cv2.COLOR_BGR2RGB)
    return carplate_img


def carplate_extract(image: np.ndarray, 
                     carplate_haar_cascade: cv2.CascadeClassifier) -> list[np.ndarray]:
    """
    Обнаруживает и извлекает номерные знаки на изображении.
    
    Args:
        image (np.ndarray): Входное изображение
        carplate_haar_cascade (cv2.CascadeClassifier): Каскадный классификатор
    
    Returns:
        List[np.ndarray]: Список обнаруженных номерных знаков
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Преобразование в серый цвет для работы каскада
    carplate_rects = carplate_haar_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

    carplates = []  # Список для хранения найденных номерных знаков

    for x, y, w, h in carplate_rects:
        carplate_img = image[y:y + h, x:x + w]
        carplates.append(carplate_img)  # Добавляем извлечённый номерной знак в список

    return carplates  # Возвращаем список всех найденных номерных знаков


def enlarge_img(image: np.ndarray, 
                scale_percent: float) -> np.ndarray:
    """
    Увеличивает изображение на указанный процент.
    
    Args:
        image (np.ndarray): Входное изображение
        scale_percent (float): Процент увеличения (например, 150 для 150%)
    
    Returns:
        np.ndarray: Увеличенное изображение
    """
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized_image


def recognize_license_plate(plate_img: np.ndarray, 
                            model_path: str) -> str:
    """
    Распознает текст на изображении номерного знака.
    
    Args:
        plate_img (np.ndarray): Изображение номерного знака
        model_path (str): Путь к файлу весов модели OCR
    
    Returns:
        str: Распознанный текст номера
    """
    # Преобразование изображения номерного знака в формат, подходящий для модели
    gray_plate_img = cv2.cvtColor(plate_img, cv2.COLOR_RGB2GRAY)
    rgb_plate_img = cv2.cvtColor(gray_plate_img, cv2.COLOR_GRAY2RGB)

    # Инициализация и использование предсказателя
    predictor = Predictor(model_path=model_path)
    prediction = predictor(rgb_plate_img)
    return prediction


def detect_and_recognize_plate(image_path: str, 
                               model_path: str, 
                               cascade_path: str) -> None:
    """
    Полный пайплайн: обнаружение и распознавание номерных знаков.
    
    Args:
        image_path (str): Путь к изображению автомобиля
        model_path (str): Путь к файлу весов модели OCR
        cascade_path (str): Путь к файлу каскада Хаара
    
    Returns:
        None: Результаты выводятся на экран
    """
    # Открываем изображение
    carplate_img_rgb = open_img(image_path)

    # Инициализируем каскад Хаара для обнаружения номерного знака
    carplate_haar_cascade = cv2.CascadeClassifier(cascade_path)

    # Извлечение номерных знаков
    carplates = carplate_extract(carplate_img_rgb, carplate_haar_cascade)

    if not carplates:
        print(f"Номерные знаки не найдены для {image_path}.")
        return

    image_name = os.path.basename(image_path)

    # Распознаем текст на всех извлечённых номерных знаках
    for idx, carplate_extract_img in enumerate(carplates):
        # Увеличиваем изображение для лучшего распознавания
        carplate_extract_img = enlarge_img(carplate_extract_img, 150)

        # Распознаем текст на извлечённом номерном знаке
        license_text = recognize_license_plate(carplate_extract_img, model_path)

        # Отображение результата
        plt.axis('off')
        plt.imshow(carplate_extract_img)
        print(f"{image_name}: Распознанный текст на фото: {license_text}")
        plt.title(f"Номерной знак {idx + 1}")
        plt.show()


# Пример использования
if __name__ == '__main__':

    for img_path in IMAGE_PATHS:
        detect_and_recognize_plate(img_path, MODEL_PATH, CASCADE_PATH)
