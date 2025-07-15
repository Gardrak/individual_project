from config import *


def decode(pred: torch.Tensor, alphabet: str) -> list[str]:
    """
    Декодирует выходы модели в строки, используя заданный алфавит.
    
    Args:
        pred (torch.Tensor): Выходной тензор модели размером (L, N, C)
        alphabet (str): Строка с допустимыми символами
    
    Returns:
        List[str]: Список декодированных строк
    """
    pred = pred.permute(1, 0, 2).cpu().data.numpy()
    outputs = []
    for i in range(len(pred)):
        outputs.append(pred_to_string(pred[i], alphabet))
    return outputs


def pred_to_string(pred: np.ndarray, alphabet: str) -> str:
    """
    Преобразует предсказания модели для одного элемента в строку.
    
    Args:
        pred (np.ndarray): Массив предсказаний размером (L, C)
        alphabet (str): Строка с допустимыми символами
    
    Returns:
        str: Декодированная строка
    """
    seq = []
    for i in range(len(pred)):
        label = np.argmax(pred[i])
        seq.append(label - 1)
    out = []
    for i in range(len(seq)):
        if len(out) == 0:
            if seq[i] != -1:
                out.append(seq[i])
        else:
            if seq[i] != -1 and seq[i] != seq[i - 1]:
                out.append(seq[i])
    out = ''.join([alphabet[c] for c in out])
    return out


def load_json(file: str) -> dict:
    """
    Загружает данные из JSON-файла.
    
    Args:
        file (str): Путь к JSON-файлу
    
    Returns:
        dict: Загруженные данные
    """
    with open(file, 'r') as f:
        return json.load(f)



# Чтобы без проблем реализовывать json. Без него есть нюансы
class npEncoder(json.JSONEncoder):
    """
    Кастомный JSON-энкодер для обработки numpy-типов.
    Наследуется от json.JSONEncoder.
    """
    def default(self, obj: any) -> any:
        if isinstance(obj, np.int32):
            return int(obj)
        return json.JSONEncoder.default(self, obj)




def get_annot(file: str, dir: str) -> str:
    """
    Получает аннотацию для указанного файла изображения.
    
    Args:
        file (str): Имя файла изображения
        dir (str): Директория с аннотациями (train/val)
    
    Returns:
        str: Текстовая аннотация
    """
    return load_json(DATASET_PATH + f'/{dir}/ann/' + file[:-3] + 'json')['description']




class OCRDataset(Dataset):
    """
    Датасет для OCR-задачи.
    
    Args:
        marks (list): Список аннотаций
        img_folder (str): Папка с изображениями
        tokenizer (Tokenizer): Токенизатор текста
        transforms (callable, optional): Трансформации изображений
    """
    def __init__(self, 
                 marks:      list[dict[str, str]], 
                 img_folder: str, 
                 tokenizer:  'Tokenizer', 
                 transforms: torch.optim[callable] = None):
        self.img_paths = []
        self.texts = []
        for item in marks:
            self.img_paths.append(os.path.join(DATASET_PATH + f'/{img_folder}/img/', item['filename']))
            self.texts.append(item['label'])

        self.enc_texts = tokenizer.encode(self.texts)
        self.img_folder = DATASET_PATH + f'/{img_folder}/img/'
        self.transforms = transforms

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str, torch.Tensor]:
        img_path = self.img_paths[idx]
        #print(img_path)
        text = self.texts[idx]
        enc_text = torch.LongTensor(self.enc_texts[idx])
        image = cv2.imread(img_path)
        if self.transforms is not None:
            image = self.transforms(image)

        return image, text, enc_text

    def __len__(self) -> int:
        return len(self.texts)




class Resize(object):
    """
    Трансформация изменения размера изображения.
    
    Args:
        size (tuple, optional): Целевой размер (ширина, высота)
    """
    def __init__(self, size: tuple[int, int] = (250, 50)):
        self.size = size


    def __call__(self, img: np.ndarray) -> np.ndarray:
        w_from, h_from = img.shape[1], img.shape[0]
        w_to, h_to = self.size

        # Сделаем разную интерполяцию при увеличении и уменьшении
        # Если увеличиваем картинку, меняем интерполяцию
        interpolation = cv2.INTER_AREA
        if w_to > w_from:
            interpolation = cv2.INTER_CUBIC

        img = cv2.resize(img, dsize=self.size, interpolation=interpolation)
        return img




class Normalize:
    """Трансформация нормализации изображения."""
    def __call__(self, img: np.ndarray) -> np.ndarray:
        img = img.astype(np.float32) / 255
        return img




def collate_fn(batch: list[tuple]) -> tuple[torch.Tensor, list[str], torch.Tensor, torch.Tensor]:
    """
    Функция для объединения элементов батча.
    
    Args:
        batch: Список элементов вида (image, text, enc_text)
    
    Returns:
        Кортеж с:
        - Тензор изображений
        - Список текстов
        - Тензор закодированных текстов
        - Тензор длин текстов
    """
    images, texts, enc_texts = zip(*batch)
    images = torch.stack(images, 0)
    text_lens = torch.LongTensor([len(text) for text in texts])
    enc_pad_texts = pad_sequence(enc_texts, batch_first=True, padding_value=0)
    return images, texts, enc_pad_texts, text_lens


def get_char_map(alphabet: str) -> dict[str, int]:
    """
    Создает словарь для преобразования символов в индексы.
    
    Args:
        alphabet (str): Строка с допустимыми символами
    
    Returns:
        Словарь {символ: индекс}
    """
    char_map = {value: idx + 2 for (idx, value) in enumerate(alphabet)}
    char_map[CTC_BLANK] = 0
    char_map[OOV_TOKEN] = 1
    return char_map




class Tokenizer:
    """
    Токенизатор для преобразования текста в индексы и обратно.
    
    Args:
        alphabet (str): Строка с допустимыми символами
    """
    def __init__(self, alphabet: str):
        self.char_map = get_char_map(alphabet)
        self.rev_char_map = {val: key for key, val in self.char_map.items()}


    def encode(self, word_list: list[str]) -> list[list[int]]:
        """
        Кодирует список строк в списки индексов.
        """
        enc_words = []
        for word in word_list:
            enc_words.append(
                [self.char_map[char] if char in self.char_map
                 else self.char_map[OOV_TOKEN]
                 for char in word]
            )
        return enc_words


    def get_num_chars(self) -> int:
        """
        Возвращает количество уникальных символов.
        """
        return len(self.char_map)


    def decode(self, enc_word_list: list[list[int]]) -> list[str]:
        """
        Декодирует списки индексов обратно в строки.
        """
        dec_words = []
        for word in enc_word_list:
            word_chars = ''
            for idx, char_enc in enumerate(word):
                # пропускаем пустые символы, токены oov или повторяющиеся символы
                if (
                    char_enc != self.char_map[OOV_TOKEN]
                    and char_enc != self.char_map[CTC_BLANK]
                    # idx > 0 чтобы избежать выбора элемента [-1]
                    and not (idx > 0 and char_enc == word[idx - 1])
                ):
                    word_chars += self.rev_char_map[char_enc]
            dec_words.append(word_chars)
        return dec_words
gc.collect()




# Переменные, которые не получилось засунуть в `config.py`
TRAIN_LABELS = pd.DataFrame([[f, get_annot(f, 'train')] for f in os.listdir(DATASET_PATH + '/train/img')], columns=['filename', 'label']).to_dict('records')
VAL_LABELS = pd.DataFrame([[f, get_annot(f, 'val')] for f in os.listdir(DATASET_PATH + '/val/img')], columns=['filename', 'label']).to_dict('records')
TOKENIZER = Tokenizer(ALPHABET)
OCR_TRANSFORMS = transforms.Compose([
    Resize(size=(250, 50)),
    Normalize(),
    transforms.ToTensor()])

VAL_OCR_DATASET = OCRDataset(
    marks=VAL_LABELS,
    img_folder='val',
    tokenizer=TOKENIZER,
    transforms=OCR_TRANSFORMS)

TRAIN_OCR_DATASET = OCRDataset(
    marks=TRAIN_LABELS,
    img_folder='train',
    tokenizer=TOKENIZER,
    transforms=OCR_TRANSFORMS)

TRAIN_LOADER = DataLoader(
    TRAIN_OCR_DATASET,
    batch_size=BATCH_SIZE_OCR,
    drop_last=True,
    num_workers=0, 
    collate_fn=collate_fn,
    timeout=0,
    shuffle=True)

VAL_LOADER = DataLoader(
    VAL_OCR_DATASET,
    batch_size=BATCH_SIZE_OCR,
    drop_last=False,
    num_workers=0, 
    collate_fn=collate_fn,
    timeout=0,)





img_tensor = TRAIN_OCR_DATASET[0][0]  # Это тензор изображения
plt.imshow(img_tensor.permute(1, 2, 0))  # Меняем порядок на H x W x C
plt.title(TRAIN_OCR_DATASET[0][1])  # Заголовок с номером
plt.axis('off')  # Отключаем оси
plt.show()

def get_resnet34_backbone():
    m = models.resnet34(pretrained=True)
    input_conv = nn.Conv2d(3, 64, 7, 1, 3)
    blocks = [input_conv, m.bn1, m.relu,
              m.maxpool, m.layer1, m.layer2, m.layer3]
    return nn.Sequential(*blocks)


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            dropout=dropout, batch_first=True, bidirectional=True)


    def forward(self, x):
        out, _ = self.lstm(x)
        return out


class CRNN(nn.Module):
    def __init__(
        self, number_class_symbols, time_feature_count=256, lstm_hidden=256,
        lstm_len=2,):

        super().__init__()
        self.feature_extractor = get_resnet34_backbone()
        self.avg_pool = nn.AdaptiveAvgPool2d(
            (time_feature_count, time_feature_count))
        self.bilstm = BiLSTM(time_feature_count, lstm_hidden, lstm_len)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, time_feature_count),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(time_feature_count, number_class_symbols))


    def forward(self, x):
        x = self.feature_extractor(x)
        b, c, h, w = x.size()
        x = x.view(b, c * h, w)
        x = self.avg_pool(x)
        x = x.transpose(1, 2)
        x = self.bilstm(x)
        x = self.classifier(x)
        x = nn.functional.log_softmax(x, dim=2).permute(1, 0, 2)
        return x




model = CRNN(number_class_symbols=TOKENIZER.get_num_chars())
model.to(DEVICE)
criterion = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001,
                              weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer, mode='max', factor=0.5, patience=15)




class AverageMeter:
    """Вычисляет и хранит среднее значение и текущее значение."""
    def __init__(self):
        self.reset()


    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0


    def update(self, val: float, n: int = 1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




def get_accuracy(y_true: list[str], y_pred: list[str]) -> float:
    """
    Вычисляет точность предсказаний.
    
    Args:
        y_true: Список истинных значений
        y_pred: Список предсказанных значений
    
    Returns:
        Средняя точность
    """
    scores = []
    for true, pred in zip(y_true, y_pred):
        scores.append(true == pred)
    avg_score = np.mean(scores)
    return avg_score


def predict(images: torch.Tensor, 
           model: nn.Module, 
           tokenizer: Tokenizer, 
           device: torch.device) -> list[str]:
    """
    Предсказывает текст для батча изображений.
    
    Args:
        images: Тензор изображений
        model: Модель для распознавания
        tokenizer: Токенизатор текста
        device: Устройство для вычислений
    
    Returns:
        Список распознанных текстов
    """
    model.eval()
    images = images.to(device)
    with torch.no_grad():
        output = model(images)
    pred = torch.argmax(output.detach().cpu(), -1).permute(1, 0).numpy()
    text_preds = tokenizer.decode(pred)
    return text_preds


def train_loop(data_loader: DataLoader, 
               model: nn.Module, 
               criterion: nn.Module, 
               optimizer: torch.optim, 
               epoch: int) -> float:
    """
    Один эпох обучения модели.
    
    Args:
        data_loader: Загрузчик данных
        model: Модель для обучения
        criterion: Функция потерь
        optimizer: Оптимизатор
        epoch: Номер текущей эпохи
    
    Returns:
        Среднее значение потерь за эпоху
    """
    loss_avg = AverageMeter()
    model.train()
    # Устанавливаем tqdm для отображения прогресса
    with tqdm(total=len(data_loader), desc=f'Training Epoch {epoch+1}', leave=True) as pbar:
        for images, texts, enc_pad_texts, text_lens in data_loader:
            model.zero_grad()
            images = images.to(DEVICE)
            batch_size = len(texts)
            output = model(images)
            output_lengths = torch.full(
                size=(output.size(1),),
                fill_value=output.size(0),
                dtype=torch.long
            )
            loss = criterion(output, enc_pad_texts, output_lengths, text_lens)
            loss_avg.update(loss.item(), batch_size)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()

            # Обновляем прогресс
            pbar.set_postfix(loss=loss_avg.avg)
            pbar.update(1)

    print(f'Epoch {epoch+1}, Loss: {loss_avg.avg:.5f}')
    return loss_avg.avg


def val_loop(data_loader: DataLoader, 
             model: nn.Module, 
             tokenizer: Tokenizer, 
             device: torch.device) -> float:
    """
    Валидация модели.
    
    Args:
        data_loader: Загрузчик данных
        model: Модель для валидации
        tokenizer: Токенизатор текста
        device: Устройство для вычислений
    
    Returns:
        Средняя точность на валидации
    """
    acc_avg = AverageMeter()
    # Устанавливаем tqdm для отображения прогресса
    with tqdm(total=len(data_loader), desc='Validation', leave=True) as pbar:
        for images, texts, _, _ in data_loader:
            batch_size = len(texts)
            text_preds = predict(images, model, tokenizer, device)
            acc_avg.update(get_accuracy(texts, text_preds), batch_size)

            # Обновляем прогресс
            pbar.update(1)

    print(f'Validation, acc: {acc_avg.avg:.4f}\n')
    return acc_avg.avg


def plot_metrics():
    """Строит графики потерь и точности."""
    # График потерь
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(TRAIN_LOSSES, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()

    # График точности
    plt.subplot(1, 2, 2)
    plt.plot(VAL_ACCURACIES, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()

    plt.show()

# Тренировка модели | была закоментирована из-за ненадобности
'''def train(model, optimizer, scheduler, TRAIN_LOADER, VAL_LOADER, epochs):
    best_acc = -np.inf
    os.makedirs('models', exist_ok=True)
    acc_avg = val_loop(VAL_LOADER, model, tokenizer, device)

    # Добавляем первые метрики
    VAL_ACCURACIES.append(acc_avg)

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1} started')

        # Обучение
        loss_avg = train_loop(TRAIN_LOADER, model, criterion, optimizer, epoch)
        TRAIN_LOSSES.append(loss_avg)  # Добавляем значение потерь

        # Валидация
        acc_avg = val_loop(VAL_LOADER, model, tokenizer, device)
        VAL_ACCURACIES.append(acc_avg)  # Добавляем точность

        # Обновляем learning rate
        scheduler.step(acc_avg)

        if acc_avg > best_acc:
            best_acc = acc_avg
            model_save_path = os.path.join(
                'models', f'model-{epoch}-{acc_avg:.4f}.ckpt')
            torch.save(model.state_dict(), model_save_path)
            print('Model weights saved')

    # После завершения обучения рисуем графики
    plot_metrics()

train(model, optimizer, scheduler, TRAIN_LOADER, VAL_LOADER, epochs=10)'''




class InferenceTransform:
    """
    Трансформация для инференса. Обрабатывает список изображений.
    
    Args:
        transforms: Композиция трансформаций
    """
    def __init__(self, transforms: callable):
        self.transforms = transforms

    def __call__(self, images: list[np.ndarray]) -> torch.Tensor:
        transformed_images = []
        for image in images:
            # Применяем преобразования
            transformed_image = self.transforms(image)
            transformed_images.append(transformed_image)

        # Стек всех преобразованных изображений в один тензор
        transformed_tensor = torch.stack(transformed_images)
        return transformed_tensor




class Predictor:
    """
    Класс для инференса модели OCR.
    
    Args:
        model_path: Путь к весам модели
        device: Устройство для вычислений (None для автоопределения)
    """
    def __init__(self, model_path: str, device: torch.optim[str] = None):  # Изменяем параметр device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = Tokenizer(ALPHABET)
        self.device = torch.device(device)

        self.model = CRNN(number_class_symbols=self.tokenizer.get_num_chars())
        # Добавляем map_location для корректной загрузки на CPU
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)

        self.transforms = InferenceTransform(OCR_TRANSFORMS)


    def __call__(self, images: list[np.ndarray, list[np.ndarray]]) -> list[str, list[str]]:
        """
        Предсказывает текст для изображения(ий).
        
        Args:
            images: Одно изображение или список изображений
        
        Returns:
            Распознанный текст (или список текстов)
        """
        if isinstance(images, np.ndarray):
            images = [images]  # Преобразуем в список, если это одно изображение
            single_image = True
        elif isinstance(images, (list, tuple)):
            single_image = False
        else:
            raise TypeError(f"Input must be np.ndarray, list, or tuple, found {type(images)}.")

        # Применяем трансформации
        images = self.transforms(images)
        images = images.to(self.device)  # Переносим на нужное устройство

        # Получаем предсказания
        predictions = self.predict(images)

        if single_image:
            return predictions[0]  # Возвращаем единственное предсказание
        return predictions


    def predict(self, images: torch.Tensor) -> list[str]:
        """
        Прямой проход модели.
        
        Args:
            images: Тензор изображений
        
        Returns:
            Список распознанных текстов
        """
        self.model.eval()  # Устанавливаем модель в режим оценки
        with torch.no_grad():
            output = self.model(images)  # Получаем предсказания модели

        pred = torch.argmax(output.detach().cpu(), dim=-1).permute(1, 0).numpy()
        text_predictions = self.tokenizer.decode(pred)  # Декодируем предсказания
        return text_predictions
    











