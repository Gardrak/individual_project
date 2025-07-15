from libraries import *

PROJECT_PATH = 'D:/programming/project/'

DATASET_PATH = f'{PROJECT_PATH}/dataset'
OCR_MODEL_PATH = f'{PROJECT_PATH}/models/model-7-0.9156.ckpt'
ALPHABET = '0123456789АВЕКМНОРСТУХ'


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


OOV_TOKEN = '<OOV>'
CTC_BLANK = '<BLANK>'


TRAIN_SIZE = 0.9
BATCH_SIZE_OCR = 16


TRAIN_LOSSES = []
VAL_ACCURACIES = []

# Тестовые изображения
IMAGE_PATHS = [
        f'{PROJECT_PATH}/CarImages/H757YC37.jpg',
        f'{PROJECT_PATH}/CarImages/M611OC32, E611CO32.png',
        f'{PROJECT_PATH}/CarImages/X010XX71, A010AA71.jpg',
        f'{PROJECT_PATH}/CarImages/Y003KK190, X248YM150, A082MP97.jpg',
        f'{PROJECT_PATH}/CarImages/C355CC35.jpg',
        f'{PROJECT_PATH}/CarImages/P600PO59.jpg'
    ]
MODEL_PATH = f'{PROJECT_PATH}/models/model-8-0.9971.ckpt'
CASCADE_PATH = f'{PROJECT_PATH}/haarcascade_russian_plate_number.xml'

