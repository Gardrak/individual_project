import numpy as np
import pandas as pd
import os
import cv2
import gc
import json
import glob
import re
import PIL
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns



from tqdm import tqdm
from collections import Counter
from matplotlib.path import Path
from torchvision import models
from torchvision import transforms
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchmetrics import CharErrorRate as CER
from IPython.display import Image as Img