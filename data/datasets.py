import os
import cv2
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, ConcatDataset
from random import random, choice
from io import BytesIO
from PIL import Image, ImageFilter, ImageEnhance
from PIL import ImageFile
from scipy.ndimage import gaussian_filter
from torchvision.transforms import InterpolationMode

ImageFile.LOAD_TRUNCATED_IMAGES = True


def dataset_folder(opt, root):
    if opt.mode == 'binary':
        return binary_dataset(opt, root)
    if opt.mode == 'filename':
        return FileNameDataset(opt, root)
    raise ValueError('opt.mode needs to be binary or filename.')


def get_transform(opt):
    if opt.isTrain:
        rz_func = transforms.Resize((opt.loadSize, opt.loadSize))
        crop_func = transforms.RandomCrop(opt.cropSize)
        transform_list = [
            rz_func,
            crop_func,
            transforms.RandomHorizontalFlip(p=0.5),
            RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            HueSaturationValue(p=0.3),
            ImageCompression(quality_lower=40, quality_upper=100, p=0.1),
            GaussNoise(p=0.1),
            MotionBlur(p=0.1),
            CLAHE(p=0.1),
            ChannelShuffle(p=0.1),
            Cutout(p=0.1),
            RandomGamma(p=0.3),
            GlassBlur(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    else:
        if opt.no_crop:
            crop_func = transforms.Lambda(lambda img: img)
        else:
            crop_func = transforms.CenterCrop(opt.cropSize)
        if opt.no_resize:
            rz_func = transforms.Lambda(lambda img: img)
        else:
            rz_func = transforms.Resize((opt.loadSize, opt.loadSize))
        transform_list = [rz_func, crop_func, transforms.ToTensor(),
                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    return transforms.Compose(transform_list)


class RandomBrightnessContrast:
    def __init__(self, brightness_limit=0.2, contrast_limit=0.2, p=0.5):
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.p = p

    def __call__(self, img):
        if random() < self.p:
            brightness_factor = 1 + np.random.uniform(-self.brightness_limit, self.brightness_limit)
            img = ImageEnhance.Brightness(img).enhance(brightness_factor)
            contrast_factor = 1 + np.random.uniform(-self.contrast_limit, self.contrast_limit)
            img = ImageEnhance.Contrast(img).enhance(contrast_factor)
        return img


class HueSaturationValue:
    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, img):
        if random() < self.p:
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            img[:, :, 0] = (img[:, :, 0] + np.random.uniform(-10, 10)) % 180
            img[:, :, 1] = np.clip(img[:, :, 1] * np.random.uniform(0.8, 1.2), 0, 255)
            img[:, :, 2] = np.clip(img[:, :, 2] * np.random.uniform(0.8, 1.2), 0, 255)
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_HSV2RGB)
            img = Image.fromarray(img)
        return img


class ImageCompression:
    def __init__(self, quality_lower=40, quality_upper=100, p=0.1):
        self.quality_lower = quality_lower
        self.quality_upper = quality_upper
        self.p = p

    def __call__(self, img):
        if random() < self.p:
            quality = np.random.randint(self.quality_lower, self.quality_upper + 1)
            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            img = Image.open(buffer).convert('RGB')
        return img


class GaussNoise:
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, img):
        if random() < self.p:
            img = np.array(img).astype(np.float32)
            noise = np.random.normal(0, np.random.uniform(5, 30), img.shape)
            img = np.clip(img + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(img)
        return img


class MotionBlur:
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, img):
        if random() < self.p:
            size = choice([3, 5, 7])
            kernel = np.zeros((size, size))
            kernel[size // 2, :] = 1 / size
            img = np.array(img)
            img = cv2.filter2D(img, -1, kernel)
            img = Image.fromarray(img)
        return img


class CLAHE:
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, img):
        if random() < self.p:
            img = np.array(img)
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            img = Image.fromarray(img)
        return img


class ChannelShuffle:
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, img):
        if random() < self.p:
            img = np.array(img)
            channels = [0, 1, 2]
            np.random.shuffle(channels)
            img = img[:, :, channels]
            img = Image.fromarray(img)
        return img


class Cutout:
    def __init__(self, p=0.1, max_holes=8, max_size=32):
        self.p = p
        self.max_holes = max_holes
        self.max_size = max_size

    def __call__(self, img):
        if random() < self.p:
            img = np.array(img)
            h, w = img.shape[:2]
            n_holes = np.random.randint(1, self.max_holes + 1)
            for _ in range(n_holes):
                size = np.random.randint(8, self.max_size + 1)
                x = np.random.randint(0, w)
                y = np.random.randint(0, h)
                x1, x2 = max(0, x - size // 2), min(w, x + size // 2)
                y1, y2 = max(0, y - size // 2), min(h, y + size // 2)
                img[y1:y2, x1:x2] = 0
            img = Image.fromarray(img)
        return img


class RandomGamma:
    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, img):
        if random() < self.p:
            gamma = np.random.uniform(0.8, 1.2)
            img = np.array(img).astype(np.float32) / 255.0
            img = np.power(img, gamma) * 255
            img = Image.fromarray(img.astype(np.uint8))
        return img


class GlassBlur:
    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, img):
        if random() < self.p:
            img = np.array(img)
            h, w = img.shape[:2]
            for _ in range(np.random.randint(1, 4)):
                dx, dy = np.random.randint(-2, 3, size=2)
                for i in range(2, h - 2):
                    for j in range(2, w - 2):
                        ni, nj = min(max(i + dx, 0), h - 1), min(max(j + dy, 0), w - 1)
                        img[i, j], img[ni, nj] = img[ni, nj].copy(), img[i, j].copy()
            img = cv2.GaussianBlur(img, (3, 3), 0)
            img = Image.fromarray(img)
        return img


class BinaryDataset(Dataset):
    def __init__(self, root, transform=None, classes=None, sample_list=None):
        self.transform = transform
        self.samples = []
        
        if sample_list is not None and os.path.exists(sample_list):
            self._load_sample_list(sample_list, root)
            print(f"Loaded sample list from {sample_list} ({len(self.samples)} samples)")
        elif os.path.exists(os.path.join(root, '0_real')) or os.path.exists(os.path.join(root, '1_fake')):
            self._add_samples(root)
        else:
            for subdir in os.listdir(root):
                if classes is not None and subdir not in classes:
                    continue
                subpath = os.path.join(root, subdir)
                if os.path.isdir(subpath):
                    self._add_samples(subpath)
    
    def _load_sample_list(self, path, root):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                fpath, label = line.rsplit('\t', 1)
                if not os.path.isabs(fpath):
                    fpath = os.path.join(root, fpath)
                self.samples.append((fpath, int(label)))
    
    def _add_samples(self, folder):
        for label_name, label in [('0_real', 0), ('1_fake', 1), ('1_false', 1)]:
            label_dir = os.path.join(folder, label_name)
            if os.path.exists(label_dir):
                for fname in os.listdir(label_dir):
                    fpath = os.path.join(label_dir, fname)
                    if os.path.isfile(fpath) and fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.jfif')):
                        self.samples.append((fpath, label))
    
    def save_sample_list(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            for fpath, label in self.samples:
                f.write(f"{fpath}\t{label}\n")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        path, label = self.samples[index]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


def binary_dataset(opt, root):
    transform = get_transform(opt)
    classes = getattr(opt, 'classes', None)
    sample_list = getattr(opt, 'sample_list', None)
    return BinaryDataset(root, transform, classes=classes, sample_list=sample_list)


class FileNameDataset(datasets.ImageFolder):
    def name(self):
        return 'FileNameDataset'

    def __init__(self, opt, root):
        self.opt = opt
        super().__init__(root)

    def __getitem__(self, index):
        path, target = self.samples[index]
        return path


def data_augment(img, opt):
    img = np.array(img)
    if random() < opt.blur_prob:
        sig = sample_continuous(opt.blur_sig)
        gaussian_blur(img, sig)
    if random() < opt.jpg_prob:
        method = sample_discrete(opt.jpg_method)
        qual = sample_discrete(opt.jpg_qual)
        img = jpeg_from_key(img, qual, method)
    return Image.fromarray(img)


def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)


def gaussian_blur(img, sigma):
    gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
    gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sigma)
    gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sigma)


def cv2_jpg(img, compress_val):
    img_cv2 = img[:, :, ::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:, :, ::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}


def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)


rz_dict = {
    'bilinear': InterpolationMode.BILINEAR,
    'bicubic': InterpolationMode.BICUBIC,
    'lanczos': InterpolationMode.LANCZOS,
    'nearest': InterpolationMode.NEAREST
}


def custom_resize(img, opt):
    interp = sample_discrete(opt.rz_interp)
    return TF.resize(img, (opt.loadSize, opt.loadSize), interpolation=rz_dict[interp])
