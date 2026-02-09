import torchvision.transforms as T
import torch
import random
from netrep.blur import add_fixed_blur, add_random_blur_no_gray, generate_blur_weights

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
CIFAR10_MEAN  = [0.4914, 0.4822, 0.4465]
CIFAR10_STD   = [0.2023, 0.1994, 0.2010]

class AugmentationPipeline:
    def __init__(self, config):
        """
        config: dict specifying which augmentations to apply
        Example:
        {
            "crop": True,
            "flip": True,
            "blur": "weak_random_blur",
            "temp": 4.0,
            "sp_prob": 0.01
        }
        """
        self.config = config

        self.transforms = []

        self.dataset = self.config.get("dataset", "imagenet")
        self.img_size = 224 if self.dataset=='imagenet' else 32

        self.to_224 = self.config.get("to_224", False)

        if self.to_224:
            # ImageNet-style geometry for BOTH imagenet and cifar10 sources
            self.transforms.append(T.Resize(256))
            self.transforms.append(T.CenterCrop(224))
        else:
            # Native resolution pipelines
            if self.config.get("crop", False):  
                if self.dataset.lower() == 'imagenet':
                    self.transforms.append(T.RandomResizedCrop(self.img_size))
                elif self.dataset.lower() == 'cifar10':
                    size = self.img_size if self.img_size <= 64 else 32
                    self.transforms.append(T.RandomCrop(size, padding=4, padding_mode="reflect"))

        self.extra_crop_scale = self.config.get("extra_crop_scale", None)
        if self.extra_crop_scale is not None:
            self.transforms.append(T.RandomResizedCrop(32, scale=(self.extra_crop_scale, 1.0), ratio=(1.0, 1.0)))

        if self.config.get("flip", False):
            self.transforms.append(T.RandomHorizontalFlip(p=1.0))

        # Geometric: Random affine transform
        degrees = self.config.get("rot_deg", None)
        shear   = self.config.get("shear_deg", None)

        if degrees is not None or shear is not None:
            self.transforms.append(
                T.RandomAffine(degrees=float(degrees or 0.0), shear=float(shear or 0.0))
            )

        # Photometric: Color jitter
        self.color_jitter_strength = self.config.get("cj_strength", None)
        if self.color_jitter_strength is not None:
            self.transforms.append(T.ColorJitter(
                brightness=self.color_jitter_strength, contrast=self.color_jitter_strength, saturation=self.color_jitter_strength, hue=self.color_jitter_strength/4))
        
        gray_prob = self.config.get("gray_prob", None)
        if gray_prob is not None:
            self.transforms.append(T.RandomGrayscale(p=float(gray_prob)))

        self.to_tensor = T.ToTensor()

        if self.dataset.lower() == "cifar10":
            mean, std = CIFAR10_MEAN, CIFAR10_STD
        else:
            mean, std = IMAGENET_MEAN, IMAGENET_STD
        self.normalize = T.Normalize(mean=mean, std=std)

        # Noise: Random blurring & salt and pepper noise
        self.blur_mode = self.config.get("blur", None)
        self.blur_handlers = {
            "fixed_blur": self._apply_fixed_blur,
            "strong_random_blur": self._apply_strong_random_blur,
            "weak_random_blur": self._apply_weak_random_blur
        }

        self.sp_noise_prob = self.config.get("sp_prob", None)
        self.noise_std = self.config.get("noise_std", None)

        # Occlusion: Cutout
        self.cutout_patch_size = self.config.get("cutout_patch_size", None)
        self.sigma = self.config.get("sigma", None)


    def __call__(self, img):
        # Apply PIL-based transforms first
        for t in self.transforms:
            img = t(img)

        if not isinstance(img, torch.Tensor):
            img = self.to_tensor(img)

        # Blur (noise/blur category)
        if self.blur_mode in self.blur_handlers:
            img = self.blur_handlers[self.blur_mode](img)

        # Salt & Pepper noise (optional extra)
        if self.sp_noise_prob:
            img = self.add_salt_and_pepper_noise(img, prob=self.sp_noise_prob)

        if self.noise_std:
            img = self.add_gaussian_noise(img, std=self.noise_std)

        # Cutout (occlusion)
        if self.cutout_patch_size:
            _, h, w = img.shape
            ps = self._scaled_cutout_size(h, w)
            img = self.random_cutout(img, patch_size=ps, sigma=self.sigma)

        return self.normalize(img)
    
    def _scaled_cutout_size(self, h, w):
        s = self.cutout_patch_size
        if self.to_224:
            scale = w / 32.0
            return max(1, int(round(s * scale)))
        return int(s)

    # --- Blur methods ---
    def _apply_fixed_blur(self, img):
        sigma = self.config.get("sigma", None)
        return add_fixed_blur(img, sigma=sigma)

    def _apply_strong_random_blur(self, img):
        return add_random_blur_no_gray(img, [0, 1, 2, 4, 8], [0.2, 0.2, 0.2, 0.2, 0.2])

    def _apply_weak_random_blur(self, img):
        temp = self.config.get("temp", None)
        weights = generate_blur_weights([0, 1, 2, 3, 4, 5], temperature=temp)
        return add_random_blur_no_gray(img, [0, 1, 2, 3, 4, 5], weights)

    # --- Noise ---
    def add_salt_and_pepper_noise(self, img, prob=0.01):
        """
        img: Tensor (C x H x W), values in [0, 1]
        prob: Probability of a pixel being flipped to salt or pepper
        """
        assert img.dim() == 3, "Expected C x H x W"
        c, h, w = img.shape
        noise = torch.rand(c, h, w, device=img.device)
        salt_mask = noise < (prob / 2)
        pepper_mask = noise > 1 - (prob / 2)
        noisy = img.clone()
        noisy[salt_mask] = 1.0
        noisy[pepper_mask] = 0.0
        return noisy
    
    def add_gaussian_noise(self, img, std=0.1):
        noisy = img + torch.randn_like(img) * std
        return noisy.clamp(0.0, 1.0)

    # --- Occlusion ---
    def random_cutout(self, img, patch_size, sigma=None):
        _, h, w = img.shape

        margin_y = patch_size // 2
        margin_x = patch_size // 2

        cy = torch.randint(margin_y, h - margin_y, (1,)).item()
        cx = torch.randint(margin_x, w - margin_x, (1,)).item()

        y1 = max(cy - patch_size // 2, 0)
        y2 = min(cy + patch_size // 2, h)
        x1 = max(cx - patch_size // 2, 0)
        x2 = min(cx + patch_size // 2, w)
        
        if not sigma: # random cutout
            img[:, y1:y2, x1:x2] = 0.0
        else: # patch gaussian 
            noise = sigma * torch.randn_like(img[:, y1:y2, x1:x2])
            img[:, y1:y2, x1:x2] = torch.clamp(img[:, y1:y2, x1:x2] + noise, 0.0, 1.0)

        return img
