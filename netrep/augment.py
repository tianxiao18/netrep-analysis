import torchvision.transforms as T
import torch
import random
from netrep.blur import add_fixed_blur, add_random_blur_no_gray, generate_blur_weights

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

        if self.config.get("crop", False):
            self.transforms.append(T.RandomResizedCrop(224))

        if self.config.get("flip", False):
            self.transforms.append(T.RandomHorizontalFlip(p=1.0))

        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        
        self.blur_mode = config.get("blur", None)
        self.blur_handlers = {
            "fixed_blur": self._apply_fixed_blur,
            "strong_random_blur": self._apply_strong_random_blur,
            "weak_random_blur": self._apply_weak_random_blur
        }

        self.sp_noise_prob = config.get("sp_prob", None)


    def __call__(self, img):
        # Apply PIL-based transforms first
        for t in self.transforms:
            img = t(img)

        img = self.to_tensor(img)

        # Apply blur after tensor conversion
        if self.blur_mode in self.blur_handlers:
            img = self.blur_handlers[self.blur_mode](img)

        if self.sp_noise_prob:
            img = self.add_salt_and_pepper_noise(img, prob=self.sp_noise_prob)

        return self.normalize(img)
    
    def _apply_fixed_blur(self, img):
        sigma = self.config.get("sigma", None)
        return add_fixed_blur(img, sigma=sigma)

    def _apply_strong_random_blur(self, img):
        return add_random_blur_no_gray(img, [0, 1, 2, 4, 8], [0.2, 0.2, 0.2, 0.2, 0.2])

    def _apply_weak_random_blur(self, img):
        temp = self.config.get("temp", None)
        weights = generate_blur_weights([0, 1, 2, 3, 4, 5], temperature=temp)
        return add_random_blur_no_gray(img, [0, 1, 2, 3, 4, 5], weights)

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
