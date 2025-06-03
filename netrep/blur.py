import torch
import torchvision.transforms as transforms
import numpy as np
from numpy.random import choice

import math
import kornia


def add_random_blur(images, sigmas, weights):
    blurred_images = torch.zeros_like(images)
    normalize = transforms.Normalize(mean=[0.449], std=[0.226]) # grayscale

    for i in range(images.size(0)): # Batch size
        image = images[i, :, :, :]
        weights = np.asarray(weights).astype('float64')
        weights = weights / np.sum(weights)
        sigma = choice(sigmas, 1, p=weights)[0]
        kernel_size = 2 * math.ceil(2.0 * sigma) + 1

        if sigma == 0:
            blurred_image = image
        else:
            blurred_image = kornia.gaussian_blur2d(torch.unsqueeze(image, dim=0), kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma))[0, :, :, :]
        blurred_image = normalize(blurred_image)
        blurred_images[i] = blurred_image

    blurred_images = blurred_images.repeat(1, 3, 1, 1) # Grayscale to RGB
    return blurred_images


def add_fixed_blur(images, sigma):
    if sigma == 0:
        return images
    
    kernel_size = 2 * int(2.0 * sigma + 0.5) + 1

    blurred_images = kornia.filters.gaussian_blur2d(
        images,
        kernel_size=(kernel_size, kernel_size),
        sigma=(sigma, sigma)
    )

    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
    blurred_images = (blurred_images - mean) / std

    return blurred_images