import torch
import torchvision.transforms as transforms
import numpy as np
from numpy.random import choice

import math
import kornia


def add_random_blur(images, sigmas, weights):
    blurred_images = torch.zeros_like(images)

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
        blurred_images[i] = blurred_image

    blurred_images = blurred_images.repeat(1, 3, 1, 1) # Grayscale to RGB
    return blurred_images


def add_fixed_blur(images, sigma):
    if sigma is None:
        return images
    
    kernel_size = 2 * int(2.0 * sigma + 0.5) + 1

    blurred_images = kornia.filters.gaussian_blur2d(
        images.unsqueeze(0),
        kernel_size=(kernel_size, kernel_size),
        sigma=(sigma, sigma)
    )

    return blurred_images[0]

def add_random_blur_no_gray(img, sigmas, weights):
    weights = np.asarray(weights).astype('float64')
    weights = weights / weights.sum()
    rng = np.random.default_rng()

    # Sample one sigma per image
    sigma = rng.choice(sigmas, p=weights)
    if sigma == 0: return img

    kernel_size = 2 * math.ceil(2.0 * sigma) + 1
    img_batch = img.unsqueeze(0)
    
    blurred = kornia.filters.gaussian_blur2d(
        img_batch,  # (T, C, H, W)
        kernel_size=(kernel_size, kernel_size),
        sigma=(sigma, sigma)
    )
    return blurred[0]

def generate_blur_weights(sigmas, temperature=1.0):
    temperature = 1
    weights = np.exp(-np.array(sigmas) / temperature)
    return (weights / weights.sum()).tolist()