import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
from .blur import add_fixed_blur, add_random_blur, add_random_blur_no_gray

# Loss with label smoothing
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(1)
        log_probs = torch.nn.functional.log_softmax(pred, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=1))

# Optimizer with weight decay exclusion for BatchNorm
def exclude_from_weight_decay(named_params, weight_decay):
    decay, no_decay = [], []
    for name, param in named_params:
        if 'bn' in name or 'bias' in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': decay, 'weight_decay': weight_decay},
            {'params': no_decay, 'weight_decay': 0.0}]

# Cosine learning rate schedule with warmup
def get_lr_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        else:
            return 0.5 * (1 + torch.cos(torch.tensor((epoch - warmup_epochs) / (total_epochs - warmup_epochs) * 3.1415926535)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train(model, dataloader, criterion, optimizer, device, blur, sigma):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(dataloader, desc="Training", leave=False)

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        if blur == "fixed_blur":
            images = add_fixed_blur(images, sigma=sigma)
        elif blur == "strong_random_blur":
            images = add_random_blur_no_gray(images, [0, 1, 2, 4, 8], [0.2, 0.2, 0.2, 0.2, 0.2])
        elif blur == "weak_random_blur":
            images = add_random_blur_no_gray(images, [0, 1, 2, 3, 4, 5], [0.6937, 0.2129, 0.0653, 0.0200, 0.0062, 0.0019])

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100.0 * correct / total

def evaluate(model, dataloader, criterion, device, sigma):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            # images = add_fixed_blur(images, sigma=sigma)
            # images = add_random_blur_no_gray(images, [0, 1, 2, 3, 4, 5], [0.6937, 0.2129, 0.0653, 0.0200, 0.0062, 0.0019]) # v6 - Sigma0to5 V1
            images = add_random_blur_no_gray(images, [0], [1])
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, 100.0 * correct / total

def show_images(images, title=""):
    # Undo normalization if applied
    inv_normalize = torchvision.transforms.Normalize(
        mean=[-m/s for m, s in zip([0.485, 0.456, 0.406],
                                   [0.229, 0.224, 0.225])],
        std=[1/s for s in [0.229, 0.224, 0.225]]
    )
    images = inv_normalize(images)

    grid_img = torchvision.utils.make_grid(images[:8], nrow=4)
    npimg = grid_img.cpu().numpy().transpose(1, 2, 0)
    plt.imshow(npimg)
    plt.title(title)
    plt.axis("off")
    plt.savefig('check.png')
