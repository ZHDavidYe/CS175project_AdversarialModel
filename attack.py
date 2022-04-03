from ctypes import sizeof
from typing import List, Iterator, Dict, Tuple, Any, Type
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import DataLoader
from transformers.data.data_collator import default_data_collator
import torch.nn.functional as F


class Attack:

    def __init__(self, model, device, attack_path, epsilon=2.5, min_val=0, max_val=1):
        self.model = model.to(device)
        self.device = device
        self.attack_path = attack_path
        self.epsilon = epsilon
        self.min_val = min_val
        self.max_val = max_val

    def compute_gradient(self, original_images, labels):
        original_images = torch.tensor(original_images).to(self.device)
        # original_images = torch.unsqueeze(original_images, 0)
        labels = torch.tensor(labels).to(self.device)
        original_images.requires_grad = True
        self.model.eval()
        outputs = self.model(original_images)
        loss = F.cross_entropy(outputs, labels)
        # self.model.zero_grad()
        data_grad = torch.autograd.grad(loss, original_images)[0]
        return data_grad

    def attack(self, original_images, labels):

        original_images = torch.tensor(original_images).to(self.device)
        # original_images = torch.unsqueeze(original_images, 0)
        labels = torch.tensor(labels).to(self.device)
        correct = 0
        perturbed_image = deepcopy(original_images)
        perturbed_image = torch.unsqueeze(perturbed_image, 0)
        numOfIterations = 10
        iterOfMinDist = numOfIterations

        for i in range(numOfIterations):
            # epsilon = np.random.uniform(0.90*self.epsilon, 1.05*self.epsilon)
            epsilon = self.epsilon
            data_grad = self.compute_gradient(perturbed_image, labels)
            sign_data_grad = data_grad.sign()
            perturbed_image.data += epsilon * sign_data_grad
            dist = (perturbed_image - original_images)
            dist = dist.view(perturbed_image.shape[0], -1)
            dist_norm = torch.norm(dist, dim=1, keepdim=True)
            mask = (dist_norm > epsilon).unsqueeze(2).unsqueeze(3)
            dist = dist / dist_norm
            dist *= epsilon
            dist = dist.view(perturbed_image.shape)

            perturbed_image = (original_images + dist) * mask.float() + perturbed_image * (1 - mask.float())
            perturbed_image.clamp_(self.min_val, self.max_val)

            adv_outputs = self.model(perturbed_image)
            final_pred = adv_outputs.max(1, keepdim=True)[1]
            if final_pred.item() != labels.item():
                correct += 1
                if iterOfMinDist != numOfIterations:
                    if (dist - minDist).sign().sum() < 0:
                        minDist = dist
                        perturbed_image = (original_images + dist) * mask.float() + perturbed_image * (1 - mask.float())
                else:
                    iterOfMinDist = i
                    minDist = dist
                    perturbed_image = (original_images + dist) * mask.float() + perturbed_image * (1 - mask.float())

        if correct >= numOfIterations:
            correct = 1
        else:
            correct = 0

        return perturbed_image.cpu().detach().numpy(), correct