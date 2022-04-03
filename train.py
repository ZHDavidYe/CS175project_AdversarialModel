"""
The template for the students to train the model.
Please do not change the name of the functions in Adv_Training.
"""
import sys
sys.path.append(".")
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from tasks.defense_project.predict import LeNet

class Adv_Training():
    """
    The class is used to set the defense related to adversarial training and adjust the loss function. Please design your own training methods and add some adversarial examples for training.
    The perturb function is used to generate the adversarial examples for training.
    """
    def __init__(self, device, epsilon=0.3, min_val=0, max_val=1):
        self.model = LeNet().to(device)
        self.epsilon = epsilon
        self.min_val = min_val
        self.max_val = max_val

    def perturb(self, original_images, labels):
        original_images.requires_grad = True
        self.model.eval()
        outputs = self.model(original_images)
        loss = F.nll_loss(outputs, labels)
        self.model.zero_grad()
        loss.backward()
        data_grad = original_images.grad.data
        sign_data_grad = data_grad.sign()
        perturbed_image = original_images + self.epsilon*sign_data_grad
        perturbed_image = torch.clamp(perturbed_image, self.min_val, self.max_val)
        self.model.zero_grad()
        return perturbed_image

    def train(self, model, trainset, valset, device, epoches=10):
        model.to(device)
        model.train()
        # print(trainset)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=10)
        # print(trainloader)
        dataset_size = len(trainset)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())

        #addl_imgs = []  ## adv images for incrementing the training set
        for epoch in range(epoches):  # loop over the dataset multiple times
            running_loss = 0.0
            training_increment = []

            for i, (inputs, labels) in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                ## Modification starts
                # if(i != 0):
                #     combined_inputs = torch.cat((inputs, training_increment[0]))
                #     combined_labels = torch.cat((labels, training_increment[1]))
                # if(i == 0):
                #     outputs = model(inputs)
                #     loss = criterion(outputs, labels)
                # else:
                #     outputs = model(combined_inputs)
                #     loss = criterion(outputs, combined_labels)
                # print("initial loss: ", loss)

                perturbed_data = self.perturb(inputs, labels)
                # if(i == 0):
                #     training_increment.append(perturbed_data)
                #     training_increment.append(labels)
                # else:
                #     training_increment[0] = torch.cat((training_increment[0], perturbed_data))
                #     training_increment[1] = torch.cat((training_increment[1], labels))
                perturbed_output = model(perturbed_data)
                ## adjust the value of epsilon here
                loss += 0.05*criterion(perturbed_output, labels)
                ## Modification ends
                loss.backward()
                optimizer.step()
                ## also pick a few random ones from addl_imgs and do the same
                # (addl_loss_imgs, addl_labels) = random.sample(addlimgs)
                #
                # loss += 0.01*criterion(addl_loss_imgs, addl_labels)
                #
                # addl_imgs.append((perturbed_data, labels))

                running_loss += loss.item()
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / dataset_size))
            running_loss = 0.0

        valloader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=True, num_workers=10)
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valloader:
                # print(inputs.shape, labels.shape)
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                # print(predicted)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print("Accuracy of the network on the val images: %.3f %%" % (100 * correct / total))

        return model


