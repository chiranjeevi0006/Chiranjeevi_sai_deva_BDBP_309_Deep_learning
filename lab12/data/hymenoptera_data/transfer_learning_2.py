import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import numpy as np
import torchvision
from torchvision import datasets, transforms, models

import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory


# ============================================================
# 1. DATA TRANSFORMS
# ============================================================

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ]),

    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ]),
}


# ============================================================
# 2. DATASET PATH
# ============================================================

data_dir = '/home/ibab/Documents/Deep_learning_chiranjeevi_sai_deva/lab12/data/hymenoptera_data'

image_datasets = {
    x: datasets.ImageFolder(
        os.path.join(data_dir, x),
        data_transforms[x]
    )
    for x in ['train', 'val']
}


# ============================================================
# 3. DATALOADERS
# ============================================================

dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x],
        batch_size=4,
        shuffle=True,
        num_workers=4
    )
    for x in ['train', 'val']
}

dataset_sizes = {
    x: len(image_datasets[x])
    for x in ['train', 'val']
}

class_names = image_datasets['train'].classes

print("Classes:", class_names)
print("Train size:", dataset_sizes['train'])
print("Validation size:", dataset_sizes['val'])


# ============================================================
# 4. DEVICE
# ============================================================

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using {device} device")


# ============================================================
# 5. DISPLAY IMAGE
# ============================================================

def imshow(inp, title=None):
    """Display a Tensor image."""

    inp = inp.numpy().transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)

    plt.imshow(inp)

    if title is not None:
        plt.title(title)

    plt.pause(0.001)


# ============================================================
# 6. DISPLAY A BATCH OF TRAINING IMAGES
# ============================================================

inputs, classes = next(iter(dataloaders['train']))

out = torchvision.utils.make_grid(inputs)

imshow(
    out,
    title=[class_names[x] for x in classes]
)

plt.show()


# ============================================================
# 7. TRAINING FUNCTION
# ============================================================

def train_model(
        model,
        criterion,
        optimizer,
        scheduler,
        num_epochs=25
):

    since = time.time()

    # Temporary directory for best model
    with TemporaryDirectory() as tempdir:

        best_model_params_path = os.path.join(
            tempdir,
            'best_model_params.pt'
        )

        torch.save(
            model.state_dict(),
            best_model_params_path
        )

        best_acc = 0.0

        # ----------------------------------------------------
        # EPOCH LOOP
        # ----------------------------------------------------

        for epoch in range(num_epochs):

            print(f'Epoch {epoch + 1}/{num_epochs}')
            print('-' * 10)

            # ------------------------------------------------
            # TRAIN + VALIDATION
            # ------------------------------------------------

            for phase in ['train', 'val']:

                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                # --------------------------------------------
                # BATCH LOOP
                # --------------------------------------------

                for inputs, labels in dataloaders[phase]:

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # Clear gradients
                    optimizer.zero_grad()

                    # Forward pass
                    with torch.set_grad_enabled(
                            phase == 'train'):

                        outputs = model(inputs)

                        _, preds = torch.max(
                            outputs,
                            1
                        )

                        loss = criterion(
                            outputs,
                            labels
                        )

                        # ------------------------------------
                        # BACKWARD PASS
                        # ------------------------------------

                        if phase == 'train':

                            loss.backward()

                            optimizer.step()

                    # ----------------------------------------
                    # STATISTICS
                    # ----------------------------------------

                    running_loss += (
                        loss.item()
                        * inputs.size(0)
                    )

                    running_corrects += torch.sum(
                        preds == labels.data
                    )

                # --------------------------------------------
                # UPDATE LEARNING RATE
                # --------------------------------------------

                if phase == 'train':
                    scheduler.step()

                # --------------------------------------------
                # EPOCH LOSS
                # --------------------------------------------

                epoch_loss = (
                    running_loss
                    / dataset_sizes[phase]
                )

                # --------------------------------------------
                # EPOCH ACCURACY
                # --------------------------------------------

                epoch_acc = (
                    running_corrects.double()
                    / dataset_sizes[phase]
                )

                print(
                    f'{phase} Loss: {epoch_loss:.4f} '
                    f'Acc: {epoch_acc:.4f}'
                )

                # --------------------------------------------
                # SAVE BEST MODEL
                # --------------------------------------------

                if (
                    phase == 'val'
                    and epoch_acc > best_acc
                ):

                    best_acc = epoch_acc

                    torch.save(
                        model.state_dict(),
                        best_model_params_path
                    )

            print()

        # ====================================================
        # TRAINING COMPLETE
        # ====================================================

        time_elapsed = time.time() - since

        print(
            f'Training complete in '
            f'{time_elapsed // 60:.0f}m '
            f'{time_elapsed % 60:.0f}s'
        )

        print(
            f'Best val Acc: {best_acc:.4f}'
        )

        # Load best weights
        model.load_state_dict(
            torch.load(
                best_model_params_path,
                weights_only=True
            )
        )

    return model


# ============================================================
# 8. VISUALIZE MODEL PREDICTIONS
# ============================================================

def visualize_model(model, num_images=6):

    was_training = model.training

    model.eval()

    images_so_far = 0

    fig = plt.figure()

    with torch.no_grad():

        for i, (inputs, labels) in enumerate(
                dataloaders['val']):

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            _, preds = torch.max(
                outputs,
                1
            )

            for j in range(inputs.size(0)):

                images_so_far += 1

                ax = plt.subplot(
                    num_images // 2,
                    2,
                    images_so_far
                )

                ax.axis('off')

                ax.set_title(
                    f'Predicted: '
                    f'{class_names[preds[j]]}'
                )

                imshow(
                    inputs.cpu().data[j]
                )

                if images_so_far == num_images:

                    model.train(
                        mode=was_training
                    )

                    plt.show()

                    return

    model.train(
        mode=was_training
    )

    plt.show()


# ============================================================
# 9. MODEL 1 — FULL FINE-TUNING
# ============================================================

print("\n======================================")
print("MODEL 1: FULL FINE-TUNING")
print("======================================\n")


# Load pretrained ResNet18
model_ft = models.resnet18(
    weights='IMAGENET1K_V1'
)


# Number of input features to final layer
num_ftrs = model_ft.fc.in_features

print("Features entering FC layer:", num_ftrs)


# Replace final layer
model_ft.fc = nn.Linear(
    num_ftrs,
    2
)


# Move model to device
model_ft = model_ft.to(device)


# Loss function
criterion = nn.CrossEntropyLoss()


# Optimizer
optimizer_ft = optim.SGD(
    model_ft.parameters(),
    lr=0.001,
    momentum=0.9
)


# Learning-rate scheduler
exp_lr_scheduler = lr_scheduler.StepLR(
    optimizer_ft,
    step_size=7,
    gamma=0.1
)


# Train model
model_ft = train_model(
    model_ft,
    criterion,
    optimizer_ft,
    exp_lr_scheduler,
    num_epochs=25
)


# Visualize predictions
visualize_model(
    model_ft,
    num_images=6
)


# ============================================================
# 10. MODEL 2 — TRANSFER LEARNING / FEATURE EXTRACTION
# ============================================================

print("\n======================================")
print("MODEL 2: FEATURE EXTRACTION")
print("======================================\n")


# Load another pretrained ResNet18
model_conv = models.resnet18(
    weights='IMAGENET1K_V1'
)


# Freeze all pretrained parameters
for param in model_conv.parameters():
    param.requires_grad = False


# Get number of input features
num_ftrs = model_conv.fc.in_features


# Replace final layer
model_conv.fc = nn.Linear(
    num_ftrs,
    2
)


# Move model to device
model_conv = model_conv.to(device)


# Loss function
criterion = nn.CrossEntropyLoss()


# IMPORTANT:
# Only the final FC layer is optimized
optimizer_conv = optim.SGD(
    model_conv.fc.parameters(),
    lr=0.001,
    momentum=0.9
)


# Learning-rate scheduler
exp_lr_scheduler = lr_scheduler.StepLR(
    optimizer_conv,
    step_size=7,
    gamma=0.1
)


# Train
model_conv = train_model(
    model_conv,
    criterion,
    optimizer_conv,
    exp_lr_scheduler,
    num_epochs=25
)


# Visualize
visualize_model(
    model_conv,
    num_images=6
)


# ============================================================
# 11. PREDICT A SINGLE IMAGE
# ============================================================

def visualize_model_predictions(
        model,
        img_path
):

    was_training = model.training

    model.eval()

    # --------------------------------------------
    # CHECK IMAGE EXISTS
    # --------------------------------------------

    if not os.path.exists(img_path):

        print(
            "Image not found:"
        )

        print(img_path)

        return

    # --------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------

    img = Image.open(img_path).convert('RGB')

    # Apply validation transformations
    img = data_transforms['val'](img)

    # Add batch dimension
    img = img.unsqueeze(0)

    # Move to device
    img = img.to(device)

    # --------------------------------------------
    # PREDICTION
    # --------------------------------------------

    with torch.no_grad():

        outputs = model(img)

        _, preds = torch.max(
            outputs,
            1
        )

    # --------------------------------------------
    # DISPLAY
    # --------------------------------------------

    plt.figure(figsize=(5, 5))

    plt.axis('off')

    plt.title(
        f'Predicted: '
        f'{class_names[preds[0]]}'
    )

    imshow(
        img.cpu().data[0]
    )

    plt.show()

    # Restore training state
    model.train(
        mode=was_training
    )


# ============================================================
# 12. TEST ONE IMAGE
# ============================================================

test_image = os.path.join(
    data_dir,
    'val',
    'bees',
    '72100438_73de9f17af.jpg'
)

print("\nTesting image:")
print(test_image)


visualize_model_predictions(
    model_conv,
    test_image
)