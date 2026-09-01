import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from monai.utils import first
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.data import decollate_batch


dice_metric = DiceMetric(include_background=False, reduction="mean")
post_pred = AsDiscrete(argmax=True, to_onehot=2)
post_label = AsDiscrete(to_onehot=2)


def calculate_dice(predicted, target):
    predicted = [post_pred(i) for i in decollate_batch(predicted)]
    target = [post_label(i) for i in decollate_batch(target)]
    dice_metric(y_pred=predicted, y=target)
    value = dice_metric.aggregate().item()
    dice_metric.reset()
    return value


def calculate_weights(val1, val2):
    count = np.array([val1, val2])
    summ = count.sum()
    weights = count/summ
    weights = 1/weights
    summ = weights.sum()
    weights = weights/summ
    return torch.tensor(weights, dtype=torch.float32)


def train(model, data_in, loss, optim, scheduler, max_epochs, model_dir, validation_interval=1, device=torch.device("cuda:0"), patience=30):

    best_metric = -1
    best_metric_epoch = -1
    epochs_without_improvement = 0

    save_loss_train = []
    save_loss_validation = []
    save_metric_train = []
    save_metric_validation = []
    save_learning_rate = []

    train_loader, validation_loader, test_loader = data_in

    os.makedirs(model_dir, exist_ok=True)

    for epoch in range(max_epochs):

        print("-" * 50)
        print(f"epoch {epoch + 1}/{max_epochs}")

        model.train()

        train_epoch_loss = 0
        train_step = 0
        epoch_metric_train = 0

        for batch_data in tqdm(train_loader):

            train_step += 1

            volume = batch_data["image"].to(device)
            label = batch_data["label"].long().to(device)

            optim.zero_grad()

            outputs = model(volume)

            train_loss = loss(outputs, label)

            train_loss.backward()

            optim.step()

            train_epoch_loss += train_loss.item()

            train_metric = calculate_dice(outputs.detach(), label)

            epoch_metric_train += train_metric

            print(f"{train_step}/{len(train_loader)}, Train_loss: {train_loss.item():.4f}, Train_dice: {train_metric:.4f}")

        train_epoch_loss /= train_step
        epoch_metric_train /= train_step

        save_loss_train.append(train_epoch_loss)
        save_metric_train.append(epoch_metric_train)

        np.save(os.path.join(model_dir, "loss_train.npy"), save_loss_train)
        np.save(os.path.join(model_dir, "metric_train.npy"), save_metric_train)

        print(f"Epoch_loss: {train_epoch_loss:.4f}")
        print(f"Epoch_metric: {epoch_metric_train:.4f}")

        if (epoch + 1) % validation_interval == 0:

            model.eval()

            validation_epoch_loss = 0
            validation_step = 0
            epoch_metric_validation = 0

            with torch.no_grad():

                for validation_data in validation_loader:

                    validation_step += 1

                    validation_volume = validation_data["image"].to(device)
                    validation_label = validation_data["label"].long().to(device)

                    validation_outputs = model(validation_volume)

                    validation_loss = loss(validation_outputs, validation_label)

                    validation_epoch_loss += validation_loss.item()

                    validation_metric = calculate_dice(validation_outputs, validation_label)

                    epoch_metric_validation += validation_metric

            validation_epoch_loss /= validation_step
            epoch_metric_validation /= validation_step

            save_loss_validation.append(validation_epoch_loss)
            save_metric_validation.append(epoch_metric_validation)

            np.save(os.path.join(model_dir, "loss_validation.npy"), save_loss_validation)
            np.save(os.path.join(model_dir, "metric_validation.npy"), save_metric_validation)

            print(f"Validation_loss_epoch: {validation_epoch_loss:.4f}")
            print(f"Validation_dice_epoch: {epoch_metric_validation:.4f}")

            if scheduler is not None:
                scheduler.step(epoch_metric_validation)

            learning_rate = optim.param_groups[0]["lr"]
            save_learning_rate.append(learning_rate)

            print(f"Learning rate: {learning_rate}")

            if epoch_metric_validation > best_metric:

                best_metric = epoch_metric_validation
                best_metric_epoch = epoch + 1
                epochs_without_improvement = 0

                torch.save(model.state_dict(), os.path.join(model_dir, "best_metric_model.pth"))

                print("Validation Dice improved. Best model saved.")

            else:

                epochs_without_improvement += 1

                print(f"No improvement for {epochs_without_improvement}/{patience} epochs.")

            print(f"Current epoch: {epoch + 1}")
            print(f"Current validation Dice: {epoch_metric_validation:.4f}")
            print(f"Best validation Dice: {best_metric:.4f}")
            print(f"Best epoch: {best_metric_epoch}")

            training_results = pd.DataFrame({
                "Epoch": range(1, len(save_loss_train) + 1),
                "Training Loss": pd.Series(save_loss_train),
                "Training Dice": pd.Series(save_metric_train),
                "Validation Loss": pd.Series(save_loss_validation),
                "Validation Dice": pd.Series(save_metric_validation),
                "Learning Rate": pd.Series(save_learning_rate)
            })

            excel_path = os.path.join(model_dir, "training_results.xlsx")
            training_results.to_excel(excel_path, index=False)

            if epochs_without_improvement >= patience:

                print("-" * 50)
                print(f"Early stopping at epoch: {epoch + 1}")
                print(f"Best validation Dice: {best_metric:.4f}")
                print(f"Best epoch: {best_metric_epoch}")

                break

    print("-" * 50)
    print("Training completed")
    print(f"Best validation Dice: {best_metric:.4f}")
    print(f"Best epoch: {best_metric_epoch}")
    print(f"Training results saved to: {excel_path}")

    return training_results


def test_model(model, test_loader, loss, model_dir, device=torch.device("cuda:0")):

    model.load_state_dict(torch.load(os.path.join(model_dir, "best_metric_model.pth"), map_location=device))
    model.to(device)
    model.eval()

    test_epoch_loss = 0
    test_epoch_metric = 0
    test_step = 0
    case_results = []

    with torch.no_grad():

        for test_data in tqdm(test_loader):

            test_step += 1

            test_volume = test_data["image"].to(device)
            test_label = test_data["label"].long().to(device)

            test_outputs = model(test_volume)

            test_loss = loss(test_outputs, test_label)

            test_metric = calculate_dice(test_outputs, test_label)

            test_epoch_loss += test_loss.item()
            test_epoch_metric += test_metric

            case_results.append({"Case": test_step, "Loss": test_loss.item(), "Dice": test_metric})

    test_epoch_loss /= test_step
    test_epoch_metric /= test_step

    print("-" * 50)
    print(f"Final Test Loss: {test_epoch_loss:.4f}")
    print(f"Final Test Dice: {test_epoch_metric:.4f}")

    test_results = pd.DataFrame(case_results)

    test_results.loc[len(test_results)] = ["Mean", test_epoch_loss, test_epoch_metric]

    excel_path = os.path.join(model_dir, "test_results.xlsx")

    test_results.to_excel(excel_path, index=False)

    print(f"Test results saved to: {excel_path}")

    return test_epoch_loss, test_epoch_metric


def show_patient(data, SLICE_NUMBER=7, train=True, validation=False, test=False):

    train_loader, validation_loader, test_loader = data

    if train:

        patient = first(train_loader)

        plt.figure("Visualization Train", (12, 6))

        plt.subplot(1, 2, 1)
        plt.title(f"Image {SLICE_NUMBER}")
        plt.imshow(patient["image"][0, 0, :, :, SLICE_NUMBER], cmap="gray")

        plt.subplot(1, 2, 2)
        plt.title(f"Label {SLICE_NUMBER}")
        plt.imshow(patient["label"][0, 0, :, :, SLICE_NUMBER])

        plt.show()

    if validation:

        patient = first(validation_loader)

        plt.figure("Visualization Validation", (12, 6))

        plt.subplot(1, 2, 1)
        plt.title(f"Image {SLICE_NUMBER}")
        plt.imshow(patient["image"][0, 0, :, :, SLICE_NUMBER], cmap="gray")

        plt.subplot(1, 2, 2)
        plt.title(f"Label {SLICE_NUMBER}")
        plt.imshow(patient["label"][0, 0, :, :, SLICE_NUMBER])

        plt.show()

    if test:

        patient = first(test_loader)

        plt.figure("Visualization Test", (12, 6))

        plt.subplot(1, 2, 1)
        plt.title(f"Image {SLICE_NUMBER}")
        plt.imshow(patient["image"][0, 0, :, :, SLICE_NUMBER], cmap="gray")

        plt.subplot(1, 2, 2)
        plt.title(f"Label {SLICE_NUMBER}")
        plt.imshow(patient["label"][0, 0, :, :, SLICE_NUMBER])

        plt.show()


def show_prediction(model, loader, device=torch.device("cuda:0"), SLICE_NUMBER=7):

    model.eval()

    patient = first(loader)

    volume = patient["image"].to(device)
    label = patient["label"].to(device)

    with torch.no_grad():

        output = model(volume)

        prediction = torch.argmax(output, dim=1, keepdim=True)

    image_slice = volume[0, 0, :, :, SLICE_NUMBER].cpu()
    label_slice = label[0, 0, :, :, SLICE_NUMBER].cpu()
    prediction_slice = prediction[0, 0, :, :, SLICE_NUMBER].cpu()

    plt.figure("Prediction", (18, 6))

    plt.subplot(1, 3, 1)
    plt.title(f"MRI {SLICE_NUMBER}")
    plt.imshow(image_slice, cmap="gray")

    plt.subplot(1, 3, 2)
    plt.title(f"Ground Truth {SLICE_NUMBER}")
    plt.imshow(label_slice)

    plt.subplot(1, 3, 3)
    plt.title(f"Prediction {SLICE_NUMBER}")
    plt.imshow(prediction_slice)

    plt.show()


def calculate_pixels(data):

    background = 0
    prostate = 0

    for batch in tqdm(data):

        label = batch["label"]

        background += torch.sum(label == 0).item()
        prostate += torch.sum(label == 1).item()

    print("Background voxels:", background)
    print("Prostate voxels:", prostate)

    return background, prostate


def plot_training_results(results):

    plt.figure(figsize=(10, 6))
    plt.plot(results["Epoch"], results["Training Loss"], label="Training Loss")
    plt.plot(results["Epoch"], results["Validation Loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(results["Epoch"], results["Training Dice"], label="Training Dice")
    plt.plot(results["Epoch"], results["Validation Dice"], label="Validation Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.title("Training and Validation Dice")
    plt.legend()
    plt.grid()
    plt.show()
