import os
import torch

from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import DiceLoss

from preprocess import prepare
from Utilities import train


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Device:", device)


in_dir = r"C:\Users\Dxogu\OneDrive\Desktop\Prostrate Segmentation Project"

model_dir = os.path.join(in_dir, "Models")

os.makedirs(model_dir, exist_ok=True)


train_loader, val_loader, test_loader = prepare(in_dir, spatial_size=[256, 256, 15], cache=True)

data_in = (train_loader, val_loader, test_loader)

print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))
print("Test batches:", len(test_loader))

model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128),
    strides=((2, 2, 1), (2, 2, 1), (2, 2, 1)),
    num_res_units=2,
    norm=Norm.BATCH
).to(device)


loss_function = DiceLoss(to_onehot_y=True, softmax=True, squared_pred=True, include_background=False)


optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10)


max_epochs = 300

early_stopping_patience = 30


sample = next(iter(train_loader))

inputs = sample["image"].to(device)

labels = sample["label"].long().to(device)


with torch.no_grad():

    outputs = model(inputs)


print("Input shape:", inputs.shape)

print("Label shape:", labels.shape)

print("Output shape:", outputs.shape)


results = train(model=model, data_in=data_in, loss=loss_function, optim=optimizer, scheduler=scheduler, max_epochs=max_epochs, model_dir=model_dir, validation_interval=1, device=device, patience=early_stopping_patience)


print("Training completed.")

print("Best model saved at:")

print(os.path.join(model_dir, "best_metric_model.pth"))