import os

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstD,
    Spacingd,
    Orientationd,
    NormalizeIntensityd,
    CropForegroundd,
    Resized,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    RandScaleIntensityd,
    ToTensord
)

from monai.data import Dataset, DataLoader, CacheDataset
from monai.utils import set_determinism


def prepare(in_dir, pixdim=(0.5, 0.5, 2.0), spatial_size=[128, 128, 15], cache=True):

    set_determinism(seed=0)

    def create_data_list(split):

        images_dir = os.path.join(in_dir, split, "Images")
        labels_dir = os.path.join(in_dir, split, "Labels")

        data = []

        for case in sorted(os.listdir(images_dir)):

            image_path = os.path.join(images_dir, case, case + ".nii.gz")
            label_path = os.path.join(labels_dir, case, case + "_segmentation.nii.gz")

            if os.path.exists(image_path) and os.path.exists(label_path):
                data.append({"image": image_path, "label": label_path})

        return data


    train_files = create_data_list("Train")
    val_files = create_data_list("Validation")
    test_files = create_data_list("Test")

    print("Training cases:", len(train_files))
    print("Validation cases:", len(val_files))
    print("Testing cases:", len(test_files))


    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstD(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS", labels=None),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Resized(keys=["image", "label"], spatial_size=spatial_size, mode=("trilinear", "nearest")),
        RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
        RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3, spatial_axes=(0, 1)),
        RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.5),
        RandScaleIntensityd(keys=["image"], factors=0.10, prob=0.5),
        ToTensord(keys=["image", "label"])
    ])


    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstD(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS", labels=None),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Resized(keys=["image", "label"], spatial_size=spatial_size, mode=("trilinear", "nearest")),
        ToTensord(keys=["image", "label"])
    ])


    test_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstD(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS", labels=None),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Resized(keys=["image", "label"], spatial_size=spatial_size, mode=("trilinear", "nearest")),
        ToTensord(keys=["image", "label"])
    ])


    if cache:

        train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0)
        val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0)
        test_ds = CacheDataset(data=test_files, transform=test_transforms, cache_rate=1.0)

    else:

        train_ds = Dataset(data=train_files, transform=train_transforms)
        val_ds = Dataset(data=val_files, transform=val_transforms)
        test_ds = Dataset(data=test_files, transform=test_transforms)


    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)


    return train_loader, val_loader, test_loader

