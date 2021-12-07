import json
import os

import boto3
import torch
from torchvision.datasets.vision import VisionDataset
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode


class A2D2_S3_dataset(VisionDataset):
    # This custom dataset class gets records one by one from S3 via the __getitem__

    def __init__(
        self,
        manifest_file,
        class_list,
        s3_bucket,
        transform,
        target_transform,
        cache,
        height=1208,
        width=1920,
    ):

        with open(os.path.join(manifest_file)) as file:
            self.manifest = json.load(file)
            self.record_list = list(self.manifest.items())  # convert to list for indexing

        with open(os.path.join(class_list)) as file:
            self.class_list = json.load(file)

        self.hex2rgb = {
            k: tuple(int(k.strip("#")[i : i + 2], 16) for i in (0, 2, 4))
            for k in self.class_list.keys()
        }
        self.rgb2ids = {r: i for i, r in enumerate(self.hex2rgb.values())}

        self.height = height
        self.width = width
        self.bucket = s3_bucket
        self.transform = transform
        self.target_transform = target_transform
        self.cache = cache
        self.s3 = boto3.resource("s3")

    def __len__(self):

        return len(self.manifest)

    def __getitem__(self, idx):

        s3 = self.s3

        img_name = self.record_list[idx][1]["image_name"]
        img_key = self.record_list[idx][1]["image_path"]
        label_name = self.record_list[idx][1]["label_name"]
        label_key = self.record_list[idx][1]["label_path"]

        # read file from S3 or locally, if already downloaded
        image_local_path = os.path.join(self.cache, img_name)
        if os.path.exists(image_local_path):
            image = read_image(image_local_path, mode=ImageReadMode.RGB)
        else:
            s3.meta.client.download_file(
                Key=img_key, Filename=image_local_path, Bucket=self.bucket
            )
            image = read_image(image_local_path, mode=ImageReadMode.RGB)
        image = self.transform(image)

        label_local_path = os.path.join(self.cache, label_name)
        if os.path.exists(label_local_path):
            label = read_image(label_local_path, mode=ImageReadMode.RGB)
        else:
            s3.meta.client.download_file(
                Key=label_key, Filename=label_local_path, Bucket=self.bucket
            )
            label = read_image(label_local_path, mode=ImageReadMode.RGB)
        label = self.target_transform(label)

        # Segmentation masks are provided in RGB. We convert to a matrix of class IDs
        mask = torch.zeros(self.height, self.width)
        for rgb, cid in self.rgb2ids.items():
            color_mask = label == torch.Tensor(rgb).reshape([3, 1, 1])
            seg_mask = color_mask.sum(dim=0) == 3
            mask[seg_mask] = cid

        return torch.div(image, 255), mask.type(torch.int64)
