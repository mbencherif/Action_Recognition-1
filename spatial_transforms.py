import random
from PIL import Image

import torch


class Compose(object):
    def __init__(self, transforms):
        self.group_transforms = []
        self.transforms = []
        for t in transforms:
            if isinstance(t, ToTensor) or isinstance(t, Normalize):
                self.transforms.append(t)
            else:
                self.group_transforms.append(t)

    def __call__(self, img_group):
        for t in self.group_transforms:
            img_group = t(img_group)
        for t in self.transforms:
            img_group = [t(img) for img in img_group]

        return img_group


class GroupScaleCenterCrop(object):

    def __init__(self, size, scale=1.0, interpolation=Image.BILINEAR):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.scale = scale
        self.interpolation = interpolation

    def __call__(self, img_group):
        x, y = img_group[0].size
        crop = int(min(x, y) * self.scale)

        x1 = (x - crop) // 2
        y1 = (y - crop) // 2
        x2 = x1 + crop
        y2 = y1 + crop

        img_group = [img.crop((x1, y1, x2, y2)) for img in img_group]
        return [img.resize(self.size, self.interpolation) for img in img_group]


class GroupRandomScaleCenterCrop(object):

    def __init__(
        self, size, scales=(0.8, 0.9, 1.0), interpolation=Image.BILINEAR
    ):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.scales = scales
        self.interpolation = interpolation

    def __call__(self, img_group):

        scale = random.choice(self.scales)

        x, y = img_group[0].size
        crop = int(min(x, y) * scale)

        x1 = (x - crop) // 2
        y1 = (y - crop) // 2
        x2 = x1 + crop
        y2 = y1 + crop

        img_group = [img.crop((x1, y1, x2, y2)) for img in img_group]
        return [img.resize(self.size, self.interpolation) for img in img_group]


class GroupRandomHorizontalFlip(object):

    def __call__(self, img_group):
        p = random.random()
        if p < 0.5:
            return [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
        else:
            return img_group


class ToTensor(object):
    def __call__(self, pic):
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        img = img.view(pic.size[1], pic.size[0], len(pic.mode))
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255)


class Normalize(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):

        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor


if __name__ == '__main__':
    trans = Compose(
        [
            GroupRandomScaleCenterCrop(112),
            GroupRandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )
    img = Image.open('../test.jpg')
    img_group = [img] * 3
    rst = trans(img_group)
