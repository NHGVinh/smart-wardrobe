from PIL import Image
from torchvision import transforms


def square_pad_white(image):
    max_size = max(image.size)
    canvas = Image.new("RGB", (max_size, max_size), (255, 255, 255))
    x = (max_size - image.size[0]) // 2
    y = (max_size - image.size[1]) // 2
    canvas.paste(image, (x, y))
    return canvas


def load_resnet_image(image_path):
    return square_pad_white(Image.open(image_path).convert("RGB"))


def build_resnet_transforms(train=False):
    steps = [transforms.Resize((224, 224))]
    if train:
        steps.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
            ]
        )

    steps.append(transforms.ToTensor())

    if train:
        steps.append(transforms.RandomErasing(p=0.3, scale=(0.02, 0.1)))

    steps.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    return transforms.Compose(steps)
