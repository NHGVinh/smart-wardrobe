from PIL import Image
from torchvision import transforms


def make_white_square_canvas(img_goc):
    # 1. Đọc ảnh gốc đã được truyền vào

    # 2. Tạo Canvas vuông trắng (bảo toàn 100% hình dáng áo, quần, giày)
    max_size = max(img_goc.size)
    image = Image.new("RGB", (max_size, max_size), (255, 255, 255))
    x = (max_size - img_goc.size[0]) // 2
    y = (max_size - img_goc.size[1]) // 2
    image.paste(img_goc, (x, y))
    return image


def load_resnet_image(image_path):
    img_goc = Image.open(image_path).convert("RGB")
    return make_white_square_canvas(img_goc)


data_transforms = {
    "train": transforms.Compose(
        [
            # Ảnh đã vuông sẵn, chỉ ép về 224x224 (KHÔNG dùng CenterCrop/RandomCrop nữa)
            transforms.Resize((224, 224)),

            # Tăng cường dữ liệu (Augmentation) để ResNet18 khôn hơn VGG
            transforms.RandomHorizontalFlip(),  # Lật ảnh ngang ngẫu nhiên
            transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Thay đổi độ sáng/tương phản nhẹ
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.1)),  # Che một mảng nhỏ trên ảnh

            # Bộ số vàng của ImageNet (Bắt buộc phải có để Transfer Learning)
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "val": transforms.Compose(
        [
            # Tập thi: Chỉ thu nhỏ và chuẩn hóa, không thêm nhiễu
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}
