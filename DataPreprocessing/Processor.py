from torchvision import transforms


def get_preprocessor():
    transform = transforms.Compose([
        transforms.Resize([64, 64]),
        transforms.CenterCrop([54, 54]),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    return transform

