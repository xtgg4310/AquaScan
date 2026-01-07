import torchvision.transforms as transforms


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

class EraseChannel:
    def __init__(self, erase_channel=[0, 1]):
        self.erase_channel = erase_channel

    def __call__(self, x):
        # x should be a tensor [C, H, W]
        # assert max(self.erase_channel) < x.shape[0]
        if x.shape[0] == 1:
            return x
        for channel in self.erase_channel:
            x[channel, :, :] = 0
        return x
    
def get_transforms(transform_type):
    data_transforms = {
        'none': transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 128)),
        ]),
        'totensor': transforms.ToTensor(),
    }
    return data_transforms[transform_type]
