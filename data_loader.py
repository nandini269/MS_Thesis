import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset

data_loc = './data'

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}

class InfluenceDataset(Dataset):
    """dataset."""

    def __init__(self, data_list, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_list = data_list # list of tuples where 1st -> tensor 2nd -> label
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        # img_name = os.path.join(self.root_dir,
        #                         self.landmarks_frame.iloc[idx, 0])
        # image = io.imread(img_name)
        # landmarks = self.landmarks_frame.iloc[idx, 1:]
        # landmarks = np.array([landmarks])
        # landmarks = landmarks.astype('float').reshape(-1, 2)
        # sample = {'image': image, 'landmarks': landmarks}

        # if self.transform:
        #     sample = self.transform(sample)
        sample = self.data_list[idx]
        return sample

def data_loader(network, dataset, batch_size):
    if dataset == 'mnist':
        return get_image_data_loader(dataset, batch_size)
    elif dataset == 'cifar_10':
        if network == 'mlp':
            raise Exception('MLP is currently only designed for grayscale images')
        return get_image_data_loader(dataset, batch_size)
    elif dataset == 'cifar_100':
        if network == 'mlp':
            raise Exception('MLP is currently only designed for grayscale images')
        return get_image_data_loader(dataset, batch_size)
    else:
        raise Exception('dataset is not supported')


def get_image_data_loader(dataset_name, batch_size):
    if dataset_name == 'mnist':
        return mnist_data_loader(batch_size)
    elif dataset_name == 'cifar_10':
        return cifar_10_data_loader(batch_size)
    elif dataset_name == 'cifar_100':
        return cifar_100_data_loader(batch_size)
    else:
        raise Exception('dataset is not supported')


def mnist_data_loader(batch_size):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST(root=data_loc, train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=1)
    test_dataset = datasets.MNIST(root=data_loc, train=False, transform=transform, )
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    return train_loader, test_loader


def cifar_10_data_loader(batch_size):
    transform_train = transforms.Compose(
        [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
    dataset = datasets.CIFAR10(root=data_loc, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=1)
    test_dataset = datasets.CIFAR10(root=data_loc, train=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    return dataset, train_loader, test_loader


def cifar_100_data_loader(batch_size):
    transform_train = transforms.Compose(
        [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
         transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)), ])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
        (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)), ])
    dataset = datasets.CIFAR100(root=data_loc, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=1)
    test_dataset = datasets.CIFAR100(root=data_loc, train=False, transform=transform_test, )
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    return train_loader, test_loader
