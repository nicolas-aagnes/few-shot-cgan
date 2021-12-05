import os
from PIL import Image
from pathlib import Path
import torch
import torchvision.transforms.functional as F


class FashionDataset:
    def __init__(self, root, annotation_file, transform):
        self.root = root
        self.transform = transform
        rows = []
        with open(annotation_file) as f:
            rows = f.readlines()

        # First 2 rows are not data, so we remove them
        rows = rows[2:]

        datalist = []
        for row in rows:
            data = row.split()
            labels = [int(r) for r in data[1:]]
            path = Path(data[0])
            filename = os.path.join(self.root, path.name.replace("-img", "/img"))
            assert os.path.isfile(filename), filename
            datalist.append((filename, labels))

        self.datalist = datalist
        self.num_classes = len(labels)  # 26

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        filename = self.datalist[index][0]
        image = Image.open(filename)
        labels = torch.tensor(self.datalist[index][1], dtype=torch.int)
        image = self.transform(image)
        return image, labels
