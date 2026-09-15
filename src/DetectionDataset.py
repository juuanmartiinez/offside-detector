import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


class DetectionDataset(Dataset):

    def __init__(self, ds):
        self.ds = ds
        self.ids = sorted(ds.id2fichero)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        imgId = self.ids[idx]

        imagen = to_tensor(self.ds.imagen(imgId))
        cajas, etiquetas = self.ds.cajasYEtiquetas(imgId)

        target = {
            "boxes":  torch.tensor(cajas, dtype=torch.float32),
            "labels": torch.tensor(etiquetas, dtype=torch.int64),
        }

        return imagen, target

def collate(batch):
    return tuple(zip(*batch))