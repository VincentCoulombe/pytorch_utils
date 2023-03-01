from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import StratifiedGroupKFold
import torch
from pathlib import Path
import os
import re
import cv2
import json
import numpy as np
import math
import shutil
import pandas as pd
import random

from utils.cloud_downloader import CloudDownloader


class DatasetsManager(object):
    """Download les images et les labels du cloud, et les split en train et val"""

    def __init__(self) -> None:
        self.dataset_dir = get_dataset_dir()
        self.labels = self._get_labels()
        self.val_labels = []
        self.train_labels = []
        self.cloud_downloader = CloudDownloader(self.dataset_dir)

    def update_datasets(self):
        print("Mise à jour des datasets...")
        # 1) Downloader les nouvelles images du cloud
        self.cloud_downloader.download()

        # 2) Déterminer quel lot appartient à quel split
        self._determine_split()

        # 3) Déplacer les images
        if not (
            len(self._get_imgs_in_folder(os.path.join(self.dataset_dir, "train")))
            == self.splits.train
            and len(self._get_imgs_in_folder(os.path.join(self.dataset_dir, "val")))
            == self.splits.val
        ):
            self._execute_split()
        print("Mise à jour des datasets terminée.")

    def _determine_split(self):
        # 1) Déterminer à quel lot appartient chaque image
        imgs_labels = self._get_label_of_every_img()
        self.splits_df = pd.DataFrame.from_dict(imgs_labels, orient="index")
        self.splits_df = self.splits_df.sort_values(by="lot")
        self.splits_df["no lot"] = (
            self.splits_df["lot"] != self.splits_df["lot"].shift()
        ).cumsum()

        # 2) Déterminer y. Chaque y correspond à une tranche de 1/10 des valeurs de "trop petit"
        self.splits_df["fold"] = -1
        x = np.linspace(0, round(max(self.splits_df["% trop petit"]) / 10) * 10, 11)
        y = self.splits_df["no lot"].groupby(pd.cut(self.splits_df["% trop petit"], x))
        intervals_id = {}
        for i, interval in enumerate(y):
            for j in np.unique(interval[1].values):
                intervals_id[j] = i
        self.splits_df["y"] = self.splits_df["no lot"].map(intervals_id)
        self.splits_df.reset_index(inplace=True)
        least_represented_bin = min(self.splits_df["y"].value_counts())

        # 3) Stratifier les données en fonction de y et ne pas séparer les images d'un même lot
        stratifier = StratifiedGroupKFold(
            n_splits=least_represented_bin if least_represented_bin >= 3 else 2,
            shuffle=True,
            random_state=42,
        )
        for i, (_, fold_idxs) in enumerate(
            stratifier.split(
                self.splits_df["id"], self.splits_df["y"], self.splits_df["no lot"]
            )
        ):
            self.splits_df.loc[fold_idxs, ["fold"]] = i

        self.splits_df["stage"] = "train"
        self.splits_df.loc[
            self.splits_df.fold == 0, ["stage"]
        ] = "val"  # 1 fold de la gagne en validation
        del self.splits_df["fold"]
        self.splits = self.splits_df.stage.value_counts()
        print(
            f"{len(self.splits_df)} labels avec images. Sérapation train/val: {self.splits['train']}/{self.splits['val']}."
        )

    def _execute_split(self):
        # 1) Préparer les dossiers
        self._prepare_dirs()

        # 2) Déplacer les images et créer les labels
        images = self._get_imgs_in_folder(self.dataset_dir)
        image_without_label = 0
        while images:
            image = images.pop()
            img_split = self.splits_df.loc[self.splits_df["id"] == image]
            if len(img_split) == 0:
                image_without_label += 1
            elif img_split["stage"].values[0] == "val":
                self.val_labels.append(
                    self._move_img_n_create_label(
                        image,
                        "val",
                        img_split["% trop petit"].values[0],
                        img_split["% trop gros"].values[0],
                    )
                )
            else:
                self.train_labels.append(
                    self._move_img_n_create_label(
                        image,
                        "train",
                        img_split["% trop petit"].values[0],
                        img_split["% trop gros"].values[0],
                    )
                )

        # 3) S'assurer que tout est ok
        assert (
            len(self._get_imgs_in_folder(os.path.join(self.dataset_dir, "val")))
            == self.splits.val
        ), f"{len(self._get_imgs_in_folder(os.path.join(self.dataset_dir, 'val')))} != {self.splits.val}"
        assert (
            len(self._get_imgs_in_folder(os.path.join(self.dataset_dir, "train")))
            == self.splits.train
        ), f"{len(self._get_imgs_in_folder(os.path.join(self.dataset_dir, 'train')))} != {self.splits.train}"
        assert (
            len(self._get_imgs_in_folder(self.dataset_dir)) == image_without_label
        ), f"{len(self._get_imgs_in_folder(self.dataset_dir))} != {image_without_label}"

        # 4) Sauvegarder les labels
        for folder in ["val", "train"]:
            with open(
                os.path.join(self.dataset_dir, folder, f"{folder}_labels.json"), "w"
            ) as f:
                if folder == "train":
                    json.dump(self.train_labels, f)
                else:
                    json.dump(self.val_labels, f)

    def _get_label_of_every_img(self):
        images = self._get_imgs_in_folder(self.dataset_dir)
        img_labels = {}
        for i, image in enumerate(images):
            for label in self.labels:
                if re.fullmatch(rf"{label}_.*\.png", image) is not None:
                    img_labels[i] = {
                        "id": image,
                        "lot": label,
                        "% trop petit": self.labels[label]["% trop petit"],
                        "% trop gros": self.labels[label]["% trop gros"],
                    }
                    break
        return img_labels

    def _prepare_dirs(self):
        for folder in ["train", "val"]:
            path = os.path.join(self.dataset_dir, folder)
            if os.path.exists(path):
                # Supprimer les images et les labels précédents
                [f.unlink() for f in Path(path).glob("*") if f.is_file()]
            else:
                os.makedirs(os.path.join(self.dataset_dir, folder))

    def _move_img_n_create_label(
        self, image: str, dest_folder: str, trop_petit: float, trop_gros: float
    ):
        shutil.move(
            os.path.join(self.dataset_dir, image),
            os.path.join(self.dataset_dir, dest_folder, image),
        )
        return {"image": image, r"% trop petit": trop_petit, r"% trop gros": trop_gros}

    def _get_labels(self):
        with open(os.path.join(self.dataset_dir, "labels.json")) as json_file:
            labels = json.load(json_file)
        return labels

    @staticmethod
    def _get_imgs_in_folder(folder: str):
        return [img for img in os.listdir(folder) if img.endswith(".png")]


def image_is_in_labels(image: str, labels: dict):
    return any(extract_label_from_image(image) == label for label in labels)


def extract_label_from_image(image: str):
    match_ = re.match(r"^([^\_]*)(.+)$", image)
    if match_ is None:
        raise ValueError(f"L'image {image} n'a pas le bon format de nom.")
    return match_.groups()[0].strip()


def get_images_w_labels(images: list, labels: dict):
    return [image for image in images if image_is_in_labels(image, labels)]


def random_split(dataset: Dataset, train_pct: float):
    if not 0 < train_pct < 1:
        raise ValueError("train_size doit être entre 0 et 1.")
    train_size = math.floor(len(dataset) * train_pct)
    val_size = math.ceil(len(dataset) * (1 - train_pct))
    return random_split(dataset, [train_size, val_size])


def get_dataset_dir():
    return os.path.join(
        Path(os.path.abspath(__file__)).resolve().parents[1],
        "data",
        "grain_dataset",
    )


class GrainDataset(Dataset):
    """Le Pytorch Dataset pour les images et les labels downloadés par le DatasetsManager."""

    def __init__(self, imgsz: int, folder: str, transform: bool = None):
        self.dir = os.path.join(get_dataset_dir(), folder)
        self.transform = transform

        with open(os.path.join(self.dir, f"{folder}_labels.json"), "r") as f:
            self.labels = json.load(f)

        self.imgsz = imgsz

    @property
    def imgsz(self):
        return self._imgsz

    @imgsz.setter
    def imgsz(self, imgsz):
        if imgsz <= 0:
            raise ValueError("L'a taille des images doit être positive.")
        if not isinstance(imgsz, int):
            raise ValueError("La taille des images doit être un int.")
        self._imgsz = imgsz

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = self.labels[idx]

        image = cv2.imread(
            os.path.join(self.dir, label["image"]),
            cv2.IMREAD_COLOR,
        )

        image = torch.from_numpy(np.moveaxis(image, -1, 0)).float()
        image = transforms.Resize((self.imgsz, self.imgsz))(image)
        if self.transform:
            pass  # TODO: Ajouter des transformations

        too_small = torch.tensor(label["% trop petit"])
        too_big = torch.tensor(label["% trop gros"])

        return image, torch.stack((too_small, too_big), dim=0)


class SingleBatch:
    """Un Dataset qui load plusieurs fois la même batch pour pouvoir faire des tests plus rapidement."""

    def __init__(self, img_size: int, batch_size: int, transform: bool = None):
        self.dir = os.path.join(get_dataset_dir(), "train")
        self.transform = transform

        with open(os.path.join(self.dir, "train_labels.json"), "r") as f:
            self.labels = json.load(f)
        self.sample = random.sample(self.labels, batch_size)
        while len(self.sample) < len(self.labels):
            self.sample += self.sample
        self.imgsz = img_size

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = self.sample[idx]

        image = cv2.imread(
            os.path.join(self.dir, label["image"]),
            cv2.IMREAD_COLOR,
        )

        image = torch.from_numpy(np.moveaxis(image, -1, 0)).float()
        image = transforms.Resize((self.imgsz, self.imgsz))(image)
        if self.transform:
            pass

        return image, torch.tensor([label["% trop petit"], label["% trop gros"]])


def create_dataloaders(
    nb_workers: int,
    batch_size: int,
    image_size: int,
    overfit_batch: bool,
    update_datasets: bool,
    cuda: bool,
    **kwargs,
):
    """Crée les dataloaders pour le train et le val.

    Args:
        nb_workers (int): Le nombre de cpu cores à utiliser pour charger les données.
        batch_size (int): Le nombre d'images par batch.
        image_size (int): La taille des images (elles sont carrées).
        overfit_batch (bool): Si on veut overfitter sur une seule batch.
        update_datasets (bool): Si on veut download des nouvelles images du cloud et de re-split en train/val.
        cuda (bool): Si on est en gpu ou non.

    Returns:
        _type_: _description_
    """
    nw = min(nb_workers, batch_size, os.cpu_count())  # number of workers
    if update_datasets:
        datasets_manager = DatasetsManager()
        datasets_manager.update_datasets()
    if overfit_batch:
        val_dataset = SingleBatch(image_size, batch_size)
        train_dataset = SingleBatch(image_size, batch_size)
    else:
        val_dataset = GrainDataset(image_size, "val")
        train_dataset = GrainDataset(image_size, "train")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=nw,
        pin_memory=cuda,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=nw,
        pin_memory=cuda,
    )
    return train_loader, val_loader


if __name__ == "__main__":
    cfg = {
        "nb_workers": 4,
        "cuda": True,
        "update_datasets": True,
        "overfit_batch": True,
    }
    train_loader, val_loader = create_dataloaders(**cfg, batch_size=1, image_size=640)
    for images, labels in train_loader:
        image = np.moveaxis(images[0].numpy(), 0, -1)
        prcnt_trop_petit = labels[:, 0].item()
        prcnt_trop_gros = labels[:, 1].item()
        cv2.imshow(
            f"% trop petit: {prcnt_trop_petit}, % trop gros: {prcnt_trop_gros}",
            image.astype(np.uint8),
        )
        cv2.waitKey(0)
        break
