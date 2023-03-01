import os
from pathlib import Path
from azure.storage.blob import BlobServiceClient
from tqdm import tqdm
import pandas as pd
import re
import json


class CloudDownloader(object):
    def __init__(
        self,
        dataset_dir: str,
        account_url: str = "https://saagriiatremblaypoc.blob.core.windows.net",
        access_key: str = "kqc1JzwP0R0JFPwfEB+NMvkEbtSGnqEb6NYgMIvZPttX43mPEZEPGDk1letdtXtbC6E09mx8k26x+ASttkLVrQ==",
    ) -> None:

        self.blob_service_client = BlobServiceClient(
            account_url=account_url, credential=access_key
        )
        self.images_container = self.blob_service_client.get_container_client("images")
        self.labels_container = self.blob_service_client.get_container_client("labels")
        self.dataset_dir = dataset_dir

    def download(self) -> None:
        # 1 - Downloader les images
        self._download_images()

        # 2 - Downloader les labels et les requis en DataFrame
        blobs = self.labels_container.list_blobs()
        for blob in blobs:
            if re.fullmatch(".+\.xlsx$", blob.name) is None:
                print(f"Le Blob {blob.name} n'est pas un excel.")
            else:
                df = self._download_xlsx(blob)
                if re.fullmatch("^.+requis.+\.xlsx$", blob.name) is not None:
                    self.df_requis = df
                elif re.fullmatch("^.+r(e|é)sultats.+\.xlsx$", blob.name) is not None:
                    self.df_labels = df
                else:
                    print(f"Le Blob {blob.name} n'est pas reconnus.")

        # 3 - Créer la liste des labels
        json_labels = self._format_labels()

        # 4 - Sauvegarder les labels
        with open(os.path.join(self.dataset_dir, "labels.json"), "w") as file:
            json.dump(json_labels, file)

    def _download_images(self):
        blob_images = self.images_container.list_blobs()
        local_folder = list(os.listdir(self.dataset_dir))
        nb_new_images = 0
        for blob in tqdm(blob_images, desc="Téléchargement des images"):
            if blob.name not in local_folder:
                blob_bytes = (
                    self.images_container.get_blob_client(blob)
                    .download_blob()
                    .readall()
                )
                self._download_blob(blob.name, blob_bytes)
                nb_new_images += 1
        print(f"{nb_new_images} images ont été téléchargées.")

    def _format_labels(self):
        json_labels = {
            str(lot): {"classe": ["0", "0"]}
            for lot in self.df_labels.iloc[:, 0].unique()
        }
        for label in self.df_labels.to_numpy():
            lot_id = str(label[0])
            json_labels[lot_id]["designation"] = label[1]
            json_labels[lot_id]["article"] = label[3]
            requis_du_lot = self._get_requis_of_lot(str(label[3]))
            if len(requis_du_lot) == 0:
                json_labels[lot_id]["classe"] = {}  # Classe inconnue

            resultat = self._get_prct(label[9])

            # 1 - Si le résultat représente le % de gros
            if re.fullmatch(r"^(?=.*\s>\s)(?!.*\s<\s).*$", label[8]) is not None:
                json_labels[lot_id][r"% trop gros"] = resultat
                if len(requis_du_lot) > 0:
                    if resultat < requis_du_lot[0][0] or resultat > requis_du_lot[0][1]:
                        json_labels[lot_id]["classe"][0] = "1"
            # 2 - Si le résultat représente le % de petit
            elif re.fullmatch(r"^(?!.*\s>\s)(?=.*\s<\s).*$", label[8]) is not None:
                json_labels[lot_id][r"% trop petit"] = resultat
                if len(requis_du_lot) > 0:
                    if resultat < requis_du_lot[2][0] or resultat > requis_du_lot[2][1]:
                        json_labels[lot_id]["classe"][1] = "1"
            # 3 - Si le résultat représente le % de moyen
            elif re.fullmatch(r"^(?=.*\s>\s)(?=.*\s<\s).*$", label[8]) is not None:
                continue
            else:
                print(
                    f"La Désignation caractéristique de contrôle {label[8]} n'est pas reconnue."
                )
                break
        # 4 - Modifier la classe en fonction des résultats
        with open(os.path.join(self.dataset_dir, "classes.json")) as json_file:
            classes = json.load(json_file)
        for label in json_labels:
            if len(json_labels[label]["classe"]) > 0:
                json_labels[label]["classe"] = classes[
                    "".join(json_labels[label]["classe"])
                ]

        return json_labels

    def _get_requis_of_lot(self, no_article_of_lot: str):
        requis = self.df_requis
        requis_du_lot = requis.loc[requis.Article == no_article_of_lot]
        if len(requis_du_lot) == 0:
            no_article_of_lot = re.sub(r"(\s)?-5001(\s?)", "", no_article_of_lot)
            requis_du_lot = requis.loc[requis.Article == no_article_of_lot]
        if len(requis_du_lot) == 0:
            no_article_of_lot = f"{no_article_of_lot}-5001"
            requis_du_lot = requis.loc[requis.Article == no_article_of_lot]
        if len(requis_du_lot) == 0:
            print(f"ATTENTION: Le lot {no_article_of_lot} n'a pas de requis.")
        return requis_du_lot.iloc[:, -2:].to_numpy()

    def _download_xlsx(self, blob):
        blob_bytes = (
            self.labels_container.get_blob_client(blob).download_blob().readall()
        )
        self._download_blob(blob.name, blob_bytes)
        return pd.read_excel(os.path.join(self.dataset_dir, blob.name))

    def _download_blob(self, file_name, file_content):
        download_file_path = os.path.join(self.dataset_dir, file_name)
        os.makedirs(os.path.dirname(download_file_path), exist_ok=True)
        with open(download_file_path, "wb") as file:
            file.write(file_content)

    @staticmethod
    def _get_prct(lot_result: str):
        try:
            resultat_reel = float(lot_result)
        except ValueError as e:
            lot_result = re.sub(r"(\s)?,(\s)?", ".", lot_result)
            resultat_reel = float(lot_result)
        return resultat_reel


if __name__ == "__main__":
    cloud_downloader = CloudDownloader()
    cloud_downloader.download()
