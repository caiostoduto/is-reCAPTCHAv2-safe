import requests
import io
import zipfile
from pathlib import Path


class DatasetDownloader:
  dataset_path: Path

  def __init__(self, dataset_path: str):
    self.dataset_path = Path(dataset_path)

  def download_all(self):
    self.__github("AdityaJain1030/recaptcha-dataset")
    self.__github("nobodyPerfecZ/recaptchav2-29k")
    self.__kaggle("cry2003/google-recaptcha-v2-images")
    self.__kaggle("mikhailma/test-dataset")

  def __kaggle(self, handler: str):
    if (self.dataset_path / handler.split("/")[-1]).exists():
      print(f"Dataset {handler} already exists. Skipping download.")
      return

    print("Downloading dataset from Kaggle...")
    res = requests.get(f"https://www.kaggle.com/api/v1/datasets/download/{handler}")
    self.__unzip_bytes_io(res, handler)

  def __github(self, handler: str):
    if (self.dataset_path / handler.split("/")[-1]).exists():
      print(f"Dataset {handler} already exists. Skipping download.")
      return

    print("Downloading dataset from GitHub...")
    res = requests.get(f"https://github.com/{handler}/archive/refs/heads/master.zip")
    self.__unzip_bytes_io(res, handler)

  def __unzip_bytes_io(self, res: requests.Response, handler: str):
    res.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(res.content)) as zf:
      zf.extractall(self.dataset_path / handler.split("/")[-1])