from pathlib import Path
import pandas as pd
import os

def load_datasets(datasets_path_str: str):
  path = Path(datasets_path_str)
  
  df_dict = {
    "Image": [],
    "Filename": [],
    "Label": [],
    "From": []
  }

  # google-recaptcha-v2-images
  rel_path = path / "google-recaptcha-v2-images/images"
  for category in os.listdir(rel_path):
    if not os.path.isdir(rel_path / category):
      continue

    for image_filename in os.listdir(rel_path / category):
      df_dict["Image"].append(str(rel_path / category / image_filename))
      df_dict["Filename"].append(image_filename)
      df_dict["Label"].append(category)
      df_dict["From"].append("google-recaptcha-v2-images")

  # recaptcha-dataset (Training)
  rel_path = path / "recaptcha-dataset/recaptcha-dataset-master/Training"
  for category in os.listdir(rel_path):
    if not os.path.isdir(rel_path / category):
      continue

    for image_filename in os.listdir(rel_path / category):
      df_dict["Image"].append(str(rel_path / category / image_filename))
      df_dict["Filename"].append(image_filename)
      df_dict["Label"].append(category)
      df_dict["From"].append("recaptcha-dataset")

  # recaptcha-dataset (Validation)
  rel_path = path / "recaptcha-dataset/recaptcha-dataset-master/Validation"
  for category in os.listdir(rel_path):
    if not os.path.isdir(rel_path / category):
      continue

    for image_filename in os.listdir(rel_path / category):
      df_dict["Image"].append(str(rel_path / category / image_filename))
      df_dict["Filename"].append(image_filename)
      df_dict["Label"].append(category)
      df_dict["From"].append("recaptcha-dataset")

  # recaptchav2-29k
  rel_path = path / "recaptchav2-29k/recaptchav2-29k-master/data"
  for category in os.listdir(rel_path):
    if not os.path.isdir(rel_path / category):
      continue

    for image_filename in os.listdir(rel_path / category):
      df_dict["Image"].append(str(rel_path / category / image_filename))
      df_dict["Filename"].append(image_filename)
      df_dict["Label"].append(category)
      df_dict["From"].append("recaptchav2-29k")

  # test-dataset
  rel_path = path / "test-dataset/Google_Recaptcha_V2_Images_Dataset/images"
  for category in os.listdir(rel_path):
    if not os.path.isdir(rel_path / category):
      continue

    for image_filename in os.listdir(rel_path / category):
      df_dict["Image"].append(str(rel_path / category / image_filename))
      df_dict["Filename"].append(image_filename)
      df_dict["Label"].append(category)
      df_dict["From"].append("test-dataset")

  df = pd.DataFrame(df_dict)

  df.drop_duplicates(subset=['Filename'], inplace=True)
  df[df['Label'] == "TLight"] = "Traffic Light"
  df['Label'] = df['Label'].str.title()

  df.to_parquet(path / "datasets.parquet", index=False)

  return df


if __name__ == "__main__":
  df = load_datasets("datasets")

  print(df['Label'].unique())

  print(df.head())
  print(df[['Label']].value_counts())
  print(df['From'].value_counts())
  print(f"Total images loaded: {len(df)}")
