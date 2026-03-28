import pandas as pd
import os

df = pd.read_csv("World Happiness Report/2019.csv")

# Об'єднаємо усі роки в один дата фрейм

data_path = "World Happiness Report"

frames = []

for file in os.listdir(data_path):
    if file.endswith(".csv"):
        year = int(file.replace(".csv", ""))
        df = pd.read_csv(os.path.join(data_path, file))
        df["Year"] = year
        frames.append(df)

df_all = pd.concat(frames, ignore_index=True)

df_all.info()
