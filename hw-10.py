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

df_all.head ()

# Оскільки я вже об’єднав усі CSV у df_all, то статистика для всього набору виглядає так:

# Інформація про типи ознак

#Цей метод показує:
#  кількість рядків і колонок
#  типи даних
#  кількість пропусків

df_all.info()

# Описова статистика

# Для числових ознак:
df_all.describe()

# Для всіх ознак, включно з категоріальними:
df_all.describe(include="all")

