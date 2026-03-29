import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns


# df = pd.read_csv("World Happiness Report/2019.csv")

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

# Оскільки я вже об’єднав усі CSV у df_all, то статистика для всього набору виглядатиме так:

# 4. Інформація про типи ознак

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


# 5. Побудова діаграм розподілу

# Визначення числових колонок
numeric_cols = df_all.select_dtypes(include=["int64", "float64"]).columns
print("Числові ознаки:", list(numeric_cols))

# Побудова гістограм
plt.figure(figsize=(16, 12))

for i, col in enumerate(numeric_cols, 1):
    plt.subplot(len(numeric_cols) // 3 + 1, 3, i)
    sns.histplot(df_all[col].dropna(), kde=True, bins=20, color="royalblue")
    plt.title(col)

plt.tight_layout()
plt.show()

