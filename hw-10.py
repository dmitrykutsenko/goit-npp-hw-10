import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

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

#6. Побудова діаграми розподілу числових ознак

# Формування підмножини числових ознак
numeric_features = [
    "Score",
    "GDP per capita",
    "Social support",
    "Healthy life expectancy",
    "Freedom to make life choices",
    "Generosity",
    "Perceptions of corruption"
]

df_num = df_all[numeric_features]
df_num.head()

# Кореляційна матриця
plt.figure(figsize=(10, 7))
sns.heatmap(df_num.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Кореляційна матриця числових ознак (усі роки)")
plt.show()


"""
# 8. Теплова мапа для всіх років
# df_all має містити колонки: Country, Year, Score
pivot = df_all.pivot_table(
    index="Country or region",
    columns="Year",
    values="Score"
)

plt.figure(figsize=(14, 22))
sns.heatmap(
    pivot,
    cmap="viridis",
    linewidths=0.5,
    linecolor="gray"
)

plt.title("Розподіл Happiness Score за країнами та роками")
plt.xlabel("Year")
plt.ylabel("Country")
plt.show()
"""
# 8.1. Географічна теплова мапа для всіх років

years = sorted(df_all["Year"].unique())

for y in years:
    df_y = df_all[df_all["Year"] == y]
    fig = px.choropleth(
        df_y,
        locations="Country or region",
        color="Score",
        locationmode="country names",
        color_continuous_scale="Viridis",
        title=f"Happiness Score ({y})"
    )
    fig.show()

# 8.2. Усі години на одному графіку, з анімацією
fig_ani = px.choropleth(
    df_all,
    locations="Country or region",
    color="Score",
    locationmode="country names",
    animation_frame="Year",
    color_continuous_scale="Viridis",
    title="Happiness Score by Country (Animated)"
)
fig_ani.show()

# 8.3. Усі години на одному графіку, але замість рівня щастя місце в рейтингу
fig_rank = px.choropleth(
    df_all,
    locations="Country or region",
    color="Overall rank",
    locationmode="country names",
    animation_frame="Year",
    color_continuous_scale="Magma_r",
    title="Happiness Rank by Country (Animated)"
)
fig_rank.show()


# 9.1. Вибір числових ознак
numeric_cols = [
    "Score",
    "GDP per capita",
    "Social support",
    "Healthy life expectancy",
    "Freedom to make life choices",
    "Generosity",
    "Perceptions of corruption"
]

original_dataframe = df_all[numeric_cols]
original_dataframe.head()

# 9.2.Функція масштабування
def data_scale(data, scaler_type='minmax'):
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import Normalizer

    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    if scaler_type == 'std':
        scaler = StandardScaler()
    if scaler_type == 'norm':
        scaler = Normalizer()

    scaler.fit(data)
    res = scaler.transform(data)
    return res

# 9.3.
data_scaled = data_scale(original_dataframe, scaler_type='std')

df_scaled = pd.DataFrame(data_scaled, columns=original_dataframe.columns)
df_scaled.head()

