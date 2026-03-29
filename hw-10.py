import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


# df = pd.read_csv("World Happiness Report/2019.csv")

# Об'єднаємо усі роки в один дата фрейм

data_path = "World Happiness Report"

def unify_columns(df):
    df = df.rename(columns={
        "Happiness Score": "Score",
        "Economy (GDP per Capita)": "GDP per capita",
        "Family": "Social support",
        "Health (Life Expectancy)": "Healthy life expectancy",
        "Freedom": "Freedom to make life choices",
        "Trust (Government Corruption)": "Perceptions of corruption",
        "Happiness Rank": "Overall rank",
        "Country": "Country or region"
    })
    return df


frames = []

for file in os.listdir(data_path):
    if file.endswith(".csv") and file[:4].isdigit():
        year = int(file[:4])
        df = pd.read_csv(os.path.join(data_path, file))
        df = unify_columns(df)
        df["Year"] = year
        frames.append(df)

df_all = pd.concat(frames, ignore_index=True)

df_all[["Score","GDP per capita","Social support",
        "Healthy life expectancy","Freedom to make life choices",
        "Generosity","Perceptions of corruption"]].isna().sum()

from sklearn.impute import SimpleImputer

numeric_cols = ["Score","GDP per capita","Social support",
                "Healthy life expectancy","Freedom to make life choices",
                "Generosity","Perceptions of corruption"]

imputer = SimpleImputer(strategy="mean")
data_imputed = imputer.fit_transform(df_all[numeric_cols])
df_imputed = pd.DataFrame(data_imputed, columns=numeric_cols)


# Оскільки я вже об’єднав усі CSV у df_all, то статистика для всього набору виглядає так:

df_all.head()

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


# 5. Побудова діаграм розподілу

# Визначення числових ознак
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
data_scaled = data_scale(df_imputed, scaler_type='std')
df_scaled = pd.DataFrame(data_scaled, columns=numeric_cols)

df_scaled.head()


# 10.1. Оригінальні дані
print("=== ОРИГІНАЛЬНІ СТАТИСТИКИ ===")
print(original_dataframe.describe())

# 10.2. Стандартизовані дані
print("\n=== СТАНДАРТИЗОВАНІ СТАТИСТИКИ ===")
print(df_scaled.describe())

# 10.3. Table format
orig_stats = original_dataframe.describe().T
scaled_stats = df_scaled.describe().T

comparison = pd.concat([orig_stats, scaled_stats], axis=1, keys=["Original", "Scaled"])
comparison


# 11.1. Побудова моделі GMM

# Кількість кластерів (можна змінювати)
n_clusters = 3

# Створення моделі
gmm = GaussianMixture(
    n_components=n_clusters,
    covariance_type='full',
    random_state=42
)

# Навчання моделі
gmm.fit(df_scaled)

# Прогнозування кластерів
clusters = gmm.predict(df_scaled)

# Додавання кластерів у DataFrame
df_clusters = original_dataframe.copy()
df_clusters["Cluster"] = clusters

df_clusters.head()


# 12

# Переконаємося, що Country or region є в df_clusters
df_clusters["Country or region"] = df_all["Country or region"].values

# Створення матриці: країна → кластер
pivot = df_clusters.pivot_table(
    index="Country or region",
    values="Cluster",
    aggfunc="first"   # бо кожна країна має один кластер
)

plt.figure(figsize=(10, 20))
sns.heatmap(
    pivot,
    cmap="viridis",
    linewidths=0.5,
    linecolor="gray",
    annot=True,
    fmt="d"
)

plt.title("Розподіл країн за кластерами (GMM)")
plt.xlabel("Cluster")
plt.ylabel("Country")
plt.show()


# 13.1. Приклад коду для порівняння різних наборів ознак

feature_sets = {
    "economic": ["GDP per capita"],
    "social": ["Social support"],
    "health": ["Healthy life expectancy"],
    "institutional": ["Freedom to make life choices", "Perceptions of corruption"],
    "behavioral": ["Generosity"],
    "core_3": ["GDP per capita", "Social support", "Healthy life expectancy"],
    "full": ["GDP per capita", "Social support", "Healthy life expectancy",
             "Freedom to make life choices", "Generosity", "Perceptions of corruption"]
}

results = {}

for name, features in feature_sets.items():
    data = df_all[features].copy()
    data = data.fillna(data.mean())
    scaled = StandardScaler().fit_transform(data)

    gmm = GaussianMixture(n_components=3, random_state=42)
    clusters = gmm.fit_predict(scaled)

    results[name] = clusters


# 13.2. Порівняння складу кластерів

# Створюється таблиця, де видно, які країни змінюють кластер при зміні ознак
comparison = pd.DataFrame({
    name: results[name]
    for name in results
})
comparison["Country"] = df_all["Country or region"].values
comparison

