# -*- coding: utf-8 -*-
"""

Main script pour l'analyse de churn client (2021–2024)

"""



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pipeline import *

# 1. Chargement et nettoyage des données
print("📥 Chargement et nettoyage des données...")
df = load_and_clean_data("sales_2021_2024.xlsx")

# 2. Filtrer les clients actifs en 2021–2022
clients_periode1 = df[df['Year'].isin([2021, 2022])]['CustNum'].unique()
df_filtered = df[df['CustNum'].isin(clients_periode1)].copy()
print(f"✅ Clients actifs sur 2021–2022 : {len(clients_periode1)}")

# Sauvegarde du dataset nettoyé
df_filtered.to_excel("sales_cleaned.xlsx", index=False)

# 3. Détection du churn
print("🔍 Détection du churn...")
churned_clients = detect_churned_clients(df_filtered)  # Ajoute déjà la colonne 'churner'
print(f"⚠️ Clients churners identifiés : {len(churned_clients)}")

# Sauvegarde avec churn
df_filtered.to_excel("sales_with_churn.xlsx", index=False)

# 4. Taux de churn global
churn_rate = compute_churn_rate(df_filtered, churned_clients)
print(f"📊 Taux global de churn : {churn_rate:.2%}")

# 5. Visualisation : Chiffre d'affaires par Segment et Année

import matplotlib.ticker as mtick

# Fonction de formatage des labels en millions
def format_millions(x, pos):
    if x >= 1_000_000:
        return f'{x/1_000_000:.1f}M'.replace('.', ',')  # Remplace le point par une virgule si souhaité
    else:
        return f'{x:.0f}'

print("📈 Visualisation des ventes par segment...")

sales_segment_year = df_filtered.groupby(['Year', 'Segment'])['TotalSalesValueLocal'].sum().reset_index()

plt.figure(figsize=(10, 6))
ax = sns.barplot(data=sales_segment_year, x='Year', y='TotalSalesValueLocal', hue='Segment', palette='Set2')
plt.title('Chiffre d’affaires par Segment et par Année')
plt.ylabel('Total des ventes (€)')
plt.xlabel('Année')

# Déplacer la légende à droite, en dehors du plot
plt.legend(title='Segment', bbox_to_anchor=(1.05, 1), loc='upper left')

# Appliquer le format personnalisé aux labels des barres
for container in ax.containers:
    labels = [format_millions(val.get_height(), None) for val in container]
    ax.bar_label(container, labels=labels, label_type='edge', padding=3)

# Appliquer le format au tick de l'axe y
ax.yaxis.set_major_formatter(mtick.FuncFormatter(format_millions))

plt.tight_layout(rect=[0, 0, 0.85, 1])  # Ajuste la marge droite pour la légende
plt.show()


# 6. Camembert des segments 2021–2022
print("📊 Répartition des clients par segment...")
df_seg = df_filtered[df_filtered['Year'].isin([2021, 2022])]
segment_dist = df_seg.groupby('Segment')['CustNum'].nunique().reset_index()
segment_dist.columns = ['Segment', 'Clients']
plt.figure(figsize=(8, 8))
plt.pie(segment_dist['Clients'], labels=segment_dist['Segment'], autopct='%1.1f%%', startangle=140)
plt.title("Répartition des clients par segment (2021–2022)")
plt.axis('equal')
plt.tight_layout()
plt.show()

# 7. Tableau récapitulatif par segment

# Créer un DataFrame client-level (une ligne par client) avec la somme du CA par client
df_clients = df_filtered.groupby('CustNum').agg({
    'churner': 'max',  # 1 si le client a churné
    'Segment': lambda x: x.mode().iloc[0] if not x.mode().empty else None,
    'Typology': lambda x: x.mode().iloc[0] if not x.mode().empty else None,
    'TotalSalesValueLocal': 'sum' # somme du CA par client
}).reset_index()

# Tableau récapitulatif par segment
summary = df_clients.groupby('Segment').agg(
    TotalClients=('CustNum', 'count'),
    ChurnedClients=('churner', 'sum'),
    TopTypology=('Typology', lambda x: x.mode().iloc[0] if not x.mode().empty else None),
    ChurnersTotalTO=('TotalSalesValueLocal', lambda x: x[df_clients.loc[x.index, 'churner'] == 1].sum())
).reset_index()

summary['ChurnRate (%)'] = 100 * summary['ChurnedClients'] / summary['TotalClients']

print("📄 Résumé du churn par segment (client-level) :")
print(summary)

summary.to_excel("churn_summary_by_segment.xlsx", index=False)



# 8. Churn par Compagnie

# Identifier les churners par client
churn_status = df_filtered.groupby(['Company', 'CustNum'])['churner'].max().reset_index()

# Calcul du churn par compagnie
churn_summary = churn_status.groupby('Company').agg(
    total_clients=('CustNum', 'nunique'),
    churned_clients=('churner', 'sum')
).reset_index()

churn_summary['churn_rate'] = 100 * churn_summary['churned_clients'] / churn_summary['total_clients']
churn_summary = churn_summary.sort_values(by='churn_rate', ascending=False)

# Affichage du graphique
plt.figure(figsize=(10, 6))
ax = sns.barplot(data=churn_summary, x='Company', y='churn_rate', palette='coolwarm')
plt.title('Taux de churn par Compagnie')
plt.ylabel('Churn rate (%)')
plt.xlabel('Company')
plt.xticks(rotation=45, ha='right')

# Ajout des étiquettes au-dessus des barres
for bar in ax.patches:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords='offset points',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()


# Regroupe les données par client sur 2021–2022
df_agg = aggregate_features_for_pca(df_filtered)


# 9. Analyse Top/Bottom clients (2021–2022)
print("🏆 Analyse des top et bottom clients...")
df_clients_agg = aggregate_client_data(df_filtered)
df_clients_agg['AvgOrderValue'] = df_clients_agg['TotalSalesValueLocal'] / df_clients_agg['NumberOfOrders']

top_5 = df_clients_agg.sort_values(by='AvgOrderValue', ascending=False).head(5)
bottom_5 = df_clients_agg.sort_values(by='AvgOrderValue', ascending=True).head(5)

print("\n🎯 Top 5 clients par AvgOrderValue :")
print(top_5[['CustNum', 'AvgOrderValue', 'Segment', 'Typology', 'churner']])

print("\n⚠️ Bottom 5 clients par AvgOrderValue :")
print(bottom_5[['CustNum', 'AvgOrderValue', 'Segment', 'Typology', 'churner']])

top_5.to_excel("top_5_clients.xlsx", index=False)
bottom_5.to_excel("bottom_5_clients.xlsx", index=False)

# 10. Visualisation : Boxplot AvgOrderValue vs Churn
filtered_plot = filter_for_plot(df_filtered, 'AvgOrderValue', quantile_limit=(0.01, 0.99))

plt.figure(figsize=(10, 6))
sns.boxplot(data=filtered_plot, x='churner', y='AvgOrderValue', palette='Set2')
plt.title('Distribution de AvgOrderValue selon le statut de churn (hors extrêmes)')
plt.xlabel('Statut de churn')
plt.ylabel('AvgOrderValue')
plt.tight_layout()
plt.show()

# 10. Visualisation : Boxplot AvgOrderValue vs Segment
filtered_plot = filter_for_plot(df_filtered, 'AvgOrderValue', quantile_limit=(0.01, 0.99))

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_agg, x='Segment', y='AvgOrderValue', palette='Set2')
plt.title('Distribution de AvgOrderValue selon le segment')
plt.xlabel('Segmentation')
plt.ylabel('AvgOrderValue')
plt.tight_layout()
plt.show()

# 11. Visualisation : SalesGrowthYoY vs Churn
filtered_plot = filter_for_plot(df_filtered, 'SalesGrowthYoY', lower=-1, upper=2)

plt.figure(figsize=(10, 6))
sns.boxplot(data=filtered_plot, x='churner', y='SalesGrowthYoY', palette='Set3')
plt.title('Distribution de SalesGrowthYoY selon le statut de churn (hors valeurs extrêmes)')
plt.xlabel('Statut de churn')
plt.ylabel('SalesGrowthYoY')
plt.tight_layout()
plt.show()


print("✅ Analyse terminée.")



# # Recode des délais en catégories (-1, 0, 1)
# df = recode_delivery_vars(df)


###################  ACP

from pipeline import aggregate_features_for_pca

# Regroupe les données par client sur 2021–2022
df_agg_pca = aggregate_features_for_pca(df_filtered)

# Rattache le statut de churn pour chaque client
df_churn = df_filtered[['CustNum', 'churner']].drop_duplicates()
df_agg_pca = df_agg_pca.merge(df_churn, on='CustNum', how='left')

from sklearn.preprocessing import StandardScaler

# Variables numériques pertinentes pour l’ACP
features = [
    'NumberOfOrders',
    'OrderToInvoiceDelayPerOrder',
    'DeliveryGapPerOrder',
    'ShippingDelayPerOrder',
    'AvgOrderValue',
    'SalesGrowthYoY',
    'NumberOfVisits',
    'AVGLineUnitCost'
]

# Standardisation des variables (centrage-réduction)
X = df_agg_pca[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y = df_agg_pca['churner']

#from pipeline_opt import run_pca

# ACP sur les données standardisées
pca, X_pca = run_pca(X_scaled)

# Ajout des deux premières composantes principales
df_agg_pca['PC1'] = X_pca[:, 0]
df_agg_pca['PC2'] = X_pca[:, 1]

from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

# Regroupe les clients en 3 groupes selon leur projection ACP
kmeans = KMeans(n_clusters=3, random_state=42).fit(X_pca[:, :2])
df_agg_pca['Cluster'] = kmeans.labels_

# Affiche la projection sur PC1/PC2, colorée par cluster et forme selon churn
plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=df_agg_pca, x='PC1', y='PC2',
    hue='Cluster', style='churner',
    palette='viridis', s=70
)
plt.title("ACP - Segmentation non supervisée des clients")
plt.grid(True)
plt.tight_layout()
plt.show()

# Calcul des coefficients de contribution (loadings)
import numpy as np
loadings = pd.DataFrame(
    pca.components_.T * np.sqrt(pca.explained_variance_),
    columns=[f"PC{i+1}" for i in range(pca.n_components_)],
    index=features
)

# Cercle de corrélation sur PC1/PC2
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=df_agg_pca, x='PC1', y='PC2',
    hue='churner', palette={0: 'skyblue', 1: 'crimson'}, s=80
)

# Ajout des vecteurs de contribution des variables
for i, feature in enumerate(features):
    plt.arrow(0, 0,
              loadings.loc[feature, 'PC1'] * 8,
              loadings.loc[feature, 'PC2'] * 8,
              color='black', alpha=0.5, head_width=0.3)
    plt.text(loadings.loc[feature, 'PC1'] * 8.5,
             loadings.loc[feature, 'PC2'] * 8.5,
             feature, color='darkgreen', fontsize=9)

plt.title("ACP - Représentation des clients et contributions des variables")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.tight_layout()
plt.show()

# Affichage des contributions des variables aux deux premières composantes
print(loadings[['PC1', 'PC2']].sort_values(by='PC1', ascending=False))







##################  Modele logit

# 
df_agg_logit = aggregate_features_for_logit(df_filtered)

print(df_agg_logit.columns)

corr_matrix = plot_correlation_matrix(df_agg_logit, target_column='churner')


# Lancer le modèle logit
# print("Estimation du modèle logit...")
# logit_model = run_logit_model(df_agg2)
# print(logit_model.summary())

# # Mode normal
# model = run_logit_model(df_agg2)

# Mode debug pour analyser les problèmes
try:
    model, X, y = run_logit_model(df_agg_logit, debug=True)
    plot_roc_auc(model, X, y)

    print(model.summary())
except Exception as e:
    print("Échec de l'analyse:", e)

from statsmodels.stats.outliers_influence import variance_inflation_factor

variables_optimisees = [
    'NumberOfOrders',
    'OrderToInvoiceDelayPerOrder',    # Fort impact (r=0.46 avec CA)
    'DeliveryGapPerOrder',       # Anti-corrélé avec délais facturation
    'ShippingDelayPerOrder',     # Lié au nombre de commandes
    'AvgOrderValue',          # Capture la valeur client
    'NumberOfVisits',
    'SalesGrowthYoY',
    'CompGI_Justified',
    'Comp_GI_NotJustified',
    'CompPck_Justified',
    'CompPhy_Justified',
    'CompPhy_NotJustified',
    'CompSQ_Justified',
    'CompSQ_NotJustified',
    'CompST_Justified',
    'CompST_NotJustified',          # Corrélé au CA (r=0.26)
]
X = df_filtered[variables_optimisees]
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)


















