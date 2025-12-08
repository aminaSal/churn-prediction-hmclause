# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 13:59:51 2025

@author: Aminata.SALL
"""

import pandas as pd

def load_and_clean_data(filepath):
    df = pd.read_excel(filepath)
    df['TotalSalesValueLocal'] = df['TotalSalesValueLocal'].astype(float)
    df = df[df['TotalSalesValueLocal'] > 0]
    df['Year'] = df['Year'].astype(int)
    df['CustNum'] = df['CustNum'].astype(str)
    df = df[df['NumberOfOrders'] > 0]
    df['AvgOrderValue'] = df['TotalSalesValueLocal'] / df['NumberOfOrders']
    df = df.sort_values(['CustNum', 'Year'])
    df['SalesGrowthYoY'] = df.groupby('CustNum')['TotalSalesValueLocal'].pct_change().fillna(0)
    
    # Colonnes à corriger
    delay_columns = ['OrderToInvoiceDelayPerOrder', 'DeliveryGapPerOrder', 'ShippingDelayPerOrder']
    
    # Inverser les signes
    df[delay_columns] = df[delay_columns] * -1
    
    df = df[df['SalesGrowthYoY'] < 1000]  # ou fixer un seuil raisonnable
    
    # colonnes complaints à calculer par commande
    import numpy as np

    # Liste des colonnes de plaintes
    complaints_columns = ['CompAP_Justified', 'CompAP_NotJustified', 'CompGP_Justified',
        'CompGP_NotJustified', 'CompGI_Justified', 'Comp_GI_NotJustified',
        'CompPck_Justified', 'CompPP_Justified', 'CompPP_NotJustified',
        'CompPhy_Justified', 'CompPhy_NotJustified', 'CompSOM_Justified',
        'CompSQ_Justified', 'CompSQ_NotJustified', 'CompST_Justified', 'CompST_NotJustified']
    
    # Vérifier que toutes les colonnes existent
    existing_cols = [col for col in complaints_columns if col in df.columns]
    
    # Remplacer les zéros par NaN pour éviter la division par zéro
    df['NumberOfOrders'] = df['NumberOfOrders'].replace(0, np.nan)
    
    # Diviser les colonnes existantes par le nombre de commandes
    df[existing_cols] = df[existing_cols].div(df['NumberOfOrders'], axis=0)

    
    # Variable d'interaction : Compagnie * Typologie
    #df['Company_Typology'] = df['Company'].astype(str) + "_" + df['Typology'].astype(str)
    
    # Variable d'interaction : nombre de commande * valeur moyenne par commande
    df['AvgOrderValue_NBOrders'] = df['AvgOrderValue'] * df['NumberOfOrders']
   
    
    # # Regrouper certaines typologies en 'Other Typology'
    # other_typologies = [
    #     'Local shops',
    #     'Processors',
    #     'Traders',
    #     'Local Brokers',
    #     'Non Commercial',
    #     'Central markets'
    # ]
    
    # df['Typology'] = df['Typology'].replace(other_typologies, 'Other Typology')
    return df




def filter_for_plot(df, column, lower=None, upper=None, quantile_limit=None):
    """
    Filtre une colonne pour les visualisations uniquement (boxplots, etc.)
    - column : nom de la colonne à filtrer
    - lower, upper : bornes manuelles
    - quantile_limit : tuple (min_quantile, max_quantile) si tu veux utiliser les percentiles
    """
    if quantile_limit:
        lower = df[column].quantile(quantile_limit[0])
        upper = df[column].quantile(quantile_limit[1])
    
    return df[(df[column] >= (lower if lower is not None else -float('inf'))) &
              (df[column] <= (upper if upper is not None else float('inf')))]



def detect_churned_clients(df):
    churned_clients = []
    grouped = df.groupby('CustNum')
    for cust_id, group in grouped:
        years = set(group['Year'].unique())
        if (2021 in years or 2022 in years) and not (2023 in years or 2024 in years):
            churned_clients.append(cust_id)
    
    df['churner'] = df['CustNum'].isin(churned_clients).astype(int)  # 1 pour churner, 0 sinon
    return churned_clients

def compute_churn_rate(df_filtered, churned_clients):
    nb_clients = df_filtered['CustNum'].nunique()
    nb_churned = len(churned_clients)
    return nb_churned / nb_clients

def aggregate_client_data(df_filtered):
    return df_filtered[df_filtered['Year'].isin([2021, 2022])].groupby('CustNum').agg({
        'TotalSalesValueLocal': 'sum',
        'NumberOfOrders': 'sum',
        'Segment': 'first',
        'Typology': 'first',
        'churner': 'first',
        'CustName': 'first'
    }).reset_index()


# ACP

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# def recode_delivery_vars(df):
#     """
#     Recode les variables de délai en catégories ordinales : -1 (avance), 0 (à l’heure), 1 (retard)
#     """
#     for col in ['TotalDeliveryGap', 'TotalShippingDelay']:
#         df[col] = df[col].apply(lambda x: -1 if x < 0 else (1 if x > 0 else 0))
#     return df

def aggregate_features_for_pca(df, years=[2021, 2022]):
    """
    Agrège les variables numériques sur la période définie
    """
    df_period = df[df['Year'].isin(years)]
    df_agg = df_period.groupby('CustNum').agg({
        'TotalSalesValueLocal': 'mean',
        'NumberOfOrders': 'mean',
        'OrderToInvoiceDelayPerOrder': 'mean',
        'DeliveryGapPerOrder': 'mean',
        'ShippingDelayPerOrder': 'mean',
        'AvgOrderValue': 'mean',
        'SalesGrowthYoY': 'mean',
        'NumberOfVisits' : 'mean',
        'AVGLineUnitCost' : 'mean',
        'CompAP_Justified' : 'mean',
        'CompAP_NotJustified' : 'mean',
        'CompGP_Justified' : 'mean',
        'CompGP_NotJustified' : 'mean',
        'CompGI_Justified' : 'mean',
        'Comp_GI_NotJustified' : 'mean',
        'CompPck_Justified' : 'mean',
        'CompPP_Justified' : 'mean',
        'CompPP_NotJustified' : 'mean',
        'CompPhy_Justified' : 'mean',
        'CompPhy_NotJustified' : 'mean',
        'CompSOM_Justified' : 'mean',
        'CompSQ_Justified' : 'mean',
        'CompSQ_NotJustified' : 'mean',
        'CompST_Justified' : 'mean',
        'CompST_NotJustified' : 'mean',
        'Segment' : 'first', 
        'Typology' : 'first', 
        'Company' : 'first'
    }).reset_index()
    return df_agg

def run_pca(X_scaled, n_components=2):
    """
    Applique une ACP sur des données standardisées
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    return pca, X_pca

######### modélisation Logit

import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

def aggregate_features_for_logit(df, years=[2021, 2022]):
    """
    Agrège les variables numériques sur la période définie
    """
    df_period = df[df['Year'].isin(years)]
    df_agg = df_period.groupby('CustNum').agg({
        'TotalSalesValueLocal': 'mean',
        'NumberOfOrders': 'mean',
        'OrderToInvoiceDelayPerOrder': 'mean',
        'DeliveryGapPerOrder': 'mean',
        'ShippingDelayPerOrder': 'mean',
        'AvgOrderValue': 'mean',
        'AvgOrderValue_NBOrders': 'mean',
        'SalesGrowthYoY': 'mean',
        'NumberOfVisits' : 'mean',
        'AVGLineUnitCost' : 'mean',
        'CompAP_Justified' : 'mean',
        'CompAP_NotJustified' : 'mean',
        'CompGP_Justified' : 'mean',
        'CompGP_NotJustified' : 'mean',
        'CompGI_Justified' : 'mean',
        'Comp_GI_NotJustified' : 'mean',
        'CompPck_Justified' : 'mean',
        'CompPP_Justified' : 'mean',
        'CompPP_NotJustified' : 'mean',
        'CompPhy_Justified' : 'mean',
        'CompPhy_NotJustified' : 'mean',
        'CompSOM_Justified' : 'mean',
        'CompSQ_Justified' : 'mean',
        'CompSQ_NotJustified' : 'mean',
        'CompST_Justified' : 'mean',
        'CompST_NotJustified' : 'mean',
        'Segment' : 'first', 
        'Typology' : 'first', 
        'Company' : 'first', 
        'churner' : 'max'#,
        #'Company_Typology' : 'first',
        #'ShippingDelay_ComplaintsST' : 'mean'
    }).reset_index()
    return df_agg

import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_matrix(df, target_column='churner', figsize=(12,10), save_path=None):
    """
    Affiche la matrice de corrélation des variables numériques incluant la variable cible.
    """
    # Sélection des colonnes numériques uniquement
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    # Calcul de la corrélation
    corr_matrix = numeric_df.corr()
    
    # Affichage
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0)
    plt.title('Matrice de corrélation (analyse du churn)')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

    return corr_matrix





from sklearn.preprocessing import StandardScaler
import pandas as pd
import statsmodels.api as sm

def run_logit_model(df, max_iter=100, debug=False):
    """
    Estime un modèle logit robuste pour prédire le churn client.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données clients
        max_iter (int): Nombre maximum d'itérations pour la convergence
        debug (bool): Mode débogage pour afficher des informations supplémentaires
        
    Returns:
        tuple: (sm.Logit model ajusté, X (features avec constante), y (cible))
        
    Raises:
        ValueError: Si problèmes avec les données d'entrée
    """
    config = {
        'numeric_vars': [
            'NumberOfOrders',
            'OrderToInvoiceDelayPerOrder',
            'DeliveryGapPerOrder', 
            'ShippingDelayPerOrder',
            'AvgOrderValue',
            'AvgOrderValue_NBOrders',
            'SalesGrowthYoY',
            'NumberOfVisits',
            'AVGLineUnitCost',
            'CompGI_Justified',
            'Comp_GI_NotJustified',
            'CompPck_Justified',
            'CompPhy_Justified',
            'CompPhy_NotJustified',
            'CompSQ_Justified',
            'CompSQ_NotJustified',
            'CompST_Justified'#,
            #'CompST_NotJustified'
        ],
        'categorical_vars': ['Segment', 'Typology', 'Company'],
        'target': 'churner',
        'min_category_size': 10
    }
    
    try:
        required_cols = config['numeric_vars'] + config['categorical_vars'] + [config['target']]
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Colonnes requises manquantes: {missing_cols}")

        df_clean = df[required_cols].copy()
        if df_clean[config['target']].nunique() != 2:
            raise ValueError("La variable cible doit être binaire (0/1)")
        
        # Regroupement catégories rares et encodage one-hot
        for cat_var in config['categorical_vars']:
            counts = df_clean[cat_var].value_counts()
            small_categories = counts[counts < config['min_category_size']].index
            df_clean[cat_var] = df_clean[cat_var].replace(small_categories, 'OTHER')
            
        df_encoded = pd.get_dummies(
            df_clean,
            columns=config['categorical_vars'],
            drop_first=True,
            dtype=float
        )
        
        initial_size = len(df_encoded)
        df_encoded.dropna(inplace=True)
        if debug and (initial_size != len(df_encoded)):
            print(f"[DEBUG] {initial_size - len(df_encoded)} lignes supprimées pour valeurs manquantes")
        
        if len(df_encoded) == 0:
            raise ValueError("Aucune donnée valide après nettoyage")

        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(df_encoded[config['numeric_vars']])
        df_encoded[config['numeric_vars']] = scaled_values
        
        if df_encoded.select_dtypes(include=['object']).shape[1] > 0:
            raise ValueError("Présence de colonnes non numériques après traitement")

        X = sm.add_constant(df_encoded.drop(config['target'], axis=1))
        y = df_encoded[config['target']].astype(int)
        
        if debug:
            print(f"[DEBUG] Dimensions finales - X: {X.shape}, y: {y.shape}")
            print(f"[DEBUG] Distribution de la cible: {y.value_counts(normalize=True)}")
        
        model = sm.Logit(y, X).fit(
            method='lbfgs',
            maxiter=max_iter,
            disp=debug
        )
        
        if not model.mle_retvals['converged']:
            print("[WARNING] Le modèle n'a pas complètement convergé. Essayez d'augmenter max_iter.")
            
        return model, X, y

    except Exception as e:
        print("Erreur dans run_logit_model:")
        print(f"Type: {type(e).__name__}")
        print(f"Message: {str(e)}")
        
        if debug and 'df_encoded' in locals():
            print("\n=== DEBUG INFORMATION ===")
            print("Types des colonnes:")
            print(df_encoded.dtypes)
            print("\nStatistiques descriptives:")
            print(df_encoded.describe())
            
        raise

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import statsmodels.api as sm


def plot_roc_auc(model, X, y):
    """
    Trace la courbe ROC et affiche l'AUC d'un modèle logit statsmodels.
    
    Args:
        model: modèle logit ajusté
        X (pd.DataFrame): Features avec constante (utilisés pour l’ajustement)
        y (pd.Series): Variable cible binaire
    """
    # Prédictions des probabilités (seulement pour la classe positive)
    y_pred_proba = model.predict(X)
    
    # Calcul des points ROC
    fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
    
    # AUC
    auc_score = roc_auc_score(y, y_pred_proba)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Courbe ROC (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Aléatoire')
    plt.xlabel('Taux de faux positifs (FPR)')
    plt.ylabel('Taux de vrais positifs (TPR)')
    plt.title('Courbe ROC - Modèle Logit')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()
