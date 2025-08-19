import pandas as pd
import numpy as np
from scipy.stats import normaltest
from scipy.stats import anderson
from scipy.stats import probplot
from imblearn.under_sampling import RandomUnderSampler
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import ClusterCentroids
from sklearn.model_selection import train_test_split, RandomizedSearchCV,GridSearchCV
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import joblib

#Loading
dataset = pd.read_csv(r"C:\Users\HP\Desktop\Fraud_detection_project\data\fraud_transactions_detectable.csv")
print(dataset.sample(10))

#Exploration du dataset
print(dataset.info(),"\n")
print (dataset.dtypes,"\n")
print(dataset.describe().T,"\n")
doublons = dataset.duplicated().sum()
print(doublons,"\n")
null_val = dataset.isnull().sum()
print(null_val,"\n")
print(dataset.columns)

#Pretraitement des données
uniq_val = dataset['label'].unique()
print(f"valeur uniques du target : {uniq_val}\n")

#Pourcentage de chaque valeur unique
percent_val_uniq = dataset['label'].value_counts(normalize=True).round(3)
# sns.set_style("darkgrid")
# sns.countplot(data = dataset ,x = 'label' ,hue='label' )
# plt.ylabel("Nombre d'observation")
# plt.show()
# plt.close()

print(f"Pourcentage des classes: {percent_val_uniq}\n") 

#Gestion valeurs manquantes
col_with_missing_val = [col for col in dataset.columns if dataset[col].isna().sum() >= 1]
print(f"Colonnes avec valeurs manquantes : {col_with_missing_val}\n")

print("Analyse de la colonne montant...\n")
stat = dataset ["amount"].describe()
stat_dtfrm = pd.DataFrame(stat).T
print(f"Stats de la colonne montant :\n {stat_dtfrm}\n")

# #Histogramme de distribution 
# mediane = np.mean(dataset['amount']).round(2)
# sns.set_style("whitegrid")
# plt.hist(x=dataset['amount'] , bins = "sturges" ,density=True, color='lightblue', edgecolor='black', alpha=0.7, align = 'mid')
# plt.axvline(mediane ,color = 'red', linestyle ="dashed", linewidth = 2 , label =f'moyenne:{mediane}' )
# plt.xlabel("Montant")
# plt.ylabel('Nombre de personne')
# plt.legend()
# #plt.show()
# plt.close()

# #Courbe de densité des observations
# sns.set_style('darkgrid')
# sns.histplot(x = dataset['amount'], kde=True,bins ="sturges", stat='density', kde_kws={'bw_adjust': 0.5},color = 'red')
# plt.xlabel("Montant")
# plt.ylabel('Nombre de personne')
# plt.show()
# plt.close()
# #Application de transformation logarithme pour reduire l'ecart
dataset['amount_log'] = np.log(dataset['amount']+1)

# #Courbe de densité des observations apres transformation logarithmique 
# sns.set_style('darkgrid')
# sns.histplot(x = dataset['amount_log'],kde=True,bins ="fd", stat='density', kde_kws={'bw_adjust': 0.5},color = 'blue')
# plt.xlabel("Montant")
# plt.ylabel('Nombre de personne')
# plt.show()
# plt.close()

# #Verifier les valeurs outliers
# plt.boxplot(x = dataset['amount_log'], 
#             flierprops=dict(marker='o', markerfacecolor='red', markersize=8),   #Les outliers forme des points, taille, couleur
#             boxprops=dict(color='blue', linewidth=2),  # La boîte 	bordures, couleur, largeur
#             showmeans=True, capprops=dict(color='green', linewidth=2), # point de la  moyenne et les extrémités horizontales
#             whiskerprops=dict(color='orange', linestyle='--'))
# plt.show()
# plt.close()

dataset['timestamp'] = pd.to_datetime(dataset['timestamp'])

#Heure de la journée 
dataset['hour'] = dataset['timestamp'].dt.hour
dataset['hour_sin'] = np.sin(2 * np.pi * dataset['hour'] / 24)
dataset['hour_cos'] = np.cos(2 * np.pi * dataset['hour'] / 24)

#Jour de la semaine (0 = lundi, 6 = dimanche)
dataset['day_of_week'] = dataset['timestamp'].dt.dayofweek

#Est-ce un week-end ? (1 = oui, 0 = non)
dataset['is_weekend'] = dataset['day_of_week'].isin([5, 6]).astype(int)

#Est-ce une heure "nocturne" (fraude plus probable entre 22h et 6h) ?
dataset['is_night'] = dataset['hour'].apply(lambda h: 1 if h < 6 or h >= 22 else 0)

#Création d'une période de la journée
def get_time_period(hour):
    if 5 <= hour < 14:
        return 'morning'
    elif 14 <= hour < 17:
        return 'afternoon'
    elif 17 <= hour < 22:
        return 'evening'
    else:
        return 'night'

dataset['time_period'] = dataset['hour'].apply(get_time_period)
print(dataset.info)
#Corrélation entre les variables numériques
numerical_cols  = dataset.select_dtypes(include=["float64", "int64"]).columns.to_list()
matrix =dataset[numerical_cols].corr()
# plt.figure(figsize=(12,10))
# sns.heatmap(matrix, annot=True, cmap='rocket',fmt=".2f", annot_kws={"size": 8})
# plt.title("Matrice de Corrélation")
# plt.show()

print("")
#Detection des outliers avec la methode Robust Z-score
mediane = dataset['amount_log'].median()
MAD = np.median(np.abs(dataset['amount_log'] - mediane)).round(2)
dataset['Robust Z-score'] = (dataset['amount_log'] - mediane)/ (1.4826 * MAD)
seuil = 3
dataset['is_outlier'] = dataset['Robust Z-score'].abs() > seuil
nbr_outlier = dataset['is_outlier'].sum()
outliers = dataset[dataset['is_outlier']]
print(f"Nombre d'outlier : {nbr_outlier}")
dataset['is_outlier'] = dataset['is_outlier'].map({False: 0 , True: 1})
print(f"Classe des des outliers : {dataset['is_outlier'].unique()}\n")


#Detection des outliers avec la methode IQR
Q1 = dataset['amount_log'].quantile(0.25)
Q3 = dataset['amount_log'].quantile(0.75)
IQR = Q3 - Q1
high = Q3 + 1.5 * IQR
low = Q1 -1.5 * IQR
dataset['is_outlier_by_IQR'] = ~dataset['amount_log'].between(low, high)
nb_outlier = dataset[dataset['is_outlier_by_IQR']==True]

# Intersection des deux méthodes
dataset["outlier_commun"] = dataset["is_outlier_by_IQR"] & dataset["is_outlier"] 
#Outliers detectés en commun
nb_outliers_communs = dataset["outlier_commun"].sum()
print(f"Le nombre d'outlier en communs : {nb_outliers_communs}")

#Suppressions des outliers
# dataset = dataset[~(dataset["is_outlier_by_IQR"].astype(bool) & dataset["is_outlier"].astype(bool))].copy()

#Graphe Q-Q Plot
cols = ["amount_log","amount"]
# for col in cols :
#     plt.figure(figsize=(6,6))
#     probplot(dataset[col],dist='norm', plot=plt)
#     plt.title(f"Q-Q Plot pour {col}")
#     plt.xlabel("Quantiles théoriques")
#     plt.ylabel("Quantiles observés")
#     plt.grid(True)
#     plt.show()
    
#test de D’Agostino–Pearson
normal_list = []
no_normal_list =[]
treshold = 0.05
for elt in cols:
    stat , p_value = normaltest(dataset[elt])      #Pas efficace
    if p_value > treshold:
        no_normal_list.append(elt)
    else:
        normal_list.append(elt)
print(f"Colonnes normales:{normal_list}")

#Suppression des colonnes inutiles 
dataset.drop(columns='outlier_commun', inplace=True)
dataset.drop(columns=[
    'Robust Z-score', 
    'is_outlier', 
    'is_outlier_by_IQR', 
    'amount_log', "date_created",'timestamp', 'hour'], inplace=True)



#Encodage
cat_cols = ['currency', 'transaction_code', 'mode', 'natur_tx',
    'status', 'statut_compte', 'type_compte',
    'day_of_week', 'time_period']

print(f"Colonne catégorielles : {cat_cols}\n")
encoder = OneHotEncoder(sparse_output=False,  handle_unknown='ignore')
column_encoded = encoder.fit_transform(dataset[cat_cols])
one_hot_dtfrm = pd.DataFrame(column_encoded , columns = encoder.get_feature_names_out(cat_cols))
dataset = pd.concat([dataset , one_hot_dtfrm] , axis = 1)
dataset = dataset.drop(columns= cat_cols)

#Ordre des colonnes
colonnes_sans_class = [col for col in dataset.columns if col != 'label']
dataset = dataset[colonnes_sans_class + ['label']]
print (f"Bonne ordere colonnes :{dataset.columns}\n")


#Sauvegarde de caracteristique
id_tx_saved = dataset['id_tx']
id_sender_saved = dataset ['id_sender']
id_receiver_saved = dataset ['id_receiver'] 
sender_last_tx_date_saved = dataset['sender_last_tx_date']

#Splitting features - target
features = dataset.drop(columns=['label','id_tx','id_receiver','id_sender','sender_last_tx_date'],axis=1)
target = dataset['label']

#Splitting the dataset into train-test
x_train , x_test , y_train , y_test = train_test_split(features, target, stratify=target, test_size=0.2, random_state=42)

#Niveau 1:  Sous-échantillonnage avec Random undersampling
rand_u_s = RandomUnderSampler(sampling_strategy={0:150000 },random_state=42)
x_train_rus, y_train_rus = rand_u_s.fit_resample(x_train, y_train)
nb_each_class = Counter(y_train_rus)
print(nb_each_class,"\n")
print(type(x_train_rus),"\n")
print(x_train_rus.columns,"\n")

ordre_col = list(x_train_rus.columns)
joblib.dump(ordre_col, 'data/data_processed/ordre.pkl')
#Niveau 2:sous-échantillonnage intelligent (K-means)
scaler = StandardScaler()
x_train_rus_scaled = scaler.fit_transform(x_train_rus)

mini_kmeans = MiniBatchKMeans(n_init=5 , n_clusters=5000, random_state=42, batch_size=1000)
cluster = ClusterCentroids(
    sampling_strategy={0: 5000},
    estimator=mini_kmeans,
    random_state= 42
)
x_train_resampled , y_train_resampled = cluster.fit_resample(x_train_rus_scaled,y_train_rus )
nb = Counter(y_train_resampled)

#Niveau 3:sur-échantillonnage avec ADASYN 

#ADASYN
over_sampler =ADASYN(sampling_strategy=0.64, n_neighbors=20,random_state=42 )
final_x_train_scaled = scaler.fit_transform(x_train_resampled)
final_x_train , final_y_train = over_sampler.fit_resample(final_x_train_scaled, y_train_resampled)
nb = Counter(final_y_train)
print(nb)
#Verification des potentiels doublons généré par ADASYN
duplicates = pd.DataFrame(final_x_train).duplicated().sum()
doublons =int(np.int16(duplicates))
print(doublons)
"""
#visualisation des comptes
sns.set_style("darkgrid")
sns.countplot(x=final_y_train)
plt.ylabel("Nombre d'observations")
plt.title("Distribution des classes après ADASYN")
plt.show()
"""
 

#Sauvegarde


data_to_save = {
    'final_x_train': final_x_train,
    'x_test': x_test,
    'final_y_train': final_y_train,
    'y_test': y_test
}
joblib.dump(data_to_save, 'data/data_processed/train_test_data.pkl')

