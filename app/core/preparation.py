import pandas as pd  # type: ignore
from typing import Dict
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder , StandardScaler # type: ignore
 
#Convertir dict en DataFrame
def dict_to_dataframe(tx: Dict) -> pd.DataFrame:
    return pd.DataFrame([tx])

# Ajout des features temporelles
def time_handling(df: pd.DataFrame) -> pd.DataFrame:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['heure'] = df['timestamp'].dt.hour
    df['jour_semaine'] = df['timestamp'].dt.weekday
    df['mois'] = df['timestamp'].dt.month
    df['is_weekend'] = df['jour_semaine'].isin([5, 6]).astype(int)
    return df

def enrich_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df['hour'] = df['timestamp'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    df['day_of_week'] = df['timestamp'].dt.dayofweek
    # df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    # df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_night'] = df['hour'].apply(lambda h: 1 if h < 6 or h >= 22 else 0)
    df = df.drop(columns=['timestamp'])

    return df

#Gestion des valeurs manquantes
def missing_values_treatment(df: pd.DataFrame) -> pd.DataFrame:
    df['id_receiver'] = df['id_receiver'].fillna("unknown")
    df['mode'] = df['mode'].fillna("unknown")
    df['simulated'] = df['simulated'].fillna(False)     
    df['natur_tx'] = df['natur_tx'].fillna("xxx-xxx-xxx")
    df['status'] = df ['status'].fillna("xxx-xxx-xxx")
    df['statut_compte'] = df['statut_compte'].fillna("xxx-xxx-xxx")
    df['type_compte'] = df['type_compte'].fillna("xxx-xxx-xxx")
     
    return df

# def scaling(df: pd.DataFrame) -> pd.DataFrame:
#     ldam_scaler = joblib.load('models/ldam_scaler.pkl')
#     col_to_scale = [
#         'amount','sender_balance_before',
#         'sender_balance_after', 'sender_avg_tx_amount_30d'
#     ]
    # df[col_to_scale] = ldam_scaler.transform(df[col_to_scale])
    # return df

cols_to_encode = ["currency", "natur_tx", "mode" ,'transaction_code',
    'status', 'statut_compte', 'type_compte',
    'day_of_week']
    
#Encodage des colonnes catégorielles
def encoding(df: pd.DataFrame) -> pd.DataFrame:
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_array = encoder.fit_transform(df[cols_to_encode])
    encoded_col_names = encoder.get_feature_names_out(cols_to_encode)
    df_encoded = pd.DataFrame(encoded_array, columns=encoded_col_names, index=df.index)
    df_final = pd.concat([df.drop(columns=cols_to_encode), df_encoded], axis=1)
    return df_final


FEATURE_ORDER = joblib.load("data/data_processed/ordre.pkl")  

def preprocess_transaction(transaction: Dict) -> np.ndarray:
 
    raw_data = pd.DataFrame([transaction])
    #Remplissage des valeurs manquantes
    processed = {}
    for feat in FEATURE_ORDER:
        if feat in raw_data:
            processed[feat] = raw_data[feat].values[0]
        else:
            if feat.startswith(('cat_', 'encoded_')):
                processed[feat] = "MISSING"  # Catégorielle
            else:
                processed[feat] = 0.0  # Numérique

    #Alignement final des colonnes
    final_features = [processed.get(feat, 0.0) for feat in FEATURE_ORDER]
    return np.array([final_features])

#Pipeline final
def final_treatment(tx: Dict) -> pd.DataFrame:
    df = dict_to_dataframe(tx)
    df = time_handling(df)
    df = enrich_time_features(df)
    df = missing_values_treatment(df)
    df = encoding(df)
    columns_to_not_train = ["id_tx", "id_sender", "id_receiver", "sender_last_tx_date"]
    df = df.drop(columns=columns_to_not_train, axis=1, errors='ignore')
    
    datetime_cols = df.select_dtypes(include=['datetime64[ns]']).columns
    if len(datetime_cols) > 0:
        df = df.drop(columns=datetime_cols)
    for feat in FEATURE_ORDER:
        if feat not in df.columns:
            df[feat] = 0.0  # valeur par défaut si colonne absente
    df = df[FEATURE_ORDER]

    return df
