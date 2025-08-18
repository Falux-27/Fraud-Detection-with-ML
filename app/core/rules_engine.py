 
import pandas as pd # type: ignore
import numpy as np  # type: ignore
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json

def unusual_amounts(id_sender : str, amount : float, engine) -> float:
     
    requete = f"""
        SELECT amount 
        FROM fraud_transactions
        WHERE id_sender = '{id_sender}'
    """
    df_transac = pd.read_sql(requete , engine)
    if df_transac.empty or len(df_transac) < 3:
        return 0.0
    moyenne = df_transac['amount'].mean()
    ecart_type = df_transac['amount'].std()
    #pas de variation
    if ecart_type == 0:
        return 0.0
    z_score_mnt =abs ((amount - moyenne)/ecart_type)
    return round(z_score_mnt, 2)


def ratio_amount_max(id_sender: str , amount: float,engine) -> float:
    requete = f"""
    SELECT max(amount) AS amount_max
    FROM fraud_transactions
    WHERE id_sender= "{id_sender}" 
    """
    dtfrm = pd.read_sql(requete , engine)
    #Si l'user existe pas
    if dtfrm.empty :
        return 0.0
    #Si le montant max est Nan (aucune tx associé à l'id) ou egale a zero
    amount_max = dtfrm.at[0 , "amount_max"]
    if pd.isnull(amount_max) or amount_max == 0:
        return 0.0
    ratio = amount/dtfrm['amount_max']
    return round(ratio, 2)

def freq_tx_by_24(id_sender: str,  engine) -> float:
     
    requete = f"""
        SELECT 
            t24.id_sender,
            t24.nb_tx_24h,
            t30.moyenne_tx_par_jour
        FROM (
            SELECT id_sender, COUNT(*) AS nb_tx_24h
            FROM fraud_transactions
            WHERE timestamp >= NOW() - INTERVAL 1 DAY
              AND id_sender = '{id_sender}'
            GROUP BY id_sender
        ) AS t24
        LEFT JOIN (
            SELECT 
                id_sender, 
                AVG(nb_tx_jour) AS moyenne_tx_par_jour
            FROM (
                SELECT 
                    id_sender, 
                    DATE(timestamp) AS jour, 
                    COUNT(*) AS nb_tx_jour
                FROM fraud_transactions
                WHERE timestamp >= CURDATE() - INTERVAL 30 DAY
                  AND id_sender = '{id_sender}'
                GROUP BY id_sender, jour
            ) AS tx_par_jour
            GROUP BY id_sender
        ) AS t30 ON t24.id_sender = t30.id_sender;
    """
    
    dtfrm = pd.read_sql(requete, engine)

    if dtfrm.empty:
        return 0.0

    nb_tx_24h = dtfrm.at[0, 'nb_tx_24h']
    moyenne = dtfrm.at[0, 'moyenne_tx_par_jour']

    if pd.isna(moyenne) or moyenne == 0:
        return 0.0

    return nb_tx_24h / moyenne

class niveau_risk (Enum):
    CRITICAL = 'CRITIQUE'
    HIGH = 'ÉLEVÉ'
    MODERATE = "MODÉRÉ"
    MONITORED = "SURVEILLÉ"
    LOW = "FAIBLE"
    
class type_de_fraude (Enum):
    CARD_FRAUD = "FRAUDE_CARTE"
    IDENTITY_THIEFT = "USURPATION_IDENTITE"
    EMERGING_FRAUD = "FRAUDE_NOUVELLE"
    CARD_THIEFT = "VOL_DE_CARTE"
    DEEPFAKE = "DEEPFAKE"
     
class status_tx(Enum):
    APPROVED = "APPROUVÉ"
    BLOCKED = "BLOQUÉ"
    MANUAL_REVIEW  = "VALIDATION_MANUELLE"
    ADDITIONAL_CONTROLS = "VERIFICATION_SUPPLEMANTAIRES"
    MONITORING = "SURVEILLANCE"
    
class niveau_alert (Enum):
    L1_AUTOMATIC = "L1_AUTOMATIQUE"
    L2_ANALYST = "L2_ANALYSTE"
    L3_SENIOR = "L3_SENIOR"
    L4_BLOCAGE = "L4_CONFORMITE"

Pays_risque = {
            #Niveau critique
            "MX":{"score": 10, "niveau_risk":niveau_risk.CRITICAL, "types_fraud":[type_de_fraude.CARD_FRAUD, type_de_fraude.CARD_THIEFT],"note": 44},
            "US":{"score": 10, "niveau_risk":niveau_risk.CRITICAL, "types_fraud":[type_de_fraude.CARD_FRAUD, type_de_fraude.CARD_THIEFT],"note": 42},
            "IN":{"score": 9, "niveau_risk":niveau_risk.CRITICAL, "types_fraud":[type_de_fraude.CARD_FRAUD, type_de_fraude.IDENTITY_THIEFT],"note": 37},
            "AE":{"score": 9, "niveau_risk":niveau_risk.CRITICAL, "types_fraud":[type_de_fraude.CARD_FRAUD],"note": 36},
            "CN":{"score": 9, "niveau_risk":niveau_risk.CRITICAL, "types_fraud":[type_de_fraude.CARD_FRAUD, type_de_fraude.CARD_THIEFT,type_de_fraude.EMERGING_FRAUD],"note": 36},
            
            #Niveau eleve
            "GB":{"score":8 , "niveau_risk":niveau_risk.HIGH,"types_fraude":[type_de_fraude.CARD_FRAUD], "note":34},
            "BR":{"score":8 , "niveau_risk":niveau_risk.HIGH,"types_fraude":[type_de_fraude.CARD_FRAUD, type_de_fraude.CARD_THIEFT], "note":33},
            "AU":{"score":7 , "niveau_risk":niveau_risk.HIGH,"types_fraude":[type_de_fraude.CARD_FRAUD], "note":31},
            "SG":{"score":7 , "niveau_risk":niveau_risk.HIGH,"types_fraude":[type_de_fraude.CARD_FRAUD], "note":26},
            "ZA": {"score": 7, "niveau_risk": niveau_risk.HIGH, "types_fraude": [type_de_fraude.CARD_FRAUD], "rate": 25},
            "CA": {"score": 7, "niveau_risk": niveau_risk.HIGH, "types_fraude": [type_de_fraude.CARD_FRAUD], "rate": 25},
            
            #Niveau MODÉRÉ 
            "IT": {"score": 6, "niveau_risk":niveau_risk.MODERATE,"types_fraude": [ type_de_fraude.CARD_FRAUD], "rate": 24},
            "FR": {"score": 5, "niveau_risk":niveau_risk.MODERATE,"types_fraude": [ type_de_fraude.CARD_FRAUD], "rate": 20},
            "ID": {"score": 6, "niveau_risk":niveau_risk.MODERATE,"types_fraude": [ type_de_fraude.CARD_FRAUD,  type_de_fraude.EMERGING_FRAUD,  type_de_fraude.IDENTITY_THIEFT], "rate": 18},
            
            #Niveau SURVEILLÉ (3-4)
            "NG": {"score": 4, "niveau_risk":  niveau_risk.MONITORED,"types_fraude": [type_de_fraude.EMERGING_FRAUD, type_de_fraude.DEEPFAKE], "rate": 0},
            "BD": {"score": 4, "niveau_risk":  niveau_risk.MONITORED,"types_fraude": [type_de_fraude. CARD_THIEFT], "rate": 5.44},
            "PK": {"score": 4, "niveau_risk":  niveau_risk.MONITORED,"types_fraude": [type_de_fraude. CARD_THIEFT], "rate": 4.59},
            "VN": {"score": 4, "niveau_risk":  niveau_risk.MONITORED,"types_fraude": [type_de_fraude. CARD_THIEFT], "rate": 9.94},
            "HK": {"score": 3, "niveau_risk":  niveau_risk.MONITORED,"types_fraude": [type_de_fraude. CARD_THIEFT], "rate": 0},
            "KH": {"score": 3, "niveau_risk":  niveau_risk.MONITORED,"types_fraude": [type_de_fraude. CARD_THIEFT], "rate": 0},
            "PH": {"score": 3, "niveau_risk":  niveau_risk.MONITORED,"types_fraude": [type_de_fraude. CARD_THIEFT], "rate": 0},
            
            # Niveau FAIBLE (1-2)
            "DE": {"score": 2, "niveau_risk": niveau_risk.LOW, "fraud_types": [type_de_fraude.CARD_FRAUD], "rate": 13}
            
        }

SEUIL_RISK = {
            niveau_risk.CRITICAL:{"montant" : 1000, "niveau_alert":niveau_alert.L4_BLOCAGE},
            niveau_risk.HIGH :{"montant" :1500 , "niveau_alert":niveau_alert.L3_SENIOR},
            niveau_risk.MODERATE:{"montant" :2000 , "niveau_alert": niveau_alert.L2_ANALYST},
            niveau_risk.MONITORED:{"montant": 3000, "niveau_alert":niveau_alert.L2_ANALYST},
            niveau_risk.LOW:{"montant": 5000,"niveau_alert":niveau_alert.L1_AUTOMATIC}
        }

#Model client vu qu'on ne dispose pas encore d'infos de client sur la base 
@dataclass
class client:
    id_cust: str
    prenom : str
    nom : str
    country_code : str
    risk_profile : str
    date_inscription : datetime

def evaluer_risk (client_obj: client, montant: float) -> Tuple[niveau_risk, niveau_alert, status_tx]:
    code_pays = client_obj.country_code
    pays_info = Pays_risque.get(code_pays)

    if not pays_info:
        # pays inconnu = risque faible par défaut  
        return niveau_risk.LOW, niveau_alert.L1_AUTOMATIC, status_tx.APPROVED

    niveau = pays_info["niveau_risk"]
    seuil_info = SEUIL_RISK[niveau]

    if montant >= seuil_info["montant"]:
        if niveau == niveau_risk.CRITICAL:
            return niveau, seuil_info["niveau_alert"], status_tx.BLOCKED
        elif niveau in [niveau_risk.HIGH, niveau_risk.MODERATE]:
            return niveau, seuil_info["niveau_alert"], status_tx.MANUAL_REVIEW
        else:
            return niveau, seuil_info["niveau_alert"], status_tx.ADDITIONAL_CONTROLS
    else:
        return niveau, niveau_alert.L1_AUTOMATIC, status_tx.APPROVED
    
#Quand on aura unne base client
# def get_client_info(id_sender: str, engine) -> client:
#     try:
#         requete = f"""
#             SELECT id_cust, prenom, nom, country_code, risk_profile, date_inscription
#             FROM client.customer
#             WHERE id_cust = '{id_sender}'
#             LIMIT 1
#         """
#         df = pd.read_sql(requete, engine)
        
#         if not df.empty:
#             return client(
#                 id_cust=df.at[0, 'id_cust'],
#                 prenom=df.at[0, 'prenom'] if pd.notna(df.at[0, 'prenom']) else "N/A",
#                 nom=df.at[0, 'nom'] if pd.notna(df.at[0, 'nom']) else "N/A", 
#                 country_code=df.at[0, 'country_code'] if pd.notna(df.at[0, 'country_code']) else "FR",
#                 risk_profile=df.at[0, 'risk_profile'] if pd.notna(df.at[0, 'risk_profile']) else "N/A",
#                 date_inscription=pd.to_datetime(df.at[0, 'date_inscription']) if pd.notna(df.at[0, 'date_inscription']) else datetime(2022, 1, 1)
#             )
#         else:
#             # Client par défaut si non trouvé
#             return client(
#                 id_cust=id_sender,
#                 prenom="Inconnu",
#                 nom="Inconnu",
#                 country_code="FR", 
#                 risk_profile="N/A",
#                 date_inscription=datetime(2022, 1, 1)
#             )
            
#     except Exception as e:
#         print(f"Erreur récupération client {id_sender}: {e}")
#         # Client par défaut en cas d'erreur
#         return client(
#             id_cust=id_sender,
#             prenom="Erreur",
#             nom="Erreur", 
#             country_code="FR",
#             risk_profile="N/A",
#             date_inscription=datetime(2022, 1, 1)
#         )
    
    
def freq_by_20mn(id_sender: str , engine) -> Tuple[int , str]:
    requete = """
            SELECT COUNT(*) as nb_tx
            FROM fraud_transactions
            WHERE id_sender = '{id_sender}'
            AND timestamp >= NOW() - INTERVAL 20 MINUTE;
    """
    dtfrm = pd.read_sql(requete , engine)
    nb_tx = dtfrm.at[0, 'nb_tx']
    if nb_tx > 3 :
        return 1, "Suspect"
    return 0, ""

def unusual_time(tx_timestamp: datetime) -> Tuple[int, str]:
    heure = tx_timestamp.hour
    if 0 <= heure < 5:
        return 1, "HEURE_INHABITUELLE"
    return 0, ""

def pipeline_comportementales(tx: dict, engine) -> Tuple[int, List[str]]:
    score = 0
    rapport =[]
    id_sender = tx["id_sender"]
    timestamp = pd.to_datetime(tx["timestamp"])
    score1 , avis1 =  freq_by_20mn(id_sender , engine)
    score2 , avis2 = unusual_time(timestamp)
    
    score = score + score1
    score = score + score2
    if avis1 != "":
        rapport.append(avis1)
    elif avis2 != "":
        rapport.append(avis2)
        
    return score , rapport

def get_country_code(id_sender:str, engine:str) -> str:
        requete = f"""
            SELECT country_code as code_alpha
            FROM client.customer
            WHERE id_cust = "{id_sender}"
        """
        dtfrm = pd.read_sql(requete, engine)
        if  not dtfrm.empty:
            return dtfrm.at[0,"code_alpha"]
        return "Inconnu"
def verifier_changement_geo(id_sender: str, pays_actuel: str, timestamp: datetime, engine) -> Tuple[int, str]:
    requete = f"""
        SELECT c.country_code, t.timestamp
        FROM fraud_transactions t
        JOIN client.customer c ON t.id_sender = c.id_cust
        WHERE t.id_sender = '{id_sender}'
        ORDER BY t.timestamp DESC
        LIMIT 1
    """
    df = pd.read_sql(requete, engine)

    if df.empty:
        return 0, ""

    pays_dernier = df.at[0, "country_code"]
    timestamp_dernier = pd.to_datetime(df.at[0, "timestamp"])
    delta = timestamp - timestamp_dernier

    if pays_dernier != pays_actuel and delta.total_seconds() < 3600:
        return 1, "CHANGEMENT_GEO_SUSPECT"

    return 0, ""

        