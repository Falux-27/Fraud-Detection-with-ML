from app.db.db_connect import connexion_base
import pandas as pd # type: ignore
from datetime import datetime
from dataclasses import dataclass, field
from app.core.rules_engine import  *

def evaluer_transaction_par_règles(tx: dict, client_obj: client) -> dict:
    engine = connexion_base()
    try:
        #Extraction des données transactionnelles
        id_sender = tx["id_sender"]
        amount = float(tx["amount"])
        timestamp = pd.to_datetime(tx["timestamp"])
        
        #Infos issues du modele client
        pays_actuel = client_obj.country_code
        risk_profile = client_obj.risk_profile
        date_inscription = client_obj.date_inscription
        print(f"Évaluation pour le client: {client_obj.prenom} {client_obj.nom} ({client_obj.id_cust}), montant: {amount}")
        
        #Analyses de montant
        zscore = unusual_amounts(id_sender, amount, engine)
        ratio_max = ratio_amount_max(id_sender, amount, engine)
        
        #Analyse de fréquence 
        freq_ratio = freq_tx_by_24(id_sender, engine)
        
        #Analyses comportementales
        comportement_score, comportement_avis = pipeline_comportementales(tx, engine)
        
        #Vérification géographique
        changement_geo_flag, changement_geo_msg = verifier_changement_geo(
            id_sender, pays_actuel, timestamp, engine
        )
        #Évaluation du risque pays
        niveau, alerte, statut = evaluer_risk(client_obj, amount)
        
        #Calcul du score final
        score = 0
        anomalies = []
        #Règles de scoring
        if zscore > 3:
            score += 2
            anomalies.append(f"Montant inhabituel (z-score: {zscore})")
        elif zscore > 2:
            score += 1
            anomalies.append(f"Montant légèrement élevé (z-score: {zscore})")
        #Montant vs historique max →
        if ratio_max > 2.5:
            score += 2 
            anomalies.append(f"Ratio montant très élevé ({ratio_max}x le maximum)")
        elif ratio_max > 1.5:
            score += 1
            anomalies.append(f"Ratio montant élevé ({ratio_max}x le maximum)")
        
        if freq_ratio > 3:
            score += 2
            anomalies.append(f"Fréquence 24h très anormale ({freq_ratio}x la moyenne)")
        elif freq_ratio > 2:
            score += 1
            anomalies.append(f"Fréquence 24h anormale ({freq_ratio}x la moyenne)")
        
        if changement_geo_flag:
            score += 2
            anomalies.append(changement_geo_msg)
        
        #Ajout du score comportemental
        score += comportement_score
        anomalies.extend(comportement_avis)
        
        #Détermination du niveau de risque final
        if score >= 5:
            risk_level = "CRITIQUE"
        elif score >= 3:
            risk_level = "ÉLEVÉ"
        elif score >= 1:
            risk_level = "MODÉRÉ"
        else:
            risk_level = "FAIBLE"
        
        return {
            "score_fraude": score,
            "niveau_risque": risk_level,
            "niveau_risque_pays": niveau.value,
            "niveau_alerte": alerte.value,
            "statut_transaction": statut.value,
            "anomalies": anomalies,
            "details": {
                "z_score_montant": zscore,
                "ratio_montant_max": ratio_max, 
                "ratio_frequence_24h": freq_ratio,
                "score_comportemental": comportement_score,
                "changement_geo": changement_geo_flag,
                "pays_client": pays_actuel,
                "profil_risque_client": risk_profile,
                "date_inscription": date_inscription
            }
        }
        
    except Exception as e:
        print(f"Erreur dans l'évaluation: {e}")
        return {
            "score_fraude": 0,
            "niveau_risque": "ERREUR",
            "niveau_risque_pays": "N/A",
            "niveau_alerte": "L1_AUTOMATIQUE", 
            "statut_transaction": "ERREUR",
            "anomalies": [f"Erreur d'évaluation: {str(e)}"],
            "details": {}
        }
    
    finally:
        if engine:
            engine.dispose()

            
            
            
            
            
            
            





















# def evaluer_transaction_par_règles(tx: dict) -> dict:
#     engine = connexion_base()
#     #Extraction des données
#     id_sender = tx["id_sender"]
#     amount = tx["amount"]
#     timestamp = pd.to_datetime(tx["timestamp"])
#     pays_actuel = get_country_code(id_sender, engine)

#     zscore = unusual_amounts(id_sender, amount,engine)
#     ratio_max = ratio_amount_max(id_sender, amount,engine)
#     freq_ratio = freq_tx_by_24(id_sender, engine)
#     comportement_score, comportement_avis = pipeline_comportementales(tx, engine)
#     changement_geo_flag, changement_geo_msg = verifier_changement_geo(id_sender, pays_actuel, timestamp, engine)


    # # Création objet client 
    # client_obj = client(
    #     id_cust="XQ1354",
    #     prenom="Malick",
    #     nom="Gueye",
    #     country_code='FR',
    #     risk_profile="N/A",
    #     date_inscription = datetime(2022, 1, 1) 
    # )
#     niveau, alerte, statut = evaluer_risk(client_obj, amount)

#     score = 0
#     anomalies = []

#     if zscore > 3:
#         score += 1
#         anomalies.append(f"Montant inhabituel (z-score: {zscore})")

#     if ratio_max > 2.5:
#         score += 1
#         anomalies.append(f"Ratio montant élevé (ratio: {ratio_max})")

#     if freq_ratio > 2:
#         score += 1
#         anomalies.append(f"Fréquence 24h anormale (ratio: {freq_ratio})")

#     if changement_geo_flag:
#         score += 1
#         anomalies.append(changement_geo_msg)

#     score += comportement_score
#     anomalies.extend(comportement_avis)

#     return {
#         "score_fraude": score,
#         "niveau_risque": niveau.value,
#         "niveau_alerte": alerte.value,
#         "statut_transaction": statut.value,
#         "anomalies": anomalies
#     }
