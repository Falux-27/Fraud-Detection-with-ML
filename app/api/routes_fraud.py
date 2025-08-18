from fastapi import APIRouter  # type: ignore
from fastapi import Request  # type: ignore
from fastapi.responses import JSONResponse # type: ignore
from app.api.schema import transaction_recu
from app.core.preparation import final_treatment
from app.core.evaluator import evaluer_transaction_par_règles
from app.db.db_connect import connexion_base
from app.ml.model_loader import predict_fraud, predict_fraud_proba, predict_voting
from app.core.rules_engine import client
from datetime import datetime
import numpy as np
import joblib
router = APIRouter()

@router.post('/predict/fraud')
async def predic_fraud(tx: transaction_recu):
     # Création d’un client fictif (on dispose pas de client encore dans la base)
    client_obj = client(
        id_cust="XQ1354",
        prenom="Malick",
        nom="Gueye",
        country_code='SN',
        risk_profile="N/A",
        date_inscription=datetime(2022, 1, 1)
    )
    tx = tx.dict()
    df_treated = final_treatment(tx)
    resultat_rules = evaluer_transaction_par_règles(tx,client_obj)
    resultat_rules["details"]["date_inscription"] = resultat_rules["details"]["date_inscription"].isoformat()
    ml_data = df_treated.values
    ml_prediction = predict_fraud(ml_data)[0]
    ml_probabilities = predict_fraud_proba(ml_data)[0]
    ml_prediction = predict_voting(ml_data)[0]
    fraud_probability = float(ml_probabilities[1]) if len(ml_probabilities) > 1 else float(ml_probabilities[0])
    
    #Conversion des colonnes datetime pour les rendre JSONifiable
    for col in df_treated.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns:
        df_treated[col] = df_treated[col].astype(str)
    
    return JSONResponse(
        status_code=200,
        content={
            "status": "success",
            "preprocessed_data": df_treated.to_dict(orient="records"),
            "rules_evaluation": resultat_rules,
            "ml_prediction": {
                "prediction": int(ml_prediction),
                "fraud_probability": fraud_probability,
                "all_probabilities": ml_probabilities.tolist()
            }
        }
    )