import pandas as pd # type: ignore
from sqlalchemy import create_engine # type: ignore

def connexion_base ():
    user = "hater_bi"
    mot_de_passe = "x1998112"
    host = "localhost"
    port = 3306
    nom_base = "fraud_db"
    
    url = f"mysql+pymysql://{user}:{mot_de_passe}@{host}:{port}/{nom_base}"
    moteur = create_engine(url)
    
    return moteur