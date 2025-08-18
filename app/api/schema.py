from fastapi import FastAPI # type: ignore
from enum import Enum
from pydantic import BaseModel  # type: ignore
from typing import Optional
from datetime import datetime

#Modele de transaction 
class transaction_recu(BaseModel):
    id_tx : str
    timestamp : datetime
    id_sender : str 
    id_receiver : Optional[str] = None
    amount : float
    currency : str
    transaction_code: str        # Code ISO 20022 (ex: RCDT001)
    mode : Optional[str] = None       #mobile, cb, virement, agence
    natur_tx : Optional[str] = None  #achat, retrait, virement
    simulated : Optional[bool ] = None
    status: Optional[str] = None
    sender_balance_before : Optional [float] = None      #solde du compte de l’expéditeur juste avant la transaction
    sender_balance_after : Optional [float] = None        
    sender_avg_tx_amount_30d : Optional [float] = None   #Moyenne des montants envoyés sur les 30 derniers jours
    sender_last_tx_date : Optional [datetime] = None     #Dernière date à laquelle ce sender a effectué une transaction.
    statut_compte : Optional [str] = None
    type_compte : Optional [str] = None
    
    