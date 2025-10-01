from fastapi import FastAPI # type: ignore
from enum import Enum
from pydantic import BaseModel  # type: ignore
from typing import Union


app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Fallou is workin here"}
#Parametre de chemin
@app.get("/home/{name}")
async def home (name):
    return f"Welcome home {name}"
#Parametre de chemin typés
@app.get("/param/{type}")
async def param (type: str):
    type =  "Il s'agit de déclarer le type d'un paramètre de chemin dans la fonction, en utilisant les annotatio"
    return f"Concept : {type}"
    
#Valeurs prédéfinies

@app.get('/valeur predefini/{concept}')
async def valeur_predefini(concept: str):
    concept = "Il consiste a definir un  ensemble de valeur possible pour les parametre de la fonction"
    return f"Concept: {concept}"


#creation de sous classe Enum 
class model_name (str , Enum):
    etudiant1 = "Abdul karim"
    etudiant2 = "Malick Cisse"
    etudiant3 = "Sophie Kyle"
    
@app.get('/students/{name}')
async def affiche_nom (name : model_name):
    if name == model_name.etudiant1:
        return {"nom": name ,
                "message": "Bienvenue au club mon ami"}
        
    elif  name.value == 'Sophie Kyle':
        return{"nom": name ,
                "message": "Bienvenue au club mon ami"} 
        
#Acces au valeur de l'enumeration
class enumeration (str , Enum):
    equipe1 = "Real Madrid"
    equipe2 = "PSG"
    equipe3 = "Inter Miami"

@app.get('/equipes/{team}')
async def equipe (team :enumeration):
    #type d'acces 1
    if team is enumeration.equipe1:
        return {
            'Team': team,
            'text' : "Best club in the world"
        }
    #type d'acces   
    elif team.value != enumeration.equipe1:
        return {
            "nom": team,
            "take": "Just a random team"
        } 


#Corps de la requête
class person (BaseModel):
    nom : str 
    age : int
    adress : str
    poids: Union[float, None] = None
    num : Union [int , None] = None
    
@app.post ("/info/")
async def infos (info : person):
     
    return info