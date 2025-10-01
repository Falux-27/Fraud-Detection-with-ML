from fastapi import FastAPI, Request # type: ignore
from app.api.routes_fraud import router as fraud_router
from fastapi.exceptions import RequestValidationError # type: ignore
from fastapi.responses import JSONResponse # type: ignore
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY # type: ignore

app = FastAPI()

app.include_router(fraud_router)


#Gestion des erreurs des données reçues
@app.exception_handler(RequestValidationError)
async def gestion_erreur(request : Request , exc : RequestValidationError):
    liste_error = []
    for erreur in exc.errors(): #Parcours du dict d'erreur
        liste_error.append({
            "champ d'erreur":erreur['loc'][-1],
            "message d'erreur": erreur['msg']
        })
    return JSONResponse (
        status_code = HTTP_422_UNPROCESSABLE_ENTITY, 
        content = {
            'Infos': "Erreur de validation de donnée",
            "errors": liste_error
        }
    )