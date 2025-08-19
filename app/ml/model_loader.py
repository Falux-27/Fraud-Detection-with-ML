import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Redéfinir SimpleMLPClassifier localement pour éviter les imports
class SimpleMLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout), 
        
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.layers(x)

# Variables globales 
ldam_scaler = None  
ldam_model = None
_models_loaded = False

def load_models():
    global ldam_scaler, ldam_model, _models_loaded
    
    if not _models_loaded:
        try:
            # Charger le scaler
            ldam_scaler = joblib.load("models/ldam_scaler.pkl")
            
            # Charger les données du modèle (avec weights_only=False pour PyTorch 2.6+)
            model_data = torch.load("models/models_ldam_pure.pth", map_location='cpu', weights_only=False)
            
            # Vérifier le format
            if isinstance(model_data, dict) and 'model_state' in model_data:
                # Nouveau format - dictionnaire
                print("Format dictionnaire détecté")
                
                # Recréer le modèle PyTorch
                pytorch_model = SimpleMLPClassifier(
                    input_dim=model_data['model_architecture']['n_features_in_'],
                    num_classes=model_data['model_architecture']['n_classes_'],
                    hidden_dim=model_data['model_architecture']['hidden_dim']
                )
                
                # Charger les poids
                pytorch_model.load_state_dict(model_data['model_state'])
                pytorch_model.eval()
                
                # Créer le prédicteur
                class SimpleLDAMPredictor:
                    def __init__(self, model, scaler, classes):
                        self._model = model
                        self._scaler = scaler
                        self.classes_ = classes
                    
                    def predict(self, X):
                        X = np.asarray(X)
                        X_scaled = self._scaler.transform(X)
                        
                        self._model.eval()
                        with torch.no_grad():
                            X_tensor = torch.FloatTensor(X_scaled)
                            outputs = self._model(X_tensor)
                            predictions = torch.argmax(outputs, dim=1)
                            return predictions.numpy()
                    
                    def predict_proba(self, X):
                        X = np.asarray(X)
                        X_scaled = self._scaler.transform(X)
                        
                        self._model.eval()
                        with torch.no_grad():
                            X_tensor = torch.FloatTensor(X_scaled)
                            outputs = self._model(X_tensor)
                            probas = F.softmax(outputs, dim=1)
                            return probas.numpy()
                
                ldam_model = SimpleLDAMPredictor(
                    pytorch_model, 
                    ldam_scaler, 
                    model_data['model_architecture']['classes_']
                )
                
            else:
                ldam_model = model_data
                
                if hasattr(ldam_model, '_model'):
                    ldam_model._model.eval()
            
            _models_loaded = True
            
        except Exception as e:
            print(f"Erreur chargement modèle: {e}")
            import traceback
            traceback.print_exc()
            raise e

def predict_fraud(data):
    """Prédire la fraude avec LDAM"""
    if not _models_loaded:
        load_models()
    
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    
    return ldam_model.predict(data)

def predict_fraud_proba(data):
    """Prédire les probabilités avec LDAM"""
    if not _models_loaded:
        load_models()
    
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    
    return ldam_model.predict_proba(data)

voting_model = None
_voting_loaded = False

def load_voting_model():
    global voting_model, _voting_loaded
    if not _voting_loaded:
        try:
            voting_model = joblib.load("models/voting_classifier.pkl")
            _voting_loaded = True
             
        except Exception as e:
            print(f"Erreur chargement VotingClassifier: {e}")
            import traceback
            traceback.print_exc()
            raise e

def predict_voting(data):
    #VotingClassifie
    if not _voting_loaded:
        load_voting_model()
    
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    
    return voting_model.predict(data)