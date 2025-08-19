import pandas as pd
import numpy as np
import joblib
from collections import Counter
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
import xgboost as xgb
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.metrics import classification_report
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')


#Modele d'attribution des logit aux classes
class SimpleMLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.layers = nn.Sequential(
            #Couche 1 avec 128 neurones cachées 
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),            #On desactive 30% des neurones pour eviter l'overfitting
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout), 
        
            nn.Linear(hidden_dim, num_classes)
            
        )
    
    def forward(self, x):
        return self.layers(x)
    
#Fonction perte LDAM personnalisée 
class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, s=30, drw_start_epoch=8, beta=0.9999, progressive_drw=True):
        super().__init__()
        #Marges LDAM par classe
        m_list = 1.0 / torch.sqrt(torch.tensor(cls_num_list, dtype=torch.float32))
        facteur_normalization = max_m / torch.max(m_list)
        m_list = m_list *  facteur_normalization
        self.register_buffer('m_list', m_list)

        self.s = s
        self.drw_start_epoch = drw_start_epoch
        self.beta = beta
        self.progressive_drw = progressive_drw

        #Poids DRW
        nbr_elt_par_class = torch.tensor(cls_num_list, dtype=torch.float32)
        nombre_effective = 1.0 - torch.pow(torch.tensor(beta), nbr_elt_par_class) # 1 - beta^ny
        #nombre_effective = torch.clamp(nombre_effective, min=1e-7)

        poids_par_class = (1.0 - beta) / nombre_effective
        poids_class_normalized = poids_par_class / torch.sum(poids_par_class) * len(cls_num_list)
        self.register_buffer('poids_class_normalized', poids_class_normalized)

        poids_uniform = torch.ones(len(cls_num_list), dtype=torch.float32) / len(cls_num_list) #Ne jamais touché jamais jamais...
        self.register_buffer('poids_uniform', poids_uniform)
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch
    #Appliquer les poids relative à l'epoch actuel
    def get_current_weights(self):
        if self.current_epoch < self.drw_start_epoch:
            return self.poids_uniform
        elif self.progressive_drw:
            niveau_progression =  (self.current_epoch - self.drw_start_epoch) / 5
            alpha = min(1.0, niveau_progression)   
            return (1 - alpha) * self.poids_uniform + alpha * self.poids_class_normalized
        else:
            return self.poids_class_normalized #application brusque poids DRW

    def forward(self, logits, target):
        m_list = self.m_list.to(logits.dtype) #liste marge convertis aux types des logits
        weight = self.get_current_weights().to(logits.dtype) #Les poids de classe en fonction du niveau d'epoch                                                                                                           
        mask = torch.zeros_like(logits, dtype=torch.uint8) #masque similaire à logit 
        target_reshaped = target.view(-1 , 1)
        mask.scatter_(1, target_reshaped, 1.0) #Dans le masque on applique des 1 sur la position de la vraie class de chaque ligne
        batch_m = m_list[target].view(-1, 1)
        x_m = logits - batch_m * mask
        return F.cross_entropy(self.s * x_m, target, weight=weight)

# Entrainement de SimpleMLPClassifier avec LDAM dans skearn
class LDAMClassifier(BaseEstimator, ClassifierMixin):
    
    estimator_type = "classifier"  #je suis un classificateur baby
    
    #Methode spéciale de compatibilité scikit-learn
    def __sklearn_tags__(self):
        #Tags minimaux pour être accepté comme estimateur par sklearn
        return SimpleNamespace(
            estimator_type="classifier", 
            requires_fit=True,  #appeler fit() avant predict()
            non_deterministic=False,
            allow_nan=False,
            poor_score=False
        )
        
    def __init__(self, hidden_dim=256, max_m=0.5, s=30, 
                 drw_start_epoch=8, beta=0.9999, progressive_drw=True, 
                 max_epochs=150, lr=0.001, batch_size=140, random_state=42):
        self.hidden_dim = hidden_dim
        self.max_m = max_m
        self.s = s
        self.drw_start_epoch = drw_start_epoch
        self.beta = beta
        self.progressive_drw = progressive_drw
        self.max_epochs = max_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state
    
    def _get_tags(self):
        return {"classifier": True}

#Specifier comment l'entrainement du MLP se fasse dans sklearn
    def fit(self, X, y):
        # Convertir X et y en numpy si besion
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X.shape[1]                #Attributs obligatoires sklearn
        class_counts = Counter(y)
        cls_num_list = [class_counts[i] for i in range(self.n_classes_)]
        #Standardisation
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        #Transformation des des X en tensor pour pouvoir l'utiliser sur le MLP
        X_tensor = torch.FloatTensor(X_scaled) 
        y_tensor = torch.LongTensor(y) 
        #Modèle SimpleMLPClassifier
        self._model = SimpleMLPClassifier(
            self.n_features_in_, self.n_classes_, self.hidden_dim
        ) 
        #Fonction perte LDAM
        self._criterion = LDAMLoss(
            cls_num_list=cls_num_list, #nombre d'element par classe
            max_m=self.max_m,
            s=self.s,
            drw_start_epoch=self.drw_start_epoch,
            beta=self.beta,
            progressive_drw=self.progressive_drw
        ) 
        #Optimizer
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        # DataLoading
        dataset = TensorDataset(X_tensor, y_tensor) #dataset personnalisé d'entrainement
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True) #je suis le serveur baby
        #Entrainement
        self._model.train()   #Indiquer que le modèle est en mode entraînement
        for epoch in range(self.max_epochs):
            self._criterion.set_epoch(epoch) #adaption des poids DRW en fonction de l'epoch
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self._model(batch_X) #pass forward
                loss = self._criterion(outputs, batch_y)
                loss.backward() #gradient
                optimizer.step()
                epoch_loss += loss.item()
            if epoch % 10 == 0:
                print(f"LDAM Epoch {epoch}, Loss: {epoch_loss/len(dataloader):.4f}")
        return self
    
#Prediction 
    def predict(self, X):
        X = np.asarray(X)
        X_scaled = self._scaler.transform(X)
        self._model.eval() #on active le mode evaluation baby
        with torch.no_grad():  #desactiver la construction de graphe
            device = next(self._model.parameters()).device
            X_tensor = torch.FloatTensor(X_scaled).to(device)
            outputs = self._model(X_tensor)
            predictions = torch.argmax(outputs, dim=1)
            return predictions.numpy()
        
    #Probabilitées predites
    def predict_proba(self, X):
        X = np.asarray(X)
        X_scaled = self._scaler.transform(X)
        self._model.eval()
        with torch.no_grad():
            device = next(self._model.parameters()).device
            X_tensor = torch.FloatTensor(X_scaled).to(device)
            outputs = self._model(X_tensor)
            probas = F.softmax(outputs, dim=1)
            return probas.cpu().numpy()

data = joblib.load('data/data_processed/train_test_data.pkl')
final_x_train = data['final_x_train']
x_test = data['x_test']
final_y_train = data['final_y_train']
y_test = data['y_test']

nb = Counter(final_y_train)
nb_ = Counter(y_test)
# print(f"Distribution train: {nb}")
# print(f"Distribution test: {nb_}")

models = {
    #'logistic': LogisticRegression(),
    'xgb': xgb.XGBClassifier(),
    'rand_f': RandomForestClassifier()
}

# param_grid_log = {
#     #"C": [0.01, 1, 10],
#     'solver': ['liblinear'],
#     'class_weight': ['balanced']
# }  

param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10],
    'class_weight': ['balanced']
}

param_grid_xgb = {
    'max_depth': [3, 5], 
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 200]
}

#GRIDSEARCH 
#grid_log = GridSearchCV(estimator=models['logistic'], param_grid=param_grid_log, 
                       #scoring='f1', cv=5, n_jobs=-1, verbose=1)
grid_rf = GridSearchCV(estimator=models['rand_f'], param_grid=param_grid_rf,
                      cv=5, scoring='f1', n_jobs=-1, verbose=1)
grid_xgb = GridSearchCV(estimator=models['xgb'], param_grid=param_grid_xgb, 
                       scoring='f1', cv=5, n_jobs=-1, verbose=1)

#TRAINING GRIDSEARCH 
#grid_log.fit(final_x_train, final_y_train)
grid_rf.fit(final_x_train, final_y_train)
grid_xgb.fit(final_x_train, final_y_train)

#best_log = grid_log.best_estimator_
best_rf = grid_rf.best_estimator_
best_xgb = grid_xgb.best_estimator_

# print(f"\nScores optimaux:")
# print(f"Logistic: {grid_log.best_score_:.4f}")
print(f"Random Forest: {grid_rf.best_score_:.4f}")
print(f"XGBoost: {grid_xgb.best_score_:.4f}")

#MODÈLE LDAM 
ldam_model = LDAMClassifier(
    hidden_dim=256,
    max_m=0.5,
    max_epochs=150,
    lr=0.001
)

estimators_list = [
    ('RF', best_rf),
    ('Xgb', best_xgb)
]
supervised_model= VotingClassifier( 
        estimators=estimators_list,
        voting='soft',
        weights=[1, 1.5]
    )
print("\nEntraînement du modèles final...")
supervised_model.fit(final_x_train, final_y_train)
class_predict = supervised_model.predict(x_test)
score_model = classification_report(y_test, class_predict, output_dict=True)
score_model_ = pd.DataFrame(score_model).T
print(score_model_)

class_predict = ldam_model.predict(x_test)
score_model = classification_report(y_test, class_predict, output_dict=True)
score_model_ = pd.DataFrame(score_model).T
print(score_model_)
    
#Sauvegarde du model
model_save_dict = {
    'model_state': ldam_model._model.state_dict(),
    'model_architecture': {
        'n_features_in_': ldam_model.n_features_in_,
        'n_classes_': ldam_model.n_classes_,
        'hidden_dim': ldam_model.hidden_dim,
        'classes_': ldam_model.classes_
    }
}
torch.save(model_save_dict, "models/models_ldam_pure.pth")
joblib.dump(ldam_model._scaler, 'models/ldam_scaler.pkl')

#Sauvegarde du voting classifier sans LDAM model
joblib.dump(supervised_model, "models/voting_classifier.pkl" )


