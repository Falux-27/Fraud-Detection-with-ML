import torch
import os

source = r"C:\Users\HP\Desktop\Fraud_detection_project\data\data_processed"
nom_fichier = "dataset_final_shuffled.pt"
chemin = os.path.join(source, nom_fichier)

# Chargement du jeu de données
    # Le paramètre weights_only=False est nécessaire pour charger la structure complète
dataset = torch.load(chemin, weights_only=False)
print(f"Le dataset contient {len(dataset)} graphes.\n")

    # Affichage des statistiques de base et de l'équilibre
num_total = len(dataset)
num_fraud = sum(data.y.item() for data in dataset)
num_normal = num_total - num_fraud
    
print("##########Statistiques Globales#########")
print(f"Total de Graphes : {num_total}")
print(f"Graphes Normaux (Label 0) : {num_normal}")
print(f"Graphes Anormaux (Label 1) : {num_fraud}")
print("-" * 30)

print("\n Inspection des 5 premier objets PYG")
for i, data in enumerate(dataset[100:105]):
        print(f"\n[Graphe n°{i+1}] (Label: {data.y.item()})")
        
        #Caractéristiques structurelles
        print(f"  - Nombre de nœuds (comptes): {data.num_nodes}")
        print(f"  - Nombre d'arêtes (transactions): {data.num_edges}")
        
        #Caractéristiques des features (dimensions)
        print(f"  - Dimensions des Features de Nœuds (x): {data.x.shape} ({data.x.shape[1]} features)")
        print(f"  - Dimensions des Features d'Arêtes (edge_attr): {data.edge_attr.shape} ({data.edge_attr.shape[1]} features)")
        
        # Aperçu des données pour le premier objet seulement
        if i == 0:
            print("\n  Aperçu des Caractéristiques des Nœuds (x[0:3]) :")
            # Affiche les 3 premières lignes et les 5 premières colonnes (les caractéristiques)
            print(data.x[:3, :5]) 
            print("\n  Aperçu des Caractéristiques des Arêtes (edge_attr[0:3]) :")
            print(data.edge_attr[:3])
            print("\n  Aperçu de la Connectivité (edge_index[0:2, 0:5]) :")
            # Affiche l'indice de l'émetteur (ligne 0) et du récepteur (ligne 1) pour les 5 premières arêtes
            print(data.edge_index[:, :5]) 
