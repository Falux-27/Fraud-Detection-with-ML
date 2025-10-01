import random
import torch
import os

# Définir le chemin du fichier que vous avez sauvegardé
dossier_sortie = r"C:\Users\HP\Desktop\Fraud_detection_project\data\data_processed"
nom_fichier_entree = "dataset_final_enriched.pt"
chemin_du_fichier = os.path.join(dossier_sortie, nom_fichier_entree)

try:
    dataset = torch.load(chemin_du_fichier, weights_only=False)
    print(f"Chargé avec succès. Il contient {len(dataset)} graphes.")
except Exception as e:
    print(f"Erreur lors du chargement: {e}")
    exit()

print("Mélange du jeu de données...")
random.shuffle(dataset)

#Vérification du mélange
print("\nLabels des 20 premiers graphes après mélange :")
for data_obj in dataset[:20]:
    print(data_obj.y.item(), end=" ")
print("\n")

# Sauvegardez le jeu de données mélangé  
nom_fichier_shuffled = "dataset_final_shuffled.pt"
chemin_shuffled = os.path.join(dossier_sortie, nom_fichier_shuffled)
torch.save(dataset, chemin_shuffled)
































































# import torch
# import os

# # Définissez le chemin du fichier que vous avez sauvegardé
# dossier_sortie = r"C:\Users\HP\Desktop\Fraud_detection_project\data\data_processed"
# nom_fichier_sortie = "dataset_final_enriched.pt"
# chemin_du_fichier = os.path.join(dossier_sortie, nom_fichier_sortie)

# try:
#     # Chargez le jeu de données en désactivant weights_only
#     dataset = torch.load(chemin_du_fichier, weights_only=False)
#     print(f"✅ Le jeu de données a été chargé avec succès. Il contient {len(dataset)} graphes.\n")

#     # Affichez la structure des 10 premiers graphes pour inspection
#     print("--- Inspection des 10 premiers graphes ---")
#     for i, data in enumerate(dataset[:10]):
#         print(f"\nGraphe n°{i+1}:")
#         print(f"  - Nombre de nœuds (comptes): {data.num_nodes}")
#         print(f"  - Nombre d'arêtes (transactions): {data.num_edges}")
#         print(f"  - Nombre de caractéristiques de nœuds: {data.x.shape[1]}")
#         print(f"  - Nombre de caractéristiques d'arêtes: {data.edge_attr.shape[1]}")
#         print(f"  - Label (y): {data.y.item()} (0 pour normal, 1 pour anormal)")
        
#         if i == 0:
#             print("\n  Aperçu des caractéristiques des nœuds (x):")
#             print(data.x)
#             print("\n  Aperçu des caractéristiques des arêtes (edge_attr):")
#             print(data.edge_attr)
        
# except FileNotFoundError:
#     print(f"❌ Erreur: Le fichier '{chemin_du_fichier}' n'a pas été trouvé. Veuillez vérifier le chemin d'accès.")
# except Exception as e:
#     print(f"❌ Erreur lors du chargement du fichier: {e}")

