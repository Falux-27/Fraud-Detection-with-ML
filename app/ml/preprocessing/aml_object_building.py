import string
from datetime import datetime
import torch
import json
import pickle
from torch_geometric.data import Data
import os
import networkx as nx
import numpy as np
import math
from collections import defaultdict

def create_pyg_data_object_enriched(subgraph_data, label):
    # Fonction principale qui crée un objet Data PyTorch Geometric enrichi
    # subgraph_data: données du sous-graphe (dict ou graphe NetworkX)
    # label: étiquette (0=normal, 1=anormal)
    
    # Convertir le dictionnaire JSON en graphe NetworkX si nécessaire
    if isinstance(subgraph_data, dict):
        # Si les données sont un dictionnaire, les convertir en graphe NetworkX
        subgraph = nx.node_link_graph(subgraph_data)
    else:
        # Sinon, on utilise directement les données comme graphe
        subgraph = subgraph_data

    # Vérifier si le graphe est valide (contient des nœuds et des arêtes)
    if len(subgraph.nodes()) == 0 or len(subgraph.edges()) == 0:
        # Si le graphe est vide, retourner None (ignorer ce motif)
        return None 

    # Création du dictionnaire pour stocker les informations financières par nœud
    node_financial_data = defaultdict(lambda: {
        'total_sent': 0.0,
        'total_received': 0.0,
        'amounts_sent': [],
        'amounts_received': [],
        'currencies': defaultdict(int),
        'timestamps': [],
        'payment_formats': defaultdict(int)
    })

    # Parcourir toutes les arêtes pour collecter les données financières
    for sender, receiver, edge_data in subgraph.edges(data=True):
        # Extraire le montant reçu de l'arête, par défaut 0
        amount_received = float(edge_data.get('Amount Received', 0))
        # Extraire le montant payé de l'arête, par défaut 0
        amount_paid = float(edge_data.get('Amount Paid', 0))
        # Extraire la devise de réception, par défaut 'US Dollar'
        receiving_currency = edge_data.get('Receiving Currency', 'US Dollar')
        # Extraire le format de paiement, par défaut 'Unknown'
        payment_format = edge_data.get('Payment Format', 'Unknown')
        # Extraire l'horodatage de la transaction
        timestamp_str = edge_data.get('Timestamp')
        
        # Convertir la chaîne de caractères en objet datetime
        try:
            #Parsing timestamp au format "%Y/%m/%d %H:%M"
            timestamp = datetime.strptime(timestamp_str, "%Y/%m/%d %H:%M")
        except (ValueError, KeyError):
            # Si le parsing échoue, on utilise la date/heure actuelle
            timestamp = datetime.now()
        
        # Mettre à jour les données financières du receveur
        node_financial_data[receiver]['total_received'] += amount_received
        node_financial_data[receiver]['amounts_received'].append(amount_received)
        node_financial_data[receiver]['currencies'][receiving_currency] += 1
        node_financial_data[receiver]['timestamps'].append(timestamp)
        node_financial_data[receiver]['payment_formats'][payment_format] += 1
        
        # Mettre à jour les données financières de l'expéditeur
        node_financial_data[sender]['total_sent'] += amount_paid
        node_financial_data[sender]['amounts_sent'].append(amount_paid)
        node_financial_data[sender]['timestamps'].append(timestamp)
        node_financial_data[sender]['payment_formats'][payment_format] += 1

    # Créer une liste des nœuds du sous-graphe
    subgraph_nodes = list(subgraph.nodes())
    # Créer un mapping des nœuds vers leurs indices locaux (0, 1, 2, ...)
    local_node_map = {node: i for i, node in enumerate(subgraph_nodes)}
    
    # Calculer les métriques de centralité et de clustering
    try:
        # Calculer le PageRank de chaque nœud
        pagerank_scores = nx.pagerank(subgraph)
        # Calculer la centralité d'intermédiarité de chaque nœud
        betweenness_scores = nx.betweenness_centrality(subgraph)
        # Calculer le coefficient de clustering de chaque nœud
        clustering_scores = nx.clustering(subgraph)
    except:
        # Si le calcul échoue, assigner des valeurs par défaut (0.0) à tous les nœuds
        pagerank_scores = {node: 0.0 for node in subgraph.nodes()}
        betweenness_scores = {node: 0.0 for node in subgraph.nodes()}
        clustering_scores = {node: 0.0 for node in subgraph.nodes()}

    # Créer la liste des caractéristiques pour chaque nœud
    node_features_list = []
    for node in subgraph_nodes:
        # Calculer le degré entrant (nombre d'arêtes entrantes)
        degree_in = subgraph.in_degree(node)
        # Calculer le degré sortant (nombre d'arêtes sortantes)
        degree_out = subgraph.out_degree(node)
        # Calculer le degré total
        total_degree = degree_in + degree_out
        
        # Récupérer les scores de centralité pour ce nœud
        pagerank = pagerank_scores.get(node, 0.0)
        betweenness = betweenness_scores.get(node, 0.0)
        clustering = clustering_scores.get(node, 0.0)
        
        # Calculer le ratio degré entrant / degré total
        if total_degree > 0:
            in_out_ratio = degree_in / total_degree
        else:
            in_out_ratio = 0.0
            
        # Récupérer les données financières de ce nœud
        financial_data = node_financial_data[node]
        total_sent = financial_data['total_sent']
        total_received = financial_data['total_received']
        
        # Calculer le solde net (reçu - envoyé)
        net_balance = total_received - total_sent
        # Appliquer une transformation logarithmique au solde net (garde le signe)
        log_net_balance = np.sign(net_balance) * np.log1p(abs(net_balance))
        
        # Calculer la moyenne des montants envoyés
        avg_sent = np.mean(financial_data['amounts_sent']) if financial_data['amounts_sent'] else 0.0
        # Calculer la moyenne des montants reçus
        avg_received = np.mean(financial_data['amounts_received']) if financial_data['amounts_received'] else 0.0
        
        # Calculer la variance des montants envoyés
        var_sent = np.var(financial_data['amounts_sent']) if len(financial_data['amounts_sent']) > 1 else 0.0
        # Calculer la variance des montants reçus
        var_received = np.var(financial_data['amounts_received']) if len(financial_data['amounts_received']) > 1 else 0.0

        # Calculer le ratio des variances (envoyé / reçu)
        var_ratio = var_sent / var_received if var_received > 0 else 0.0
        
        # Calculer le ratio de la devise principale
        if financial_data['currencies']:
            total_currencies = sum(financial_data['currencies'].values())
            main_currency_ratio = max(financial_data['currencies'].values()) / total_currencies
        else:
            main_currency_ratio = 0.0
        
        # Compter le nombre de devises uniques utilisées par ce nœud
        num_unique_currencies = len(financial_data['currencies'])
        
        # Calculer le ratio du format de paiement le plus utilisé
        if financial_data['payment_formats']:
            total_payment_formats = sum(financial_data['payment_formats'].values())
            main_payment_format_ratio = max(financial_data['payment_formats'].values()) / total_payment_formats
        else:
            main_payment_format_ratio = 0.0

        # Calculer le score de "burstiness" (irrégularité temporelle)
        burstiness_score = 0.0
        # Trier les timestamps par ordre chronologique
        timestamps = sorted(financial_data['timestamps'])
        if len(timestamps) > 1:
            # Calculer les intervalles entre transactions successives
            intervals = [(timestamps[i] - timestamps[i-1]).total_seconds() for i in range(1, len(timestamps))]
            if len(intervals) > 0:
                # Calculer la moyenne et l'écart-type des intervalles
                mean_interval = np.mean(intervals)
                std_interval = np.std(intervals)
                # Calculer le score de burstiness normalisé
                if mean_interval + std_interval > 0:
                    burstiness_score = (std_interval - mean_interval) / (std_interval + mean_interval)

        # Ajouter toutes les caractéristiques de ce nœud à la liste
        node_features_list.append([
            float(degree_in),
            float(degree_out),
            float(total_degree),
            float(clustering),
            float(betweenness),
            float(pagerank),
            float(in_out_ratio),
            float(total_sent),
            float(total_received),
            float(net_balance),
            float(log_net_balance),
            float(var_ratio),
            float(num_unique_currencies),
            float(main_payment_format_ratio),
            float(avg_sent),
            float(avg_received),
            float(var_sent),
            float(var_received),
            float(main_currency_ratio),
            float(burstiness_score)
        ])

    # Convertir la liste des caractéristiques en tenseur PyTorch
    x = torch.tensor(node_features_list, dtype=torch.float)
    
    # Listes pour stocker les indices et attributs des arêtes
    edge_index_list = []
    edge_attr_list = []

    # Parcourir toutes les arêtes pour créer les attributs d'arêtes
    for sender, receiver, data_edge in subgraph.edges(data=True):
        # Récupérer l'ID local de l'expéditeur
        sender_id = local_node_map[sender]
        #Idem
        receiver_id = local_node_map[receiver] 
        # Ajouter cette arête à la liste des indices d'arêtes
        edge_index_list.append([sender_id, receiver_id])

        # Parser l'horodatage de la transaction
        timestamp = datetime.strptime(data_edge["Timestamp"], "%Y/%m/%d %H:%M")
        
        # Normaliser l'heure (0-23 devient 0.0-1.0)
        hour = timestamp.hour / 23.0
        # Normaliser le jour de la semaine (0-6 devient 0.0-1.0)
        day_of_week = timestamp.weekday() / 6.0
        # Indicateur binaire pour le weekend (samedi=5, dimanche=6)
        is_weekend = 1.0 if timestamp.weekday() >= 5 else 0.0
        
        # Récupérer le montant reçu et appliquer une transformation logarithmique
        amount_received = float(data_edge.get('Amount Received', 0))
        log_amount = np.log1p(amount_received)
        
        # Créer le vecteur d'attributs pour cette arête
        edge_attr = [
            hour, 
            day_of_week, 
            is_weekend, 
            log_amount,
        ]
        # Ajouter les attributs de cette arête à la liste
        edge_attr_list.append(edge_attr)
    
    # Vérifier qu'il y a au moins une arête
    if not edge_index_list:
        # Si pas d'arêtes, retourner None
        return None

    # Convertir la liste des indices d'arêtes en tenseur PyTorch et transposer
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    # Convertir la liste des attributs d'arêtes en tenseur PyTorch
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
    # Créer le tenseur d'étiquette (label)
    y = torch.tensor([label], dtype=torch.long)
    
    # Créer l'objet Data PyTorch Geometric final
    data_obj = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
    # Retourner l'objet Data créé
    return data_obj

#SCRIPT PRINCIPAL 

#Liste des fichiers contenant les motifs anormaux (blanchiment d'argent)
fichiers_anormaux = [
    r"data\IBM Transactions\HI-Large_Patterns.txt",
    r"data\IBM Transactions\HI-Medium_Patterns.txt",
    r"data\IBM Transactions\HI-Small_Patterns.txt",
    r"data\IBM Transactions\LI-Large_Patterns.txt",
    r"data\IBM Transactions\LI-Medium_Patterns.txt",
    r"data\IBM Transactions\LI-Small_Patterns.txt"
]

#Liste pour stocker les motifs extraits
patterns = []
# Variable temporaire pour stocker le motif en cours de lecture
actuel_pattern = []

# Parcourir chaque fichier de motifs anormaux
for file in fichiers_anormaux:
    # Ouvrir le fichier en mode lecture avec encodage UTF-8
    with open(file, "r", encoding="utf-8") as fichier:
        # Lire chaque ligne du fichier
        for line in fichier:
            # Supprimer les espaces en début et fin de ligne
            line = line.strip()
            # Ignorer les lignes vides
            if not line:
                continue
            # Détecter le début d'un nouveau motif de blanchiment
            if line.startswith("BEGIN LAUNDERING ATTEMPT"):
                # Initialiser un nouveau motif vide
                actuel_pattern = []
            # Détecter la fin du motif courant
            elif line.startswith("END LAUNDERING ATTEMPT"):
                # Ajouter le motif complet à la liste des motifs
                patterns.append(actuel_pattern)
            else:
                # Cette ligne contient une transaction, la parser
                transaction = line.split(",")
                # Ajouter la transaction au motif courant
                actuel_pattern.append(transaction)
 
# Création des graphes NetworkX anormaux à partir des motifs extraits
anormal_graphs = []
# Parcourir chaque motif extrait
for motif in patterns:
    # Créer un graphe orienté multiple (plusieurs arêtes possibles entre mêmes nœuds)
    G = nx.MultiDiGraph()
    # Parcourir chaque transaction dans ce motif
    for transac in motif:
        # Format: [timestamp, from_bank, from_account, to_bank, to_account, amount, etc.]
        # Extraire l'ID du compte expéditeur (supprimer espaces et quotes)
        sender = transac[2].strip(" '")
        # Extraire l'ID du compte receveur
        receiver = transac[4].strip(" '")
        
        # Créer un dictionnaire avec toutes les données de l'arête
        edge_data = {
            "Timestamp": transac[0],
            "From Bank": int(transac[1]),
            "From_Account": sender,
            "To Bank": int(transac[3]),
            "To_Account": receiver,
            "Amount Received": float(transac[5]),
            "Receiving Currency": transac[6],
            "Amount Paid": float(transac[7]),
            "Payment Currency": transac[8],
            "Payment Format": transac[9],
            "Is Laundering": int(transac[10]),
        }
        # Ajouter les nœuds au graphe (expéditeur et receveur)
        G.add_node(sender)
        G.add_node(receiver)
        # Ajouter l'arête avec ses données
        G.add_edge(sender, receiver, **edge_data)
    
    # Vérifier que le graphe a au moins une arête
    if len(G.edges) > 0:
        # Ajouter le graphe à la liste des graphes anormaux
        anormal_graphs.append(G)

# Chemins vers les fichiers contenant les graphes normaux (transactions légitimes)
chemin_normaux = [
    r"C:\Users\HP\Desktop\Fraud_detection_project\data\IBM Transactions\Graph_Trans_normaux.json",
    r"C:\Users\HP\Desktop\Fraud_detection_project\data\IBM Transactions\Graph_Trans_normaux_3.json"
]

# Liste pour stocker tous les graphes normaux fusionnés
all_normal_graphs = []
# Parcourir chaque fichier de graphes normaux
for file_path in chemin_normaux:
    try:
        # Ouvrir et charger le fichier JSON
        with open(file_path, "r") as f:
            graphs = json.load(f)
            # Ajouter tous les graphes de ce fichier à la liste globale
            all_normal_graphs.extend(graphs)
    except FileNotFoundError:
        # Si le fichier n'existe pas, continuer avec le suivant
        continue

# Afficher le nombre total de graphes normaux chargés
print(f"Total fusionné: {len(all_normal_graphs)} graphes normaux.")

# Liste finale pour stocker tous les objets Data PyTorch Geometric
dataset_final = []

# Conversion des motifs anormaux en objets Data
print("Conversion des motifs anormaux...")
for graph in anormal_graphs:
    # Créer un objet Data avec label=1 (anormal)
    data_obj = create_pyg_data_object_enriched(graph, label=1)
    if data_obj:
        # Si la création a réussi, ajouter à la liste finale
        dataset_final.append(data_obj)

# Conversion des graphes normaux en objets Data
print("Conversion des graphes normaux...")
for graph_info in all_normal_graphs:
    # Créer un objet Data avec label=0 (normal) - extraire le graphe de la structure
    data_obj = create_pyg_data_object_enriched(graph_info['graph'], label=0)
    if data_obj:
        # Si la création a réussi, ajouter à la liste finale
        dataset_final.append(data_obj)

print(f"{len(all_normal_graphs)} graphes normaux traités.")
print(f"Total d'objets Data créés : {len(dataset_final)}")

# # Sauvegarde
# dossier_sortie = r"C:\Users\HP\Desktop\Fraud_detection_project\data\data_processed"
# nom_fichier_sortie = "dataset_final_enriched.pt"
# chemin_sortie = os.path.join(dossier_sortie, nom_fichier_sortie)
# torch.save(dataset_final, chemin_sortie)