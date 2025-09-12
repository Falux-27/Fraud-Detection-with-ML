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
    
    if isinstance(subgraph_data, dict):
        subgraph = nx.node_link_graph(subgraph_data)
    else:
        subgraph = subgraph_data

    if len(subgraph.nodes()) == 0 or len(subgraph.edges()) == 0:
        return None 

    node_financial_data = defaultdict(lambda: {
        'total_sent': 0.0,
        'total_received': 0.0,
        'amounts_sent': [],
        'amounts_received': [],
        'currencies': defaultdict(int),
        'timestamps': [],
        'payment_formats': defaultdict(int)
    })
    
    for sender, receiver, edge_data in subgraph.edges(data=True):
        amount_received = float(edge_data.get('Amount Received', 0))
        amount_paid = float(edge_data.get('Amount Paid', 0))
        receiving_currency = edge_data.get('Receiving Currency', 'US Dollar')
        payment_format = edge_data.get('Payment Format', 'Unknown')
        timestamp_str = edge_data.get('Timestamp')
        
        try:
            timestamp = datetime.strptime(timestamp_str, "%Y/%m/%d %H:%M")
        except (ValueError, KeyError):
            timestamp = datetime.now()
        
        node_financial_data[receiver]['total_received'] += amount_received
        node_financial_data[receiver]['amounts_received'].append(amount_received)
        node_financial_data[receiver]['currencies'][receiving_currency] += 1
        node_financial_data[receiver]['timestamps'].append(timestamp)
        node_financial_data[receiver]['payment_formats'][payment_format] += 1
        node_financial_data[sender]['total_sent'] += amount_paid
        node_financial_data[sender]['amounts_sent'].append(amount_paid)
        node_financial_data[sender]['timestamps'].append(timestamp)
        node_financial_data[sender]['payment_formats'][payment_format] += 1

    subgraph_nodes = list(subgraph.nodes())
    local_node_map = {node: i for i, node in enumerate(subgraph_nodes)}
    
    try:
        pagerank_scores = nx.pagerank(subgraph)
        betweenness_scores = nx.betweenness_centrality(subgraph)
        clustering_scores = nx.clustering(subgraph)
    except:
        pagerank_scores = {node: 0.0 for node in subgraph.nodes()}
        betweenness_scores = {node: 0.0 for node in subgraph.nodes()}
        clustering_scores = {node: 0.0 for node in subgraph.nodes()}

    node_features_list = []
    for node in subgraph_nodes:
        degree_in = subgraph.in_degree(node)
        degree_out = subgraph.out_degree(node)
        total_degree = degree_in + degree_out
        
        pagerank = pagerank_scores.get(node, 0.0)
        betweenness = betweenness_scores.get(node, 0.0)
        clustering = clustering_scores.get(node, 0.0)
        
        if total_degree > 0:
            in_out_ratio = degree_in / total_degree
        else:
            in_out_ratio = 0.0
            
        financial_data = node_financial_data[node]
        total_sent = financial_data['total_sent']
        total_received = financial_data['total_received']
        
        net_balance = total_received - total_sent
        #logarithme du solde net
        log_net_balance = np.sign(net_balance) * np.log1p(abs(net_balance))
        
        avg_sent = np.mean(financial_data['amounts_sent']) if financial_data['amounts_sent'] else 0.0
        avg_received = np.mean(financial_data['amounts_received']) if financial_data['amounts_received'] else 0.0
        
        var_sent = np.var(financial_data['amounts_sent']) if len(financial_data['amounts_sent']) > 1 else 0.0
        var_received = np.var(financial_data['amounts_received']) if len(financial_data['amounts_received']) > 1 else 0.0

        #ratio de variance
        var_ratio = var_sent / var_received if var_received > 0 else 0.0
        
        if financial_data['currencies']:
            total_currencies = sum(financial_data['currencies'].values())
            main_currency_ratio = max(financial_data['currencies'].values()) / total_currencies
        else:
            main_currency_ratio = 0.0
        
        #nombre de devises uniques
        num_unique_currencies = len(financial_data['currencies'])
        
        #ratio du format de paiement le plus utilisé
        if financial_data['payment_formats']:
            total_payment_formats = sum(financial_data['payment_formats'].values())
            main_payment_format_ratio = max(financial_data['payment_formats'].values()) / total_payment_formats
        else:
            main_payment_format_ratio = 0.0

        burstiness_score = 0.0
        timestamps = sorted(financial_data['timestamps'])
        if len(timestamps) > 1:
            intervals = [(timestamps[i] - timestamps[i-1]).total_seconds() for i in range(1, len(timestamps))]
            if len(intervals) > 0:
                mean_interval = np.mean(intervals)
                std_interval = np.std(intervals)
                if mean_interval + std_interval > 0:
                    burstiness_score = (std_interval - mean_interval) / (std_interval + mean_interval)

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

    x = torch.tensor(node_features_list, dtype=torch.float)
    edge_index_list = []
    edge_attr_list = []

    for sender, receiver, data_edge in subgraph.edges(data=True):
        sender_id = local_node_map[sender]
        receiver_id = local_node_map[receiver] 
        edge_index_list.append([sender_id, receiver_id])

        timestamp = datetime.strptime(data_edge["Timestamp"], "%Y/%m/%d %H:%M")
        
        hour = timestamp.hour / 23.0
        day_of_week = timestamp.weekday() / 6.0
        is_weekend = 1.0 if timestamp.weekday() >= 5 else 0.0
        
        amount_received = float(data_edge.get('Amount Received', 0))
        log_amount = np.log1p(amount_received)
        
        edge_attr = [
            hour, 
            day_of_week, 
            is_weekend, 
            log_amount,
        ]
        edge_attr_list.append(edge_attr)
    
    if not edge_index_list:
        return None

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
    y = torch.tensor([label], dtype=torch.long)
    
    data_obj = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
    return data_obj

#script principal 
fichiers_anormaux = [
    r"data\IBM Transactions\HI-Large_Patterns.txt",
    r"data\IBM Transactions\HI-Medium_Patterns.txt",
    r"data\IBM Transactions\HI-Small_Patterns.txt",
    r"data\IBM Transactions\LI-Large_Patterns.txt",
    r"data\IBM Transactions\LI-Medium_Patterns.txt",
    r"data\IBM Transactions\LI-Small_Patterns.txt"
]

patterns = []
actuel_pattern = []
for file in fichiers_anormaux:
    with open(file, "r", encoding="utf-8") as fichier:
        for line in fichier:
            line = line.strip()
            if not line:
                continue
            if line.startswith("BEGIN LAUNDERING ATTEMPT"):
                actuel_pattern = []
            elif line.startswith("END LAUNDERING ATTEMPT"):
                patterns.append(actuel_pattern)
            else:
                transaction = line.split(",")
                actuel_pattern.append(transaction)
 
#Création des graphes NetworkX anormaux à partir des motifs
anormal_graphs = []
for motif in patterns:
    G = nx.MultiDiGraph()
    for transac in motif:
        # transac: [timestamp, from_bank, from_account, to_bank, to_account, amount, etc.]
        sender = transac[2].strip(" '")
        receiver = transac[4].strip(" '")
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
        G.add_node(sender)
        G.add_node(receiver)
        G.add_edge(sender, receiver, **edge_data)
    if len(G.edges) > 0:
        anormal_graphs.append(G)

#fusion
chemin_normaux = [
    r"C:\Users\HP\Desktop\Fraud_detection_project\data\IBM Transactions\Graph_Trans_normaux.json",
    r"C:\Users\HP\Desktop\Fraud_detection_project\data\IBM Transactions\Graph_Trans_normaux_3.json"
]

all_normal_graphs = []
for file_path in chemin_normaux:
    try:
        with open(file_path, "r") as f:
            graphs = json.load(f)
            all_normal_graphs.extend(graphs)
    except FileNotFoundError:
        continue

print(f"Total fusionné: {len(all_normal_graphs)} graphes normaux.")

#Création des objets Data PyG enrichis
dataset_final = []

# Conversion des motifs anormaux
print("Conversion des motifs anormaux...")
for graph in anormal_graphs:
    data_obj = create_pyg_data_object_enriched(graph, label=1)
    if data_obj:
        dataset_final.append(data_obj)
# Conversion des graphes normaux
print("Conversion des graphes normaux...")
for graph_info in all_normal_graphs:
    data_obj = create_pyg_data_object_enriched(graph_info['graph'], label=0)
    if data_obj:
        dataset_final.append(data_obj)

print(f"{len(all_normal_graphs)} graphes normaux traités.")
print(f"Total d'objets Data créés : {len(dataset_final)}")


# Sauvegarde
dossier_sortie = r"C:\Users\HP\Desktop\Fraud_detection_project\data\data_processed"
nom_fichier_sortie = "dataset_final_enriched.pt"
chemin_sortie = os.path.join(dossier_sortie, nom_fichier_sortie)
torch.save(dataset_final, chemin_sortie)

