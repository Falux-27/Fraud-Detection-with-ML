import json
import networkx as nx
import random
import os
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

# Chargement des données normales existantes ---
try:
    with open(r"Graph_Trans_normaux.json", "r") as f:
        existing_normal_graphs = json.load(f)
except FileNotFoundError:
    print("Erreur: Le fichier 'Graph_Trans_normaux.json' n'a pas été trouvé.")
    exit()

template_graphs = [nx.node_link_graph(d['graph']) for d in existing_normal_graphs]

if not template_graphs:
    print("Erreur: Le fichier JSON ne contient pas de données de graphes valides.")
    exit()

# Analyse des données existantes pour la génération réaliste 
def analyze_existing_data(graphs):
    """Analyse les données existantes pour créer des distributions réalistes."""
    bank_ids = set()
    currencies = defaultdict(int)
    payment_formats = defaultdict(int)
    amounts = []
    
    for graph in graphs:
        for u, v, data in graph.edges(data=True):
            bank_ids.add(data.get('From Bank', 0))
            bank_ids.add(data.get('To Bank', 0))
            currencies[data.get('Receiving Currency', 'US Dollar')] += 1
            currencies[data.get('Payment Currency', 'US Dollar')] += 1
            payment_formats[data.get('Payment Format', 'Credit Card')] += 1
            amounts.append(float(data.get('Amount Received', 0)))
    
    return {
        'bank_ids': list(bank_ids),
        'currencies': list(currencies.keys()),
        'payment_formats': list(payment_formats.keys()),
        'amount_stats': {
            'mean': np.mean(amounts),
            'std': np.std(amounts),
            'min': np.min(amounts),
            'max': np.max(amounts)
        }
    }

# Analyser les données existantes
data_stats = analyze_existing_data(template_graphs)
print(f"Analyse terminée: {len(data_stats['bank_ids'])} banques, {len(data_stats['currencies'])} devises")

#Fonction de génération améliorée  
def generate_synthetic_graph(template_graph, data_stats, graph_id):
    """Génère un nouveau graphe synthétique basé sur un graphe existant avec des variations réalistes."""
    new_graph = nx.MultiDiGraph()
    
    original_nodes = list(template_graph.nodes(data=True))
    node_map = {}
    
    for i, (node_id, node_data) in enumerate(original_nodes):
        new_account_id = f"ACC_{graph_id:06d}_{i:03d}_{random.randint(1000, 9999)}"
        node_map[node_id] = new_account_id

        bank_id = random.choice(data_stats['bank_ids']) if data_stats['bank_ids'] else random.randint(100, 1000)
        new_graph.add_node(new_account_id, bank=bank_id, account=new_account_id)

    #Créer de nouvelles arêtes avec des variations 
    for u, v, edge_data in template_graph.edges(data=True):
        new_u = node_map[u]
        new_v = node_map[v]
        
        #Variation temporelle 
        try:
            original_timestamp = datetime.strptime(edge_data['Timestamp'], "%Y/%m/%d %H:%M")
            # Variation de +/- 30 jours maximum
            time_variation = timedelta(
                days=random.randint(-30, 30),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )
            new_timestamp = original_timestamp + time_variation
        except (ValueError, KeyError):
            new_timestamp = datetime.now() - timedelta(days=random.randint(1, 365))
        #Variation des montants  
        original_amount = float(edge_data.get('Amount Received', data_stats['amount_stats']['mean']))
        
        #Appliquer une variation gaussienne contrôlée (max ±50%)
        variation_factor = np.random.normal(1.0, 0.15)  # variation de ±15% en moyenne
        variation_factor = np.clip(variation_factor, 0.5, 2.0)  # limiter à ±50%
        new_amount = max(0.01, original_amount * variation_factor)  # éviter les montants négatifs
        
        from_bank = new_graph.nodes[new_u]['bank']
        to_bank = new_graph.nodes[new_v]['bank']
      
        receiving_currency = random.choice(data_stats['currencies']) if data_stats['currencies'] else 'US Dollar'
        payment_currency = receiving_currency 
        payment_format = random.choice(data_stats['payment_formats']) if data_stats['payment_formats'] else 'Credit Card'
        
        new_edge_data = {
            "Timestamp": new_timestamp.strftime("%Y/%m/%d %H:%M"),
            "From Bank": from_bank,
            "From_Account": new_u,
            "To Bank": to_bank,
            "To_Account": new_v,
            "Amount Received": round(new_amount, 2),
            "Receiving Currency": receiving_currency,
            "Amount Paid": round(new_amount, 2), 
            "Payment Currency": payment_currency,
            "Payment Format": payment_format,
            "Is Laundering": 0,
        }
        
        new_graph.add_edge(new_u, new_v, **new_edge_data)
        
    return {"graph": new_graph, "label": 0}

#Génération  de graphes  
nombre_de_graphes_a_generer = 22000
synthetic_normal_graphs = []

print(f"Génération de {nombre_de_graphes_a_generer} graphes synthétiques...")

#distribution équilibrée des templates
templates_cycle = []
for _ in range(nombre_de_graphes_a_generer):
    if not templates_cycle:
        templates_cycle = template_graphs.copy()
        random.shuffle(templates_cycle)
    
    template = templates_cycle.pop()
    new_graph = generate_synthetic_graph(template, data_stats, len(synthetic_normal_graphs))
    synthetic_normal_graphs.append(new_graph)
    if (len(synthetic_normal_graphs) % 2000) == 0:
        print(f"Progrès: {len(synthetic_normal_graphs)}/{nombre_de_graphes_a_generer}")

print(f"Génération terminée: {len(synthetic_normal_graphs)} graphes créés")

def validate_graphs(graphs, sample_size=100):
    """Valide un échantillon des graphes générés."""
    sample = random.sample(graphs, min(sample_size, len(graphs)))
    
    valid_count = 0
    issues = []
    
    for i, graph_info in enumerate(sample):
        graph = graph_info["graph"]
     
        if len(graph.nodes()) == 0:
            issues.append(f"Graphe {i}: aucun nœud")
            continue
            
        if len(graph.edges()) == 0:
            issues.append(f"Graphe {i}: aucune arête")
            continue
        edge_issues = 0
        for u, v, data in graph.edges(data=True):
            if data.get('Amount Received', 0) <= 0:
                edge_issues += 1
            if 'Timestamp' not in data:
                edge_issues += 1
                
        if edge_issues == 0:
            valid_count += 1
        else:
            issues.append(f"Graphe {i}: {edge_issues} arêtes avec problèmes")
    
    return valid_count, issues

valid_count, issues = validate_graphs(synthetic_normal_graphs)
print(f"\nValidation: {valid_count}/100 graphes valides")
if issues:
    print("Problèmes détectés:")
    for issue in issues[:5]: 
        print(f"  - {issue}")

#Sauvegarde 
print("\nConversion en format JSON...")
all_subgraphs_json = []

for i, graph_info in enumerate(synthetic_normal_graphs):
    try:
        graph_data = nx.node_link_data(graph_info["graph"])
        all_subgraphs_json.append({
            "graph": graph_data, 
            "label": graph_info["label"]
        })
    except Exception as e:
        print(f"Erreur lors de la conversion du graphe {i}: {e}")

BASE_DIR = r"C:\Users\HP\Desktop\Fraud_detection_project"
output_file_path = os.path.join(BASE_DIR, r"data\IBM Transactions\Graph_Trans_normaux_3.json")
try:
    with open(output_file_path, "w") as f:
        json.dump(all_subgraphs_json, f, indent=2)
    print(f"{len(all_subgraphs_json)} graphes sauvegardés dans '{output_file_path}'")
except Exception as e:
    print(f"Erreur lors de la sauvegarde: {e}")
