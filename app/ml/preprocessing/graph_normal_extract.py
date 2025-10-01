import pandas as pd
import networkx as nx 
import random 
import pickle
import json

chemins = [
    r"data\IBM Transactions\HI-Large_Trans.csv",
    r"data\IBM Transactions\HI-Medium_Trans.csv", 
    r"data\IBM Transactions\HI-Small_Trans.csv",
    r"data\IBM Transactions\LI-Large_Trans.csv",
    r"data\IBM Transactions\LI-Medium_Trans.csv",
    r"data\IBM Transactions\LI-Small_Trans.csv"
]
all_data = []
for fichier in chemins:
    dataframe = pd.read_csv(fichier, nrows=1000)
    dataframe = dataframe.rename(columns={
        "Account": "From_Account", 
        "Account.1": "To_Account",
        "Is Laundering?": "Is Laundering"
    })
    all_data.append(dataframe)

#Construction du graphe
graph_global = nx.MultiDiGraph()
for dtfrm in all_data:
    for _, ligne in dtfrm.iterrows():
        sender = ligne["From_Account"]
        receiver = ligne["To_Account"]
        
        #Ajout des noeuds
        graph_global.add_node(sender, bank=ligne["From Bank"], account=ligne["From_Account"])
        graph_global.add_node(receiver, bank=ligne["To Bank"], account=ligne["To_Account"])

        #Ajout de l'arête avec un id unique
        graph_global.add_edge(
            sender, 
            receiver, 
            key=ligne.get("Transaction_ID", f"{sender}_{receiver}_{ligne['Timestamp']}_{ligne['Amount Received']}"), 
            **ligne.to_dict()
        )

sous_graph_normaux = []
extracted_subgraphs_nodes = set()
hop_limit = 3
max_to_extract = 30000
max_tentatives = 20000

#Tx normaux
normal_tx = [(u, v) for u, v, data in graph_global.edges(data=True) if data.get("Is Laundering") == 0]

if not normal_tx:
    print("Aucune transaction normale trouvée.")
else:
    #Boucle pour un nombre limité de tentatives
    for _ in range(max_tentatives):
        #Échantillonnage aléatoire
        sender, receiver = random.choice(normal_tx)
        noeud_depart = receiver
        
        try: 
            noeud_voisins = list(nx.bfs_tree(graph_global, noeud_depart, depth_limit=hop_limit).nodes())
        except nx.NetworkXError:
            continue    
        #Vérifier si un sous-graphe a déjà été extrait pour éviter les doublons
        # if frozenset(noeud_voisins) in extracted_subgraphs_nodes:
        #     continue
        subgraph = graph_global.subgraph(noeud_voisins)
        if any(data.get("Is Laundering") == 1 for _, _, data in subgraph.edges(data=True)):
            continue  
        #Ajouter le sous-graphe  
        if len(noeud_voisins) > 1:
            sous_graph_normaux.append({"graph": subgraph.copy(), "label": 0})
            #extracted_subgraphs_nodes.add(frozenset(noeud_voisins))
        if len(sous_graph_normaux) >= max_to_extract:
            break

print(f"Nombre de sous-graphes normaux extraits : {len(sous_graph_normaux)}")


# Afficher les informations du premier sous-graphe
all_subgraphs_json = []

# Boucler sur tous les sous-graphes normaux
for graph_info in sous_graph_normaux:
    # Convertir chaque objet NetworkX en dictionnaire
    graph_data = nx.node_link_data(graph_info["graph"])
    
    # Ajouter le dictionnaire avec le label à la liste
    all_subgraphs_json.append({
        "graph": graph_data, 
        "label": graph_info["label"]
    })

#Convertir la liste complète en chaîne JSON et l'enregistrer dans un fichier
with open(r"C:\Users\HP\Desktop\Fraud_detection_project\data\IBM Transactions\Graph_Trans_normaux2.json", "w") as f:
    json.dump(all_subgraphs_json, f, indent=4)
    
 