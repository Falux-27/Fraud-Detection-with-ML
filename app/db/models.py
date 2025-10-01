
import pandas as pd
import networkx as nx 
import random 


chemins = [r"data\IBM Transactions\HI-Large_Trans.csv",
           r"data\IBM Transactions\HI-Medium_Trans.csv",
           r"data\IBM Transactions\HI-Small_Trans.csv",
           r"data\IBM Transactions\LI-Large_Trans.csv",
           r"data\IBM Transactions\LI-Medium_Trans.csv",
           r"data\IBM Transactions\LI-Small_Trans.csv"
           ]
all_data = []
for fichier in chemins:
    dataframe = pd.read_csv(fichier, nrows= 1000)
    dataframe = dataframe.rename(columns={
        "Account": "From_Account",  
        "Account.1": "To_Account"
    })
    all_data.append(dataframe)
    
def contruire_graph_global (all_data):
     #l'objet de transformation du file en graphe global
    graph = nx.MultiDiGraph()
    for dtfrm in all_data:
        #Ajout des noeud 
        for index , ligne in dtfrm.iterrows():
            sender = ligne["From_Account"]
            receiver = ligne["To_Account"]
            graph.add_node(
                sender,
                bank= ligne["From Bank"],
                account = ligne["From_Account"]
                )
            graph.add_node(
                receiver,
                bank= ligne["To Bank"],
                account= ligne["To_Account"]
               
                )
        #Ajout des relations (edges)
            timestamp = ligne["Timestamp"]
            amount = ligne["Amount"]
            tx_id = f"{sender}_{receiver}_{timestamp}_{amount}"
            graph.add_edge(sender , receiver , **ligne.to_dict())
        
    graph

def extraire_sous_graph_normaux(graph):
        sous_graph_normaux = []
        normal_tx = []
        for noeud_A , noeud_B , infos_tx in graph.edges(data=True):
            if infos_tx.get('Is Laundering') == 0 :
                normal_tx.append((noeud_A , noeud_B))
        #Choisir noeud de depart
        sender , receiver = random.choice(normal_tx)
        noeud_depart = receiver
        #extraction des des noeuds voisins
        noeud_voisins =nx.bfs_tree(graph,noeud_depart, depth_limit= 3).nodes()
        #Verification si y'a pas de classe 1
        is_positive = False
        for noeud in noeud_voisins:
            for noeud_A, noeud_B, data in graph.edges(noeud, data=True):
                if data.get('Is Laundering') == 1:
                    is_positive = True
                    break
            if is_positive:
                break
        #Ajout des graphes valide 
        if not is_positive and len(noeud_voisins) > 1:
            subgraph = graph.subgraph(noeud_voisins).copy()
            sous_graph_normaux.append({'graph': subgraph, 'label': 0})

        return sous_graph_normaux