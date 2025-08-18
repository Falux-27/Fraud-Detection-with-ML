import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from collections import defaultdict

def random_datetime_since_2023(hour=None):
    """Retourne une datetime aléatoire entre le 1er janvier 2023 et maintenant."""
    start_date = datetime(2023, 1, 1)
    end_date = datetime.now()
    total_seconds = int((end_date - start_date).total_seconds())
    random_seconds = random.randint(0, total_seconds)
    dt = start_date + timedelta(seconds=random_seconds)
    if hour is not None:
        dt = dt.replace(hour=int(hour), minute=random.randint(0, 59), second=random.randint(0, 59))
    return dt

nb_total = 284887
nb_fraude = 492
nb_non_fraude = nb_total - nb_fraude

# Configuration des paramètres
currencies = ['XOF', 'USD', 'EUR']
transaction_codes = ['RCDT001', 'RCDT002', 'RCDT003']
modes = ['mobile', 'cb', 'virement', 'agence']
natures = ['achat', 'retrait', 'virement']
status_options = ['success', 'failed', 'waiting']
account_status = ['actif', 'inactif', 'fermé']
account_types = ['courant', 'epargne']

# Structures de données pour simuler l'historique des comptes
account_balances = defaultdict(lambda: random.uniform(10000, 500000))
account_histories = defaultdict(list)
account_metadata = {}

# PATTERNS DE FRAUDE ALLÉGÉS - Plus détectables
FRAUD_PATTERNS = {
    'stealth': 0.60,      # 60% - Toujours camouflées mais avec quelques indices
    'opportunist': 0.30,  # 30% - Plus visibles, exploitent failles
    'aggressive': 0.10    # 10% - Très visibles, remplace "random"
}

def get_normal_transaction_stats(transactions_sample):
    """Analyse les patterns normaux"""
    if not transactions_sample:
        return {
            'amount_mean': 50000,
            'amount_std': 20000,
            'amount_median': 45000,
            'common_hours': list(range(8, 20)),
            'weekend_ratio': 0.3,
            'mobile_ratio': 0.6,
            'success_ratio': 0.95
        }
    
    amounts = [tx['amount'] for tx in transactions_sample]
    return {
        'amount_mean': np.mean(amounts),
        'amount_std': np.std(amounts),
        'amount_median': np.median(amounts),
        'common_hours': list(range(8, 20)),
        'weekend_ratio': 0.3,
        'mobile_ratio': 0.6,
        'success_ratio': 0.95
    }

def init_account(account_id):
    """Initialise les métadonnées d'un compte"""
    if account_id not in account_metadata:
        account_metadata[account_id] = {
            'statut_compte': np.random.choice(account_status, p=[0.85, 0.13, 0.02]),
            'type_compte': np.random.choice(account_types, p=[0.7, 0.3]),
            'creation_date': datetime.now() - timedelta(days=random.randint(30, 1095)),
            'last_activity': None,
            'fraud_history': 0  # Nouveau : compteur de fraudes
        }

def calculate_account_features(account_id, current_amount, current_time):
    """Calcule les features enrichies pour un compte"""
    init_account(account_id)
    
    sender_balance_before = account_balances[account_id]
    sender_balance_after = max(0, sender_balance_before - current_amount)
    account_balances[account_id] = sender_balance_after
    
    history = account_histories[account_id]
    sender_last_tx_date = history[-1]['timestamp'] if history else account_metadata[account_id]['creation_date']
    
    # Calculs enrichis
    recent_txs = [tx for tx in history if (current_time - tx['timestamp']).days <= 30]
    sender_avg_tx_amount_30d = np.mean([tx['amount'] for tx in recent_txs]) if recent_txs else current_amount
    
    # Nouvelles features détectables
    tx_count_24h = len([tx for tx in history if (current_time - tx['timestamp']).total_seconds() <= 86400])
    days_since_last_tx = (current_time - sender_last_tx_date).days if sender_last_tx_date else 0
    
    account_histories[account_id].append({
        'timestamp': current_time,
        'amount': current_amount
    })
    
    account_metadata[account_id]['last_activity'] = current_time
    
    return {
        'sender_balance_before': round(sender_balance_before, 2),
        'sender_balance_after': round(sender_balance_after, 2),
        'sender_last_tx_date': sender_last_tx_date,
        'sender_avg_tx_amount_30d': round(sender_avg_tx_amount_30d, 2),
        'statut_compte': account_metadata[account_id]['statut_compte'],
        'type_compte': account_metadata[account_id]['type_compte'],
        'tx_count_24h': tx_count_24h,
        'days_since_last_tx': days_since_last_tx,
        'balance_ratio': round(current_amount / sender_balance_before if sender_balance_before > 0 else 0, 3)
    }

def generate_stealth_fraud(id_sender_pool, normal_stats):
    """Fraude camouflée - ALLÉGÉE avec indices détectables"""
    # Montant légèrement au-dessus de la normale (indice subtil)
    amount_factor = random.uniform(1.1, 1.8)  # Au lieu de copier exactement
    base_amount = np.random.normal(normal_stats['amount_mean'], normal_stats['amount_std'])
    amount = max(1000, base_amount * amount_factor)
    
    # 70% horaires normaux, 30% horaires suspects (indice)
    if random.random() < 0.7:
        fraud_time = random_datetime_since_2023(hour=random.choice(normal_stats['common_hours']))
    else:
        fraud_time = random_datetime_since_2023(hour=random.choice([0, 1, 2, 3, 22, 23]))
    
    sender_id = random.choice(id_sender_pool)
    
    # Marquer ce compte comme ayant une fraude (pattern détectable)
    init_account(sender_id)
    account_metadata[sender_id]['fraud_history'] += 1
    
    account_features = calculate_account_features(sender_id, amount, fraud_time)
    
    tx = {
        "id_tx": f"TX{random.randint(100000, 999999)}",
        "timestamp": fraud_time,
        "id_sender": sender_id,
        "id_receiver": f"RX{random.randint(100000, 999999)}",
        "amount": round(amount, 2),
        "currency": 'XOF',  # Toujours XOF pour stealth
        "transaction_code": 'RCDT001',
        "mode": 'mobile',
        "natur_tx": 'achat',
        "status": 'success',
        "date_created": datetime.now()
    }
    
    tx.update(account_features)
    return tx

def generate_opportunist_fraud(id_sender_pool, normal_stats):
    """Fraude opportuniste - Plus visible, exploite failles"""
    # Montants nettement plus élevés
    amount_multiplier = random.uniform(2.0, 5.0)
    base_amount = np.random.normal(normal_stats['amount_mean'], normal_stats['amount_std'])
    amount = max(1000, base_amount * amount_multiplier)
    
    # Horaires suspects plus fréquents
    if random.random() < 0.6:  # 60% horaires suspects
        fraud_time = random_datetime_since_2023(hour=random.choice([0, 1, 2, 3, 4, 22, 23]))
    else:
        fraud_time = random_datetime_since_2023(hour=random.choice(range(24)))
    
    sender_id = random.choice(id_sender_pool)
    init_account(sender_id)
    account_metadata[sender_id]['fraud_history'] += 1
    
    # Comportement plus visible : vider le compte ou montant > solde
    if random.random() < 0.4:
        account_balances[sender_id] = amount * random.uniform(0.8, 1.1)
    
    account_features = calculate_account_features(sender_id, amount, fraud_time)
    
    tx = {
        "id_tx": f"TX{random.randint(100000, 999999)}",
        "timestamp": fraud_time,
        "id_sender": sender_id,
        "id_receiver": f"RX{random.randint(100000, 999999)}",
        "amount": round(amount, 2),
        "currency": random.choice(['USD', 'EUR']),  # Devises moins communes
        "transaction_code": random.choice(['RCDT002', 'RCDT003']),
        "mode": random.choice(['cb', 'virement']),  # Modes moins sécurisés
        "natur_tx": 'retrait',  # Principalement des retraits
        "status": 'success',
        "date_created": datetime.now()
    }
    
    tx.update(account_features)
    return tx

def generate_aggressive_fraud(id_sender_pool):
    """Fraude agressive - Très visible et détectable"""
    # Montants très élevés
    amount = round(random.uniform(100000, 500000), 2)
    
    # Horaires très suspects
    fraud_time = random_datetime_since_2023(hour=random.choice([1, 2, 3, 4]))
    
    sender_id = random.choice(id_sender_pool)
    init_account(sender_id)
    account_metadata[sender_id]['fraud_history'] += 1
    
    # Comportements très suspects
    if random.random() < 0.6:
        account_metadata[sender_id]['statut_compte'] = random.choice(['inactif', 'fermé'])
    
    # Vider complètement le compte
    account_balances[sender_id] = amount * random.uniform(0.9, 1.1)
    
    account_features = calculate_account_features(sender_id, amount, fraud_time)
    
    tx = {
        "id_tx": f"TX{random.randint(100000, 999999)}",
        "timestamp": fraud_time,
        "id_sender": sender_id,
        "id_receiver": f"RX{random.randint(100000, 999999)}" if random.random() > 0.3 else None,
        "amount": amount,
        "currency": random.choice(['USD', 'EUR']),
        "transaction_code": random.choice(transaction_codes),
        "mode": random.choice(['cb', 'virement']) if random.random() > 0.2 else None,
        "natur_tx": 'retrait',
        "status": random.choice(['success', 'failed']),
        "date_created": datetime.now()
    }
    
    tx.update(account_features)
    return tx

def generate_normal_tx(id_sender_pool):
    """Génère une transaction normale réaliste"""
    # Distribution réaliste des montants
    amount = max(500, np.random.lognormal(mean=10.5, sigma=1.0))
    
    # Horaires réalistes
    hour_weights = np.array([0.01]*6 + [0.05]*2 + [0.15]*10 + [0.08]*4 + [0.02]*2)
    hour_weights = hour_weights / hour_weights.sum()
    hour = np.random.choice(range(24), p=hour_weights)
    
    tx_time = random_datetime_since_2023(hour)
    
    sender_id = random.choice(id_sender_pool)
    account_features = calculate_account_features(sender_id, amount, tx_time)
    
    tx = {
        "id_tx": f"TX{random.randint(100000, 999999)}",
        "timestamp": tx_time,
        "id_sender": sender_id,
        "id_receiver": f"RX{random.randint(100000, 999999)}" if random.random() > 0.05 else None,
        "amount": round(amount, 2),
        "currency": np.random.choice(currencies, p=[0.7, 0.2, 0.1]),
        "transaction_code": np.random.choice(transaction_codes, p=[0.6, 0.3, 0.1]),
        "mode": np.random.choice(modes, p=[0.6, 0.25, 0.1, 0.05]) if random.random() > 0.1 else None,
        "natur_tx": np.random.choice(natures, p=[0.5, 0.3, 0.2]) if random.random() > 0.1 else None,
        "status": np.random.choice(status_options, p=[0.95, 0.04, 0.01]),
        "date_created": datetime.now()
    }
    
    tx.update(account_features)
    return tx

# Génération des données
print("Génération des données en cours...")

# Pool d'ID sender
id_sender_pool = [f"USR{random.randint(10000, 99999)}" for _ in range(2000)]

# Transactions non frauduleuses
print("Génération des transactions normales...")
tx_non_fraude = [generate_normal_tx(id_sender_pool) for _ in range(nb_non_fraude)]

# Analyse des patterns normaux
normal_stats = get_normal_transaction_stats(tx_non_fraude[:1000])

# Transactions frauduleuses avec nouvelle distribution
tx_fraude = []

nb_stealth = int(nb_fraude * FRAUD_PATTERNS['stealth'])
nb_opportunist = int(nb_fraude * FRAUD_PATTERNS['opportunist'])
nb_aggressive = nb_fraude - nb_stealth - nb_opportunist

print(f"Génération des fraudes:")
print(f"- Stealth: {nb_stealth}")
print(f"- Opportunistes: {nb_opportunist}")
print(f"- Agressives: {nb_aggressive}")

# Génération des fraudes
for _ in range(nb_stealth):
    tx_fraude.append(generate_stealth_fraud(id_sender_pool, normal_stats))

for _ in range(nb_opportunist):
    tx_fraude.append(generate_opportunist_fraud(id_sender_pool, normal_stats))

for _ in range(nb_aggressive):
    tx_fraude.append(generate_aggressive_fraud(id_sender_pool))

# Ajouter labels
for tx in tx_non_fraude:
    tx['label'] = 0
for tx in tx_fraude:
    tx['label'] = 1

# Fusionner et mélanger
all_tx = tx_non_fraude + tx_fraude
random.shuffle(all_tx)

# Convertir en DataFrame
df_simulated = pd.DataFrame(all_tx)

 