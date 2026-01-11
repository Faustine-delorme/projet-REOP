import pandas as pd
import numpy as np
import math
import pulp
import os

##### Constantes #####
rho = 6.371E6
phi_0 = 48.764246


#########################################
#           CHARGEMENT DONNÉES
#########################################

# Chargement véhicules
data_vehicles = pd.read_csv("vehicles.csv")
columns = data_vehicles.columns[1:]

vehicles = []
for _, row in data_vehicles.iterrows():
    vehicles.append({c: row[c] for c in columns})

# Chargement instances
instances = []
for k in range(1, 11):
    file_path = f"instance_{k:02d}.csv"
    df = pd.read_csv(file_path)
    instances.append(df.to_dict(orient="records"))


#########################################
#          FONCTIONS GÉOMÉTRIQUES
#########################################

def yj_yi(phij, phii):
    return rho * 2 * np.pi * (phij - phii) / 360

def xj_xi(lambdaj, lambdai):
    return rho * math.cos(2 * np.pi * phi_0 / 360) * 2 * np.pi * (lambdaj - lambdai) / 360

def distM(i, j, A):
    deltax = xj_xi(instances[A][j]["longitude"], instances[A][i]["longitude"])
    deltay = yj_yi(instances[A][j]["latitude"], instances[A][i]["latitude"])
    return abs(deltax) + abs(deltay)


#########################################
#          BOUCLE SUR LES INSTANCES
#########################################

for A in range(10):   # A = 0..9
    print(f"\n=============================")
    print(f" Résolution instance {A+1:02d}")
    print(f"=============================\n")

    instance = instances[A]

    #########################################
    #           DONNÉES DU MODÈLE
    #########################################

    clients = list(range(len(instance)))   # 0 = dépôt
    familles = [1, 2, 3]                   # véhicules

    w = {i: instance[i]["order_weight"] if not pd.isna(instance[i]["order_weight"]) else 0.0 for i in clients}
    l = {i: instance[i]["delivery_duration"] if not pd.isna(instance[i]["delivery_duration"]) else 0.0 for i in clients}
    tmin = {i: instance[i]["window_start"] if not pd.isna(instance[i]["window_start"]) else 0.0 for i in clients}
    tmax = {i: instance[i]["window_end"] if not pd.isna(instance[i]["window_end"]) else 86400.0 for i in clients}

    # Pré-calcul gamma_max (approx linéaire)
    gamma_max = []
    for v in vehicles:
        g = 0
        for n in range(4):
            g += abs(v[f"fourier_cos_{n}"]) + abs(v[f"fourier_sin_{n}"])
        gamma_max.append(g)

    # Tous arcs (i ≠ j)
    arcs = [(i, j, f) for i in clients for j in clients if i != j for f in familles]


    #########################################
    #         CONSTRUCTION DU MILP
    #########################################

    model = pulp.LpProblem(f"Califrais_{A+1:02d}", pulp.LpMinimize)

    # VARIABLES
    x = pulp.LpVariable.dicts("x", arcs, 0, 1, cat="Binary")
    tvar = pulp.LpVariable.dicts("t", clients, lowBound=0)

    load = pulp.LpVariable.dicts(
        "load",
        [(i, f) for i in clients for f in familles],
        lowBound=0
    )

    # Temps au dépôt
    model += tvar[0] == 0, "time_at_depot"

    # Charge au dépôt
    for f in familles:
        model += load[(0, f)] == 0, f"load_depot_f{f}"

    # Charge <= capacité
    for i in clients:
        for f in familles:
            model += load[(i, f)] <= vehicles[f-1]["max_capacity"]


    #########################################
    #            CONTRAINTES
    #########################################

    # 1) Chaque client visité exactement une fois
    for j in clients:
        if j != 0:
            model += (
                pulp.lpSum(x[i, j, f] for i in clients if i != j for f in familles) == 1,
                f"visit_once_{j}"
            )

    # 2) Flux entrant = sortant pour chaque famille
    for j in clients:
        if j != 0:
            for f in familles:
                model += (
                    pulp.lpSum(x[i, j, f] for i in clients if i != j)
                    ==
                    pulp.lpSum(x[j, k, f] for k in clients if k != j),
                    f"flow_{j}_{f}"
                )

    # 3) Dépôt départ = retour
    for f in familles:
        model += (
            pulp.lpSum(x[0, j, f] for j in clients if j != 0)
            ==
            pulp.lpSum(x[i, 0, f] for i in clients if i != 0),
            f"depot_balance_{f}"
        )

    # 4) Charge MTZ
    for i in clients:
        for j in clients:
            if i != j:
                for f in familles:
                    Qf = vehicles[f-1]["max_capacity"]
                    model += (
                        load[(j, f)] >= load[(i, f)] + w[j] - Qf * (1 - x[i, j, f]),
                        f"cap_{i}_{j}_{f}"
                    )

    # 5) Fenêtres de temps
    for i in clients:
        if i != 0:
            model += tvar[i] >= tmin[i]
            model += tvar[i] <= tmax[i]

    # 6) Séquencement temporel
    M = 10**7

    for i in clients:
        for j in clients:
            if i != j:
                for f in familles:
                    v = vehicles[f-1]
                    base = distM(i, j, A) / v["speed"] + v["parking_time"]
                    tau_ij = base * gamma_max[f-1]

                    model += (
                        tvar[j] >= tvar[i] + l[i] + tau_ij - M * (1 - x[i, j, f]),
                        f"time_{i}_{j}_{f}"
                    )

    #########################################
    #           OBJECTIF LINÉAIRE
    #########################################

    fuel_term = pulp.lpSum(
        vehicles[f-1]["fuel_cost"] * distM(i, j, A) * x[i, j, f]
        for (i, j, f) in arcs
    )

    rental_term = pulp.lpSum(
        vehicles[f-1]["rental_cost"] * pulp.lpSum(x[0, j, f] for j in clients if j != 0)
        for f in familles
    )

    model += fuel_term + rental_term


    #########################################
    #              RÉSOLUTION
    #########################################

    status = model.solve(pulp.PULP_CBC_CMD(msg=0))
    print("Status :", pulp.LpStatus[status])


    #########################################
    #      RECONSTRUCTION DES ROUTES
    #########################################

    EPS = 1e-5
    successors = {(i, f): [] for i in clients for f in familles}

    for (i, j, f) in arcs:
        if pulp.value(x[i, j, f]) and pulp.value(x[i, j, f]) > EPS:
            successors[(i, f)].append(j)

    routes = []

    for f in familles:
        starts = successors[(0, f)]
        for start in starts:
            route = []
            current = start
            visited = set()
            while current != 0 and current not in visited:
                visited.add(current)
                route.append(current)

                nxt = successors[(current, f)]
                if len(nxt) == 0:
                    break
                current = nxt[0]

            if route:
                routes.append({"family": f, "nodes": route})

    # Conversion en IDs
    for r in routes:
        r["order_ids"] = [instance[i]["id"] for i in r["nodes"]]


    #########################################
    #      CRÉATION DES DOSSIERS
    #########################################

    inst_folder = f"solutions/instance_{A+1:02d}"
    os.makedirs(inst_folder, exist_ok=True)


    #########################################
    #         EXPORT ROUTES.CSV
    #########################################

    if routes:
        max_len = max(len(r["order_ids"]) for r in routes)
    else:
        max_len = 0

    cols = ["family"] + [f"order_{k+1}" for k in range(max_len)]
    rows = []

    for r in routes:
        row = [r["family"]] + r["order_ids"]
        while len(row) < len(cols):
            row.append("")
        rows.append(row)

    df_routes = pd.DataFrame(rows, columns=cols)
    df_routes.to_csv(f"{inst_folder}/routes.csv", index=False)

    print(f" → Fichier créé : {inst_folder}/routes.csv")
