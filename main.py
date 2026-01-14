#Bloc d'importation
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
#Tri par coût de location croissant
vehicules_sorted = data_vehicles.sort_values(by='rental_cost').to_dict(orient="records")

def get_vehicule(weight, dist):
    #renvoie l'id de la famille de véhicule au coût le moins cher pr un poids et un trajet donné
    best_family = None
    min_total_cost = float('inf')
    #On parcourt tous les véhicules disponibles
    for index, vehicle in data_vehicles.iterrows():
        if vehicle['max_capacity'] >= weight:
            #Trajet aller-retour (dépôt->commande->dépôt) VALABLE SEULEMENT PR HEURISTIQUE DE GROS BEBE
            total_dist = 2*dist
            #Rayon=distance la plus lointaine du dépôt
            radius = dist
            
            #Minimiser coût total
            current_cost = vehicle['rental_cost'] + vehicle['fuel_cost']*total_dist + vehicle['radius_cost']*radius
            if current_cost < min_total_cost:
                min_total_cost = current_cost
                best_family = vehicle['family']
    return best_family


# Chargement instances
instances = []
for k in range(1, 11):
    file_path = f"instance_{k:02d}.csv"
    df = pd.read_csv(file_path)
    instances.append(df.to_dict(orient="records"))


#########################################
#          FONCTIONS GÉOMÉTRIQUES
#########################################
#Coordonnees
def yj_yi(phij, phii): #yj - yi
    return rho * 2 * np.pi * (phij - phii) / 360

def xj_xi(lambdaj, lambdai): #xj - xi
    return rho * math.cos(2 * np.pi * phi_0 / 360) * 2 * np.pi * (lambdaj - lambdai) / 360

def distM(i, j, A): #distance de manhattan entre i et j du fichier A
    deltax = xj_xi(instances[A][j]["longitude"], instances[A][i]["longitude"])
    deltay = yj_yi(instances[A][j]["latitude"], instances[A][i]["latitude"])
    return abs(deltax) + abs(deltay)


#########################################
#          BOUCLE SUR LES INSTANCES
#########################################


for A in range(10):   # A = 0..9
    #liste de routes (pour la solution)
    routes_list = []

    fichier_instance = f"instance_{A+1:02d}.csv"
    fichier_sol = f"solution_{A+1:02d}.csv"

    print(f"Instance {A+1:02d}...")
    df_inst = pd.read_csv(fichier_instance)
    
    #commandes (séparées du dépot)
    index_depot = 0
    orders = df_inst[df_inst['order_weight'].notna()].copy()

    for index, row in orders.iterrows():
        order_id = row['id']
        weight = row['order_weight']
        #distance de manhattan entre dépôt et commande
        dist_depot_commande = distM(index_depot, index, A)
        vehicule_id = get_vehicule(weight, dist_depot_commande)
        routes_list.append([vehicule_id, order_id])

    ##FORMAT SOLUTION
    #Trouver N = nombre max de commandes dans une route
    N=0
    if len(routes_list)>0:
        for r in routes_list:
            if len(r)-1>N:
                N = len(r)-1 #car len(r) contient la colonne family

    #Noms de colonne
    sol_columns = ["family"] + [f"order_{i+1}" for i in range(N)]

    #Définition des lignes
    sol_rows = []
    for r in routes_list:
        row = list(r)
        while len(row)<len(sol_columns):
            row.append("")
        sol_rows.append(row)
    
    #Création du fichier solution
    df_sol = pd.DataFrame(sol_rows, columns = sol_columns)

    #Nomenclature
    df_sol.to_csv(fichier_sol, index=False)
    print(f" {fichier_sol} généré (N={N}, {len(routes_list)} routes).")

    #########################################
