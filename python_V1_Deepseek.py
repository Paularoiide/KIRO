import pandas as pd
import numpy as np
from math import radians, cos, sin, sqrt, pi
import warnings
import os
import time
warnings.filterwarnings('ignore')

# Constantes
EARTH_RADIUS = 6371000  # Rayon de la Terre en mètres
DEPOT_LATITUDE = 48.764246

class Vehicle:
    def __init__(self, family, max_capacity, rental_cost, fuel_cost, radius_cost, 
                 speed, parking_time, fourier_cos, fourier_sin):
        self.family = int(family)
        self.max_capacity = max_capacity
        self.rental_cost = rental_cost
        self.fuel_cost = fuel_cost
        self.radius_cost = radius_cost
        self.speed = speed
        self.parking_time = parking_time
        self.fourier_cos = fourier_cos
        self.fourier_sin = fourier_sin

class Order:
    def __init__(self, id, latitude, longitude, weight, window_start, window_end, delivery_duration):
        self.id = int(id)
        self.latitude = latitude
        self.longitude = longitude
        self.weight = weight
        self.window_start = window_start
        self.window_end = window_end
        self.delivery_duration = delivery_duration

class Route:
    def __init__(self, family, orders):
        self.family = int(family)
        self.orders = [int(order) for order in orders]
        self.capacity_used = 0.0

class CalifraisSolver:
    def __init__(self, vehicles_file, instance_file):
        self.vehicles = self.load_vehicles(vehicles_file)
        self.orders, self.depot = self.load_orders(instance_file)
        self.positions = self.calculate_positions()
        self.distance_matrix = self.precompute_distances()
    
    def load_vehicles(self, vehicles_file):
        """Charge les données des véhicules"""
        df = pd.read_csv(vehicles_file)
        vehicles = []
        
        for _, row in df.iterrows():
            vehicle = Vehicle(
                family=row['family'],
                max_capacity=row['max_capacity'],
                rental_cost=row['rental_cost'],
                fuel_cost=row['fuel_cost'],
                radius_cost=row['radius_cost'],
                speed=row['speed'],
                parking_time=row['parking_time'],
                fourier_cos=[row['fourier_cos_0'], row['fourier_cos_1'], 
                           row['fourier_cos_2'], row['fourier_cos_3']],
                fourier_sin=[row['fourier_sin_0'], row['fourier_sin_1'], 
                           row['fourier_sin_2'], row['fourier_sin_3']]
            )
            vehicles.append(vehicle)
        
        return vehicles
    
    def load_orders(self, instance_file):
        """Charge les données des commandes"""
        df = pd.read_csv(instance_file)
        orders = []
        depot = None
        
        for _, row in df.iterrows():
            if row['id'] == 0:
                depot = Order(0, row['latitude'], row['longitude'], 0.0, 0.0, float('inf'), 0.0)
            else:
                # Gérer les valeurs manquantes
                weight = row['order_weight'] if pd.notna(row['order_weight']) else 0.0
                window_start = row['window_start'] if pd.notna(row['window_start']) else 0.0
                window_end = row['window_end'] if pd.notna(row['window_end']) else float('inf')
                delivery_duration = row['delivery_duration'] if pd.notna(row['delivery_duration']) else 0.0
                
                order = Order(
                    id=row['id'],
                    latitude=row['latitude'],
                    longitude=row['longitude'],
                    weight=weight,
                    window_start=window_start,
                    window_end=window_end,
                    delivery_duration=delivery_duration
                )
                orders.append(order)
        
        return orders, depot
    
    def geographic_to_cartesian(self, lat, lon):
        """Convertit les coordonnées géographiques en coordonnées cartésiennes"""
        lat_rad = radians(lat)
        lon_rad = radians(lon)
        depo_lat_rad = radians(DEPOT_LATITUDE)
        
        y = EARTH_RADIUS * (lat_rad - depo_lat_rad)
        x = EARTH_RADIUS * cos(depo_lat_rad) * (lon_rad - radians(DEPOT_LATITUDE))
        
        return x, y
    
    def calculate_positions(self):
        """Calcule les positions cartésiennes de tous les points"""
        positions = {}
        
        # Position du dépôt
        positions[0] = self.geographic_to_cartesian(self.depot.latitude, self.depot.longitude)
        
        # Positions des commandes
        for order in self.orders:
            positions[order.id] = self.geographic_to_cartesian(order.latitude, order.longitude)
        
        return positions
    
    def manhattan_distance(self, pos1, pos2):
        """Calcule la distance de Manhattan entre deux points"""
        return abs(pos2[0] - pos1[0]) + abs(pos2[1] - pos1[1])
    
    def euclidean_distance(self, pos1, pos2):
        """Calcule la distance Euclidienne entre deux points"""
        return sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
    
    def precompute_distances(self):
        """Pré-calcule la matrice des distances Manhattan"""
        print("Pré-calcul des distances...")
        distance_matrix = {}
        all_nodes = [0] + [order.id for order in self.orders]
        
        for i in all_nodes:
            for j in all_nodes:
                if i != j:
                    distance_matrix[(i, j)] = self.manhattan_distance(
                        self.positions[i], self.positions[j]
                    )
        
        print("Pré-calcul des distances terminé")
        return distance_matrix
    
    def travel_time(self, vehicle, dist_manhattan, departure_time):
        """Calcule le temps de trajet avec coefficients de Fourier"""
        base_time = dist_manhattan / vehicle.speed + vehicle.parking_time
        
        T = 86400.0  # 1 jour en secondes
        ω = 2 * pi / T
        
        # Calcul du facteur temporel γ
        γ = vehicle.fourier_cos[0]  # Terme constant (fourier_cos_0)
        
        for n in range(1, 4):
            γ += vehicle.fourier_cos[n] * cos(n * ω * departure_time)
            γ += vehicle.fourier_sin[n] * sin(n * ω * departure_time)
        
        return base_time * max(γ, 0.1)  # Éviter les valeurs négatives
    
    def get_order_by_id(self, order_id):
        """Retourne un objet Order par son ID"""
        order_id = int(order_id)
        if order_id == 0:
            return self.depot
        for order in self.orders:
            if order.id == order_id:
                return order
        return None
    
    def find_best_vehicle(self, cluster_weight):
        """Trouve le véhicule optimal pour un poids donné"""
        for vehicle in self.vehicles:
            if cluster_weight <= vehicle.max_capacity:
                return vehicle
        return self.vehicles[-1]
    
    def calculate_route_cost(self, route, vehicle):
        """Calcule le coût total d'une route"""
        # Coût de location
        rental_cost = vehicle.rental_cost
        
        # Coût carburant (distance Manhattan)
        fuel_cost = 0.0
        nodes = [0] + route.orders + [0]  # Dépôt -> commandes -> dépôt
        
        for i in range(len(nodes) - 1):
            fuel_cost += self.distance_matrix[(nodes[i], nodes[i+1])] * vehicle.fuel_cost
        
        # Coût de rayon (pénalité de dispersion)
        if len(route.orders) > 1:
            positions = [self.positions[id] for id in route.orders]
            max_distance = 0.0
            
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    dist = self.euclidean_distance(positions[i], positions[j])
                    max_distance = max(max_distance, dist)
            
            radius_cost = vehicle.radius_cost * (max_distance / 2) ** 2
        else:
            radius_cost = 0.0
        
        return rental_cost + fuel_cost + radius_cost
    
    def solve_with_time_windows(self):
        """Nouvelle méthode qui respecte strictement les fenêtres de temps"""
        print("=== Début de la résolution avec contraintes temporelles strictes ===")
        print(f"Nombre de commandes: {len(self.orders)}")
        
        # Trier les commandes par fenêtre de fin (plus restrictive en premier)
        sorted_orders = sorted(self.orders, key=lambda o: o.window_end)
        
        routes = []
        
        for order in sorted_orders:
            best_route = None
            best_position = -1
            best_arrival_time = float('inf')
            
            # Essayer d'insérer dans les routes existantes
            for route in routes:
                vehicle = self.vehicles[route.family - 1]
                
                # Vérifier la capacité
                if route.capacity_used + order.weight > vehicle.max_capacity:
                    continue
                
                # Essayer toutes les positions d'insertion
                for pos in range(len(route.orders) + 1):
                    # Créer une nouvelle séquence test
                    test_orders = route.orders[:pos] + [order.id] + route.orders[pos:]
                    
                    # Vérifier la faisabilité temporelle
                    if self.is_sequence_feasible(vehicle, test_orders):
                        # Calculer le temps d'arrivée pour cette commande
                        arrival_time = self.calculate_arrival_time_at_order(vehicle, test_orders, pos)
                        
                        if arrival_time < best_arrival_time:
                            best_route = route
                            best_position = pos
                            best_arrival_time = arrival_time
            
            # Si une insertion valide est trouvée
            if best_route is not None:
                best_route.orders.insert(best_position, order.id)
                best_route.capacity_used += order.weight
            else:
                # Créer une nouvelle route
                vehicle = self.find_best_vehicle(order.weight)
                
                # Vérifier que la commande peut être livrée seule
                travel_time = self.travel_time(vehicle, self.distance_matrix[(0, order.id)], 0)
                arrival_time = travel_time
                if arrival_time < order.window_start:
                    arrival_time = order.window_start
                
                if arrival_time <= order.window_end:
                    new_route = Route(vehicle.family, [order.id])
                    new_route.capacity_used = order.weight
                    routes.append(new_route)
                else:
                    print(f"⚠️  Commande {order.id} ne peut pas être livrée à temps même seule")
                    # Forcer la création quand même (solution de dernier recours)
                    new_route = Route(vehicle.family, [order.id])
                    new_route.capacity_used = order.weight
                    routes.append(new_route)
        
        # Phase d'optimisation : supprimer les retards en divisant les routes
        routes = self.ensure_time_feasibility(routes)
        
        print(f"Nombre de routes créées: {len(routes)}")
        return routes
    
    def is_sequence_feasible(self, vehicle, orders):
        """Vérifie si une séquence de commandes respecte toutes les contraintes temporelles"""
        current_time = 0.0  # Départ du dépôt
        prev_node = 0
        
        for i, order_id in enumerate(orders):
            order = self.get_order_by_id(order_id)
            
            # Temps de trajet
            travel_time = self.travel_time(vehicle, self.distance_matrix[(prev_node, order_id)], current_time)
            arrival_time = current_time + travel_time
            
            # Ajuster si arrivée avant l'ouverture
            if arrival_time < order.window_start:
                arrival_time = order.window_start
            
            # Vérifier si on arrive après la fermeture
            if arrival_time > order.window_end:
                return False
            
            # Temps de départ
            current_time = arrival_time + order.delivery_duration
            prev_node = order_id
        
        return True
    
    def calculate_arrival_time_at_order(self, vehicle, orders, target_index):
        """Calcule le temps d'arrivée à une commande spécifique dans une séquence"""
        current_time = 0.0  # Départ du dépôt
        prev_node = 0
        
        for i, order_id in enumerate(orders):
            order = self.get_order_by_id(order_id)
            
            # Temps de trajet
            travel_time = self.travel_time(vehicle, self.distance_matrix[(prev_node, order_id)], current_time)
            arrival_time = current_time + travel_time
            
            # Ajuster si arrivée avant l'ouverture
            if arrival_time < order.window_start:
                arrival_time = order.window_start
            
            # Si c'est la commande cible, retourner le temps d'arrivée
            if i == target_index:
                return arrival_time
            
            # Temps de départ
            current_time = arrival_time + order.delivery_duration
            prev_node = order_id
        
        return float('inf')
    
    def ensure_time_feasibility(self, routes):
        """Garantit que toutes les routes respectent les contraintes temporelles"""
        feasible_routes = []
        
        for route in routes:
            vehicle = self.vehicles[route.family - 1]
            
            if self.is_sequence_feasible(vehicle, route.orders):
                feasible_routes.append(route)
            else:
                # Diviser la route en plusieurs routes plus petites
                split_routes = self.split_route_to_respect_times(route, vehicle)
                feasible_routes.extend(split_routes)
        
        return feasible_routes
    
    def split_route_to_respect_times(self, route, vehicle):
        """Divise une route pour respecter les contraintes temporelles"""
        if not route.orders:
            return []
        
        sub_routes = []
        current_sub_route = []
        current_weight = 0.0
        
        for order_id in route.orders:
            order = self.get_order_by_id(order_id)
            
            # Vérifier si l'ajout respecte les contraintes
            test_route = current_sub_route + [order_id]
            if (current_weight + order.weight <= vehicle.max_capacity and 
                self.is_sequence_feasible(vehicle, test_route)):
                
                current_sub_route.append(order_id)
                current_weight += order.weight
            else:
                # Créer une nouvelle sous-route
                if current_sub_route:
                    new_route = Route(vehicle.family, current_sub_route)
                    new_route.capacity_used = current_weight
                    sub_routes.append(new_route)
                
                current_sub_route = [order_id]
                current_weight = order.weight
        
        # Ajouter la dernière sous-route
        if current_sub_route:
            new_route = Route(vehicle.family, current_sub_route)
            new_route.capacity_used = current_weight
            sub_routes.append(new_route)
        
        return sub_routes
    
    def solve(self):
        """Méthode principale de résolution"""
        start_time = time.time()
        
        # Utiliser l'algorithme avec contraintes temporelles strictes
        routes = self.solve_with_time_windows()
        
        # Calcul du coût total
        total_cost = 0.0
        for route in routes:
            family_int = int(route.family)
            vehicle = self.vehicles[family_int - 1]
            total_cost += self.calculate_route_cost(route, vehicle)
        
        end_time = time.time()
        
        print(f"\n=== Résolution terminée ===")
        print(f"Temps d'exécution: {end_time - start_time:.2f} secondes")
        print(f"Nombre de routes créées: {len(routes)}")
        print(f"Coût total estimé: {total_cost:.2f} €")
        print(f"Commandes livrées: {sum(len(route.orders) for route in routes)}/{len(self.orders)}")
        
        return routes
    
    def write_solution(self, routes, output_file="routes.csv"):
        """Écrit la solution dans un fichier CSV avec des entiers uniquement"""
        if not routes:
            print("Aucune route à sauvegarder!")
            return
        
        # Trouver le nombre maximum de commandes dans une route
        max_orders = max(len(route.orders) for route in routes)
        
        # Préparer les données - FORCER les types int
        data = []
        
        for route in routes:
            # Conversion EXPLICITE en int pour éviter les .0
            row = [int(route.family)]
            for order in route.orders:
                row.append(int(order))
            
            # Remplir avec des chaînes vides (pas None) pour les colonnes manquantes
            while len(row) < max_orders + 1:
                row.append("")  # Chaîne vide au lieu de None
            
            data.append(row)
        
        # Créer le DataFrame avec gestion stricte des types
        columns = ['family'] + [f'order_{i+1}' for i in range(max_orders)]
        
        # Créer un DataFrame avec conversion forcée en int où c'est possible
        df = pd.DataFrame(data, columns=columns)
        
        # Forcer les colonnes order_* à être des entiers ou des chaînes vides
        for col in df.columns:
            if col.startswith('order_'):
                # Remplacer les valeurs vides par des chaînes vides
                df[col] = df[col].apply(lambda x: '' if pd.isna(x) or x == '' else int(x))
        
        # Forcer la colonne family à être int
        df['family'] = df['family'].astype(int)
        
        # Sauvegarder sans index et en s'assurant qu'aucune valeur décimale n'est écrite
        df.to_csv(output_file, index=False, na_rep='')
        print(f"Solution sauvegardée dans: {output_file}")
    
    def print_statistics(self, routes):
        """Affiche les statistiques de la solution"""
        if not routes:
            print("Aucune statistique disponible")
            return
        
        print("\n=== Statistiques détaillées ===")
        
        # Comptage par famille de véhicules
        vehicle_counts = {}
        for route in routes:
            family = int(route.family)
            vehicle_counts[family] = vehicle_counts.get(family, 0) + 1
        
        for family in sorted(vehicle_counts.keys()):
            count = vehicle_counts[family]
            vehicle = self.vehicles[family - 1]
            print(f"Véhicules famille {family}: {count} (capacité: {vehicle.max_capacity}kg)")
        
        # Statistiques des routes
        route_lengths = [len(route.orders) for route in routes]
        route_weights = [route.capacity_used for route in routes]
        
        print(f"\nTaille moyenne des routes: {np.mean(route_lengths):.1f} commandes")
        print(f"Charge moyenne: {np.mean(route_weights):.1f} kg")
        print(f"Route la plus grande: {max(route_lengths)} commandes")
        print(f"Route la plus petite: {min(route_lengths)} commandes")
        print(f"Nombre total de commandes livrées: {sum(route_lengths)}")

def process_instance(instance_number):
    """Traite une instance spécifique"""
    instance_file = f"instance_{instance_number:02d}.csv"
    output_file = f"routes_{instance_number:02d}.csv"
    
    if not os.path.exists(instance_file):
        print(f"❌ Fichier {instance_file} non trouvé")
        return None
    
    print(f"\n{'='*60}")
    print(f"TRAITEMENT DE {instance_file.upper()}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        solver = CalifraisSolver("vehicles.csv", instance_file)
        routes = solver.solve()
        solver.write_solution(routes, output_file)
        solver.print_statistics(routes)
        
        # Calcul du coût
        total_cost = 0.0
        for route in routes:
            family_int = int(route.family)
            vehicle = solver.vehicles[family_int - 1]
            total_cost += solver.calculate_route_cost(route, vehicle)
        
        end_time = time.time()
        print(f"⏱️  Temps d'exécution: {end_time - start_time:.2f} secondes")
        print(f"💰 Coût total pour {instance_file}: {total_cost:.2f} €")
        
        return total_cost
        
    except Exception as e:
        end_time = time.time()
        print(f"❌ Erreur après {end_time - start_time:.2f} secondes: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_all_instances():
    """Traite automatiquement les 10 instances"""
    print("=== TRAITEMENT DE TOUTES LES INSTANCES CALIFRAIS ===\n")
    
    total_cost_all_instances = 0.0
    successful_instances = 0
    
    for i in range(1, 11):
        cost = process_instance(i)
        if cost is not None:
            total_cost_all_instances += cost
            successful_instances += 1
    
    print(f"\n{'='*60}")
    print("RÉSUMÉ FINAL")
    print(f"{'='*60}")
    print(f"Instances traitées avec succès: {successful_instances}/10")
    print(f"Coût total pour toutes les instances: {total_cost_all_instances:.2f} €")
    
    # Lister les fichiers générés
    print(f"\nFichiers de solution générés:")
    for i in range(1, 11):
        route_file = f"routes_{i:02d}.csv"
        if os.path.exists(route_file):
            file_size = os.path.getsize(route_file)
            print(f"  ✅ {route_file} ({file_size} octets)")
        else:
            print(f"  ❌ {route_file} (non généré)")

def main():
    print("=== Solveur Califrais KIRO 2025 - Contraintes Temporelles Strictes ===\n")
    
    print("Options disponibles:")
    print("1. Traiter toutes les instances (01 à 10)")
    print("2. Traiter une instance spécifique")
    
    choice = input("\nChoisissez une option (1 ou 2): ").strip()
    
    if choice == "1":
        process_all_instances()
    elif choice == "2":
        try:
            instance_num = int(input("Entrez le numéro de l'instance (1-10): "))
            if 1 <= instance_num <= 10:
                process_instance(instance_num)
            else:
                print("❌ Numéro d'instance invalide. Doit être entre 1 et 10.")
        except ValueError:
            print("❌ Veuillez entrer un nombre valide.")
    else:
        print("Option non reconnue. Traitement de toutes les instances...")
        process_all_instances()

if __name__ == "__main__":
    main()