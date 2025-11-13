using CSV, DataFrames, LinearAlgebra, Dates, Random

# Structures de données
struct Vehicle
    family::Int
    max_capacity::Float64
    rental_cost::Float64
    fuel_cost::Float64
    radius_cost::Float64
    speed::Float64
    parking_time::Float64
    fourier_coeffs::Vector{Float64}
end

struct Order
    id::Int
    latitude::Float64
    longitude::Float64
    weight::Float64
    window_start::Float64
    window_end::Float64
    delivery_duration::Float64
end

struct Route
    family::Int
    orders::Vector{Int}
end

# Conversion des coordonnées
const EARTH_RADIUS = 6.371e6
const DEPOT_LAT = 48.764246

function convert_coordinates(lat, lon)
    y = EARTH_RADIUS * (2π/360) * (lat - DEPOT_LAT)
    x = EARTH_RADIUS * cos(2π/360 * DEPOT_LAT) * (2π/360) * (lon - 2.34842)
    return (x, y)
end

# Calcul des distances
function manhattan_distance(coord1, coord2)
    return abs(coord2[1] - coord1[1]) + abs(coord2[2] - coord1[2])
end

function euclidean_distance(coord1, coord2)
    return sqrt((coord2[1] - coord1[1])^2 + (coord2[2] - coord1[2])^2)
end

# Calcul du temps de trajet
function travel_time(vehicle::Vehicle, dist_manhattan, departure_time)
    # Temps de référence
    τ_ref = dist_manhattan / vehicle.speed + vehicle.parking_time
    
    # Facteur temporel (Fourier series)
    ω = 2π / 86400.0
    t = departure_time
    γ = vehicle.fourier_coeffs[1] + 
        vehicle.fourier_coeffs[2] * cos(ω*t) + vehicle.fourier_coeffs[3] * sin(ω*t) +
        vehicle.fourier_coeffs[4] * cos(2ω*t) + vehicle.fourier_coeffs[5] * sin(2ω*t) +
        vehicle.fourier_coeffs[6] * cos(3ω*t) + vehicle.fourier_coeffs[7] * sin(3ω*t)
    
    return τ_ref * γ
end

# Fonction de coût pour une route
function route_cost(route::Route, vehicle::Vehicle, orders_dict::Dict{Int,Order}, coords_dict::Dict{Int,Tuple{Float64,Float64}})
    # Coût de location
    cost = vehicle.rental_cost
    
    # Coût de carburant (distance de Manhattan)
    total_manhattan = 0.0
    sequence = [0; route.orders; 0]  # Dépôt -> commandes -> dépôt
    
    for i in 1:(length(sequence)-1)
        dist = manhattan_distance(coords_dict[sequence[i]], coords_dict[sequence[i+1]])
        total_manhattan += dist
    end
    cost += vehicle.fuel_cost * total_manhattan
    
    # Coût de rayon (diamètre Euclidien au carré)
    if !isempty(route.orders)
        order_coords = [coords_dict[order_id] for order_id in route.orders]
        max_distance = 0.0
        for i in eachindex(order_coords)
            for j in (i+1):length(order_coords)
                dist = euclidean_distance(order_coords[i], order_coords[j])
                max_distance = max(max_distance, dist)
            end
        end
        cost += vehicle.radius_cost * (0.5 * max_distance)^2
    end
    
    return cost
end

# Vérification des fenêtres de temps
function check_time_windows(route::Route, vehicle::Vehicle, orders_dict::Dict{Int,Order}, coords_dict::Dict{Int,Tuple{Float64,Float64}})
    current_time = 0.0  # Départ du dépôt
    current_pos = 0
    
    for order_id in route.orders
        order = orders_dict[order_id]
        dist = manhattan_distance(coords_dict[current_pos], coords_dict[order_id])
        arrival_time = current_time + travel_time(vehicle, dist, current_time)
        
        if arrival_time < order.window_start
            arrival_time = order.window_start  # Attente
        end
        
        if arrival_time > order.window_end
            return false
        end
        
        current_time = arrival_time + order.delivery_duration
        current_pos = order_id
    end
    
    return true
end

# Algorithme de construction par plus proche voisin
function nearest_neighbor_construction(orders::Vector{Order}, vehicles::Vector{Vehicle}, coords_dict::Dict{Int,Tuple{Float64,Float64}})
    orders_dict = Dict(o.id => o for o in orders)
    unserved = Set([o.id for o in orders])
    routes = Vector{Route}()
    
    # Trier les véhicules par capacité (du plus petit au plus grand)
    sorted_vehicles = sort(vehicles, by=v->v.max_capacity)
    
    while !isempty(unserved)
        # Choisir le véhicule le plus petit pouvant servir au moins une commande
        current_vehicle = sorted_vehicles[1]
        
        # Démarrer une nouvelle route depuis le dépôt
        current_route = Int[]
        current_time = 0.0  # Départ à t=0
        current_load = 0.0
        current_pos = 0  # Dépôt
        
        while !isempty(unserved)
            # Trouver la commande non servie la plus proche
            best_order = nothing
            best_distance = Inf
            best_arrival_time = 0.0
            
            for order_id in unserved
                order = orders_dict[order_id]
                
                # Vérifier la capacité
                if current_load + order.weight > current_vehicle.max_capacity
                    continue
                end
                
                # Calculer la distance et le temps d'arrivée
                dist = manhattan_distance(coords_dict[current_pos], coords_dict[order_id])
                arrival_time = current_time + travel_time(current_vehicle, dist, current_time)
                
                # Vérifier la fenêtre de temps
                if arrival_time <= order.window_end  # On peut attendre pour window_start
                    if dist < best_distance
                        best_order = order_id
                        best_distance = dist
                        best_arrival_time = arrival_time
                    end
                end
            end
            
            if best_order === nothing
                break  # Plus de commandes réalisables pour ce véhicule
            end
            
            # Ajouter la commande à la route
            push!(current_route, best_order)
            current_load += orders_dict[best_order].weight
            current_time = max(best_arrival_time, orders_dict[best_order].window_start) + orders_dict[best_order].delivery_duration
            current_pos = best_order
            delete!(unserved, best_order)
        end
        
        if !isempty(current_route)
            push!(routes, Route(current_vehicle.family, current_route))
        else
            # Si aucun véhicule ne peut servir les commandes restantes, on passe au véhicule suivant
            if length(sorted_vehicles) > 1
                sorted_vehicles = sorted_vehicles[2:end]
            else
                error("Impossible de servir toutes les commandes avec les véhicules disponibles")
            end
        end
    end
    
    return routes
end

# Amélioration par échange 2-opt
function two_opt_improvement!(route::Route, vehicle::Vehicle, orders_dict::Dict{Int,Order}, coords_dict::Dict{Int,Tuple{Float64,Float64}})
    improved = true
    current_cost = route_cost(route, vehicle, orders_dict, coords_dict)
    
    while improved
        improved = false
        for i in 1:(length(route.orders)-1)
            for j in (i+1):length(route.orders)
                # Créer une nouvelle route en inversant le segment i-j
                new_orders = vcat(route.orders[1:i-1], reverse(route.orders[i:j]), route.orders[j+1:end])
                new_route = Route(route.family, new_orders)
                new_cost = route_cost(new_route, vehicle, orders_dict, coords_dict)
                
                if new_cost < current_cost && check_time_windows(new_route, vehicle, orders_dict, coords_dict)
                    route.orders = new_orders
                    current_cost = new_cost
                    improved = true
                    break
                end
            end
            improved && break
        end
    end
end

# Échange inter-route
function inter_route_exchange!(routes::Vector{Route}, vehicles_dict::Dict{Int,Vehicle}, orders_dict::Dict{Int,Order}, coords_dict::Dict{Int,Tuple{Float64,Float64}})
    improved = true
    
    while improved
        improved = false
        
        for i in eachindex(routes)
            for j in (i+1):length(routes)
                for pos_i in 1:length(routes[i].orders)
                    for pos_j in 1:length(routes[j].orders)
                        # Essayer d'échanger deux commandes
                        route1_orders = copy(routes[i].orders)
                        route2_orders = copy(routes[j].orders)
                        
                        order1 = route1_orders[pos_i]
                        order2 = route2_orders[pos_j]
                        
                        # Vérifier les capacités
                        vehicle1 = vehicles_dict[routes[i].family]
                        vehicle2 = vehicles_dict[routes[j].family]
                        
                        load1 = sum(orders_dict[o].weight for o in route1_orders) - orders_dict[order1].weight + orders_dict[order2].weight
                        load2 = sum(orders_dict[o].weight for o in route2_orders) - orders_dict[order2].weight + orders_dict[order1].weight
                        
                        if load1 <= vehicle1.max_capacity && load2 <= vehicle2.max_capacity
                            # Effectuer l'échange
                            route1_orders[pos_i] = order2
                            route2_orders[pos_j] = order1
                            
                            new_route1 = Route(routes[i].family, route1_orders)
                            new_route2 = Route(routes[j].family, route2_orders)
                            
                            if check_time_windows(new_route1, vehicle1, orders_dict, coords_dict) &&
                               check_time_windows(new_route2, vehicle2, orders_dict, coords_dict)
                               
                                old_cost = route_cost(routes[i], vehicle1, orders_dict, coords_dict) + 
                                         route_cost(routes[j], vehicle2, orders_dict, coords_dict)
                                new_cost = route_cost(new_route1, vehicle1, orders_dict, coords_dict) + 
                                         route_cost(new_route2, vehicle2, orders_dict, coords_dict)
                                
                                if new_cost < old_cost
                                    routes[i].orders = route1_orders
                                    routes[j].orders = route2_orders
                                    improved = true
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

# Fonction principale
function solve_vrp(instance_file::String, vehicles_file::String)
    # Charger les données
    vehicles_df = CSV.read(vehicles_file, DataFrame)
    instance_df = CSV.read(instance_file, DataFrame)
    
    # Créer les véhicules
    vehicles = Vector{Vehicle}()
    for row in eachrow(vehicles_df)
        fourier_coeffs = [row.fourier_cos_0, row.fourier_cos_1, row.fourier_sin_1,
                         row.fourier_cos_2, row.fourier_sin_2, row.fourier_cos_3, row.fourier_sin_3]
        push!(vehicles, Vehicle(
            row.family, row.max_capacity, row.rental_cost, row.fuel_cost,
            row.radius_cost, row.speed, row.parking_time, fourier_coeffs
        ))
    end
    
    # Créer les commandes (en ignorant le dépôt)
    orders = Vector{Order}()
    coords_dict = Dict{Int,Tuple{Float64,Float64}}()
    
    for row in eachrow(instance_df)
        if row.id != 0  # Ignorer le dépôt
            order = Order(
                row.id, row.latitude, row.longitude,
                coalesce(row.order_weight, 0.0), 
                coalesce(row.window_start, 0.0), 
                coalesce(row.window_end, Inf),
                coalesce(row.delivery_duration, 0.0)
            )
            push!(orders, order)
            coords_dict[row.id] = convert_coordinates(row.latitude, row.longitude)
        end
    end
    
    # Ajouter les coordonnées du dépôt
    depot_row = instance_df[instance_df.id .== 0, :]
    if nrow(depot_row) > 0
        coords_dict[0] = convert_coordinates(depot_row.latitude[1], depot_row.longitude[1])
    else
        # Coordonnées par défaut du dépôt
        coords_dict[0] = (0.0, 0.0)
    end
    
    # Construction initiale
    println("Construction de la solution initiale...")
    routes = nearest_neighbor_construction(orders, vehicles, coords_dict)
    
    # Améliorations
    println("Amélioration de la solution...")
    orders_dict = Dict(o.id => o for o in orders)
    vehicles_dict = Dict(v.family => v for v in vehicles)
    
    # Amélioration intra-route
    for route in routes
        two_opt_improvement!(route, vehicles_dict[route.family], orders_dict, coords_dict)
    end
    
    # Amélioration inter-route
    inter_route_exchange!(routes, vehicles_dict, orders_dict, coords_dict)
    
    # Calcul du coût total
    total_cost = 0.0
    for route in routes
        total_cost += route_cost(route, vehicles_dict[route.family], orders_dict, coords_dict)
    end
    
    println("Coût total: ", total_cost)
    println("Nombre de routes: ", length(routes))
    
    return routes
end

# Fonction pour écrire la solution
function write_solution(routes::Vector{Route}, output_file::String)
    # Trouver le nombre maximum de commandes dans une route
    if isempty(routes)
        max_orders = 0
    else
        max_orders = maximum(length(route.orders) for route in routes)
    end
    
    # Créer le DataFrame
    data = []
    for route in routes
        row = Dict(:family => route.family)
        for i in 1:max_orders
            col_name = Symbol("order_$i")
            if i <= length(route.orders)
                row[col_name] = route.orders[i]
            else
                row[col_name] = missing
            end
        end
        push!(data, row)
    end
    
    # Créer le DataFrame avec les bonnes colonnes
    if !isempty(data)
        df = DataFrame(data)
        # Réorganiser les colonnes: family puis order_1, order_2, ...
        col_order = [:family]
        for i in 1:max_orders
            push!(col_order, Symbol("order_$i"))
        end
        df = df[:, col_order]
    else
        # Si pas de routes, créer un DataFrame vide avec les bonnes colonnes
        df = DataFrame(family=Int[])
        for i in 1:max_orders
            df[!, Symbol("order_$i")] = Int[]
        end
    end
    
    # Écrire le fichier CSV
    CSV.write(output_file, df)
    println("Solution écrite dans: ", output_file)
end

# Exemple d'utilisation
function main()
    # Chemin vers le dossier contenant les instances
    instance_dir = "Instances"
    vehicles_file = "vehicles.csv"
    
    instance_files = [
        "instance_01.csv", "instance_02.csv", "instance_03.csv", 
        "instance_04.csv", "instance_05.csv", "instance_06.csv",
        "instance_07.csv", "instance_08.csv", "instance_09.csv", 
        "instance_10.csv"
    ]
    
    for instance_file in instance_files
        println("Traitement de: ", instance_file)
        try
            # Utiliser le chemin complet
            instance_path = joinpath(instance_dir, instance_file)
            routes = solve_vrp(instance_path, vehicles_file)
            output_file = replace(instance_file, ".csv" => "_solution.csv")
            write_solution(routes, output_file)
        catch e
            println("Erreur avec ", instance_file, ": ", e)
        end
        println()
    end
end

# Exécuter le programme
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end