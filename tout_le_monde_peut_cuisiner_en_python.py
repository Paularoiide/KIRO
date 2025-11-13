import csv
import math
import sys
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

# ======================================================
# PARAMÈTRES À MODIFIER SELON L’INSTANCE À TESTER
# ======================================================
INSTANCE_PATH = "instance_02.csv"     # ⬅️ CHANGE JUSTE CECI
VEHICLES_PATH = "vehicles.csv"
OUTPUT_PATH = "routes.csv"

# ======================================================
# CONSTANTES MATHÉMATIQUES
# ======================================================

R_EARTH = 6_371_000.0
T_DAY = 86400.0
OMEGA = 2.0 * math.pi / T_DAY

# ======================================================
# STRUCTURES DE DONNÉES
# ======================================================

@dataclass
class VehicleFamily:
    family: int
    max_capacity: float
    rental_cost: float
    fuel_cost: float
    radius_cost: float
    speed: float
    parking_time: float
    cos_coeffs: List[float]
    sin_coeffs: List[float]


@dataclass
class Node:
    idx: int
    lat: float
    lon: float
    weight: float
    window_start: Optional[float]
    window_end: Optional[float]
    service_time: float
    x: float = 0.0
    y: float = 0.0


# ======================================================
# LECTURE DES FICHIERS CSV
# ======================================================

def load_vehicles(path: str) -> Dict[int, VehicleFamily]:
    families = {}
    with open(path, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fam = int(row["family"])
            families[fam] = VehicleFamily(
                family=fam,
                max_capacity=float(row["max_capacity"]),
                rental_cost=float(row["rental_cost"]),
                fuel_cost=float(row["fuel_cost"]),
                radius_cost=float(row["radius_cost"]),
                speed=float(row["speed"]),
                parking_time=float(row["parking_time"]),
                cos_coeffs=[float(row[f"fourier_cos_{i}"]) for i in range(4)],
                sin_coeffs=[float(row[f"fourier_sin_{i}"]) for i in range(4)],
            )
    return families


def load_instance(path: str) -> List[Node]:
    nodes = []
    with open(path, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            def fp(v):
                return float(v) if v not in ("", None) else None

            idx = int(row["id"])
            nodes.append(Node(
                idx=idx,
                lat=float(row["latitude"]),
                lon=float(row["longitude"]),
                weight=fp(row["order_weight"]) or 0.0,
                window_start=fp(row["window_start"]),
                window_end=fp(row["window_end"]),
                service_time=fp(row["delivery_duration"]) or 0.0,
            ))
    return nodes


# ======================================================
# GÉOMÉTRIE ET TEMPS
# ======================================================

def compute_coordinates(nodes: List[Node]):
    depot = nodes[0]
    phi0 = math.radians(depot.lat)

    for n in nodes:
        dphi = math.radians(n.lat - depot.lat)
        dlam = math.radians(n.lon - depot.lon)
        n.y = R_EARTH * dphi
        n.x = R_EARTH * math.cos(phi0) * dlam


def compute_manhattan(nodes):
    n = len(nodes)
    M = [[0]*n for _ in range(n)]
    for i in range(n):
        xi, yi = nodes[i].x, nodes[i].y
        for j in range(n):
            xj, yj = nodes[j].x, nodes[j].y
            M[i][j] = abs(xj - xi) + abs(yj - yi)
    return M


def gamma_t(fam: VehicleFamily, t: float) -> float:
    g = 0
    for k in range(4):
        g += fam.cos_coeffs[k] * math.cos(k * OMEGA * t)
        g += fam.sin_coeffs[k] * math.sin(k * OMEGA * t)
    return max(0.1, min(g, 10))


def travel_time(fam: VehicleFamily, M, i, j, t):
    base = M[i][j] / fam.speed + fam.parking_time
    return base * gamma_t(fam, t)


# ======================================================
# HEURISTIQUE GLOUTONNE
# ======================================================

def choose_family(families):
    return max(families.values(), key=lambda f: (f.max_capacity, -f.rental_cost))


def build_routes(nodes, fam: VehicleFamily, M):
    unserved = set(range(1, len(nodes)))
    routes = []

    while unserved:
        route = []
        load = 0
        t = 0
        cur = 0

        while True:
            best = None
            best_d = None

            for i in list(unserved):
                node = nodes[i]

                if load + node.weight > fam.max_capacity:
                    continue

                tt = travel_time(fam, M, cur, i, t)
                arr = t + tt

                ws = node.window_start or 0
                we = node.window_end or 10**12
                start = max(arr, ws)

                if start > we:
                    continue

                d = M[cur][i]
                if best is None or d < best_d:
                    best = (i, start + node.service_time, start)
                    best_d = d

            if best is None:
                break

            i, new_t, start = best
            unserved.remove(i)
            route.append(i)
            load += nodes[i].weight
            t = new_t
            cur = i

        routes.append(route)

    return routes


# ======================================================
# EXPORT CSV FINAL
# ======================================================

def write_routes_csv(routes, fam, nodes, path):
    maxlen = max(len(r) for r in routes)
    header = ["family"] + [f"order_{i+1}" for i in range(maxlen)]

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)

        for r in routes:
            ids = [nodes[i].idx for i in r]
            w.writerow([fam.family] + ids + [""]*(maxlen - len(ids)))


# ======================================================
# MAIN
# ======================================================

def main():
    print("Chargement…")

    families = load_vehicles(VEHICLES_PATH)
    nodes = load_instance(INSTANCE_PATH)

    compute_coordinates(nodes)
    M = compute_manhattan(nodes)

    fam = choose_family(families)
    print(f"→ Famille choisie : {fam.family} (capacité = {fam.max_capacity})")

    routes = build_routes(nodes, fam, M)
    print(f"→ {len(routes)} tournées générées")

    write_routes_csv(routes, fam, nodes, OUTPUT_PATH)
    print(f"→ Fichier '{OUTPUT_PATH}' écrit avec succès !")


if __name__ == "__main__":
    main()
