import json
import heapq
import math

with open('G.json', 'r') as f:
   G = json.load(f)

with open('Coord.json', 'r') as f:
    Coord = json.load(f)
        
with open('Dist.json', 'r') as f:
    Dist = json.load(f)
        
with open('Cost.json', 'r') as f:
    Cost = json.load(f)

## helper functions:

def reconstruct_path(parent, start, end):
    path = []
    current = end
    
    while current is not None:
        path.append(current)
        current = parent.get(current)
        
    path.reverse()
    return path


def calculate_path_stats(path):
    total_distance = 0
    total_energy = 0
        
    for i in range(len(path) - 1):
        edge_key = f"{path[i]},{path[i+1]}"
        total_distance += Dist[edge_key]
        total_energy += Cost[edge_key]
    
    return total_distance, total_energy
    

def print_result(task_num, path, distance, energy):
    print(f"Task {task_num}:")
    print(f"Shortest Path: {' -> '.join(path)}")
    print(f"Shortest Distance: {distance:.2f}")
    print(f"Total Energy Cost: {energy:.2f}\n")


## TASK 1 --> using Uniform Cost Search (UCS) without Energy Constraint
def task1_UCS(start, end):
    frontier = [(0, start)] # (distance, node)
    distance_to = {start: 0}  # to find best dist to reach each node
    parent = {start: None} # track parent node for path reconstuction
    explored = set() 
    nodes_explored = 0

    while frontier:
        current_dist, current_node = heapq.heappop(frontier)
        
        if current_node in explored:
            continue
        
        explored.add(current_node)
        nodes_explored += 1
        
        if current_node == end:
            break

        if current_node not in G:
            continue
        
        for neighbor in G[current_node]:
            if neighbor in explored:
                continue

            edge_key = f"{current_node},{neighbor}"
            if edge_key not in Dist:
                continue

            new_dist = current_dist + Dist[edge_key]
            
            if neighbor not in distance_to or new_dist < distance_to[neighbor]:
                distance_to[neighbor] = new_dist
                parent[neighbor] = current_node
                heapq.heappush(frontier, (new_dist, neighbor))
    
    if end not in parent:
        return None
    
    path = reconstruct_path(parent, start, end)
    distance, energy = calculate_path_stats(path)
    return path, distance, energy 


## TASK 2 --> using UCS with Energy Constraint
def task2_UCS(start, end, energy_budget):
    frontier = [(0, 0, start)] # (distance, energy used, node)
    best_dist = {start: (0, 0)}  # (distance, energy) 
    parent = {start: None} # track parent node for path reconstuction
    explored = {} # track explored nodes with best energy used w/ dictionary
    nodes_explored = 0

    while frontier:
        current_dist, current_energy, current_node = heapq.heappop(frontier)
        
        if current_node in explored:
            if explored[current_node] <= current_energy:
                continue
        
        explored[current_node] = current_energy
        nodes_explored += 1
        
        if current_node == end:
            break

        if current_node not in G:
            continue
        
        for neighbor in G[current_node]:
            edge = f"{current_node},{neighbor}"

            if edge not in Dist or edge not in Cost:
                continue

            new_dist = current_dist + Dist[edge]
            new_energy = current_energy + Cost[edge]

            if new_energy > energy_budget: # exceeds budget == skip
                continue

            if neighbor not in best_dist or new_dist < best_dist[neighbor][0]:
                best_dist[neighbor] = (new_dist, new_energy)
                parent[neighbor] = current_node
                heapq.heappush(frontier, (new_dist, new_energy, neighbor))
            
    if end not in parent:
        return None
    
    path = reconstruct_path(parent, start, end)
    distance, energy = calculate_path_stats(path)
    return path, distance, energy
## TASK 3 --> using A* 
def heuristic(node, goal):
    x1, y1 = Coord[node]
    x2, y2 = Coord[goal]
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def task3_Astar(start, end, energy_budget):

    frontier = [(heuristic(start, end), 0, 0, start)]  # (f, distance, energy, node)

    distance_to = {start: 0}
    parent = {start: None}

    explored = {}

    while frontier:

        f, current_dist, current_energy, current_node = heapq.heappop(frontier)

        if current_node in explored:
            if explored[current_node] <= current_energy:
                continue

        explored[current_node] = current_energy

        if current_node == end:
            break

        if current_node not in G:
            continue

        for neighbor in G[current_node]:

            edge = f"{current_node},{neighbor}"

            if edge not in Dist or edge not in Cost:
                continue

            new_dist = current_dist + Dist[edge]
            new_energy = current_energy + Cost[edge]

            if new_energy > energy_budget:
                continue

            if neighbor not in distance_to or new_dist < distance_to[neighbor]:

                distance_to[neighbor] = new_dist
                parent[neighbor] = current_node

                f = new_dist + heuristic(neighbor, end)

                heapq.heappush(frontier, (f, new_dist, new_energy, neighbor))

    if end not in parent:
        return None

    path = reconstruct_path(parent, start, end)
    distance, energy = calculate_path_stats(path)

    return path, distance, energy
def main():
    start_node = '1'
    end_node = '50'
    energy_budget = 287932
    print("====================================")
    print("SC3000 Lab Assignment 1")
    print("Part 1: Graph Search Algorithms")
    print("====================================\n")
    # run task 1:
    result1 = task1_UCS(start_node, end_node)

    if result1:
        path1, dist1, energy1 = result1
        print_result(1, path1, dist1, energy1)

    # run task 2:
    result2 = task2_UCS(start_node, end_node, energy_budget)

    if result2:
        path2, dist2, energy2 = result2
        print_result(2, path2, dist2, energy2)  
    # run task 3:
    result3 = task3_Astar(start_node, end_node, energy_budget)

    if result3:
        path3, dist3, energy3 = result3
        print_result(3, path3, dist3, energy3)

if __name__ == "__main__":
    main()
print("\n==============================")
print("Running Part 2 (Reinforcement Learning)")
print("==============================\n")

import runpy
runpy.run_module("part2", run_name="__main__")
 