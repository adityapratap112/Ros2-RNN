import cv2
import numpy as np
import heapq
import yaml
import os

def load_map(yaml_path):
    with open(yaml_path, 'r') as f:
        info = yaml.safe_load(f)
    map_img = cv2.imread(yaml_path.replace('.yaml', '.pgm'), cv2.IMREAD_GRAYSCALE)
    occ_grid = (map_img < 250).astype(np.uint8)
    resolution = info['resolution']
    origin = info['origin']
    return occ_grid, resolution, origin

def dijkstra(grid, start, goal):
    h, w = grid.shape
    visited = np.zeros_like(grid)
    dist = np.full_like(grid, np.inf, dtype=float)
    prev = {}
    q = [(0, start)]
    dist[start] = 0

    while q:
        d, current = heapq.heappop(q)
        if current == goal:
            break
        if visited[current]:
            continue
        visited[current] = 1
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = current[0]+dx, current[1]+dy
            if 0 <= nx < h and 0 <= ny < w and grid[nx, ny] == 0:
                alt = d + 1
                if alt < dist[nx, ny]:
                    dist[nx, ny] = alt
                    prev[(nx, ny)] = current
                    heapq.heappush(q, (alt, (nx, ny)))

    path = []
    p = goal
    while p != start:
        path.append(p)
        p = prev.get(p, start)
    path.append(start)
    path.reverse()
    return path

if __name__ == "__main__":
    yaml_path = "/home/user/map.yaml"
    occ_grid, res, origin = load_map(yaml_path)

    start = (100, 100)
    goal = (30, 50)

    path_px = dijkstra(occ_grid, start, goal)
    path_world = [(res * y + origin[0], res * (occ_grid.shape[0] - 1 - x) + origin[1]) for x, y in path_px]

    save_path = "dijkstra_path.csv"
    np.savetxt(save_path, path_world, delimiter=",", header="x,y", comments='')
    print(f"[✅] Path saved to {save_path}")
