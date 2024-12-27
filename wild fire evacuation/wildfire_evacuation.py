import numpy as np
import random
from collections import deque
from tkinter import Tk,Canvas
import heapq

rows,cols=30,30

grass,forest,water,fire,hill=1,2,3,4,5
house,safehouse=10,11
buffer_zone=12

def generate_grid(rows,cols):
    grid=np.full((rows,cols),grass) 
    add_clusters(grid,terrain=water,cluster_size=8,num_clusters=10) 
    add_clusters(grid,terrain=hill,cluster_size=8,num_clusters=8)
    add_forest(grid) 
    return grid

def add_clusters(grid,terrain,cluster_size,num_clusters):
    directions=[(-1,0),(1,0),(0,-1),(0,1)] 
    for _ in range(num_clusters):
        retries=100
        while retries>0:
            start_x,start_y=random.randint(0,rows-1),random.randint(0,cols-1)
            if grid[start_x,start_y]==grass:
                break
            retries-=1
        else:
            continue
        queue=deque([(start_x,start_y)])
        grid[start_x,start_y]=terrain
        filled=1
        while queue and filled<cluster_size:
            x,y=queue.popleft()
            for dx,dy in directions:
                nx,ny=x+dx,y+dy
                if 0<=nx<rows and 0<=ny<cols and grid[nx,ny]==grass:
                    grid[nx,ny]=terrain
                    queue.append((nx,ny))
                    filled+=1
                    if filled>=cluster_size:
                        break

def add_forest(grid):
    for i in range(rows):
        for j in range(cols):
            if grid[i,j]==grass and random.random()<0.2:
                grid[i,j]=forest

def add_features(grid,num_houses=10,num_safehouses=5,min_distance=5):
    features={}
    houses=place_random_points(grid,num_houses,avoid_values=[water,hill])
    for hx,hy in houses:
        grid[hx][hy]=house
    features['houses']=houses
    safehouses=place_safehouses_far_from_houses(grid,houses,num_safehouses,min_distance)
    for sx,sy in safehouses:
        grid[sx][sy]=safehouse
    features['safehouses']=safehouses
    return features

def place_safehouses_far_from_houses(grid,houses,num_safehouses,min_distance):
    safehouses=set()
    for _ in range(num_safehouses):
        retries=100
        while retries>0:
            sx,sy=random.randint(0,rows-1),random.randint(0,cols-1)
            too_close=False
            for hx,hy in houses:
                if abs(sx-hx)<min_distance and abs(sy-hy)<min_distance:
                    too_close=True
                    break
            if grid[sx][sy] not in [water,hill,house] and not too_close:
                safehouses.add((sx,sy))
                break
            retries-=1
        else:
            continue 
    return list(safehouses)

def place_random_points(grid,count,avoid_values):
    points=set()
    while len(points)<count:
        x,y=random.randint(0,rows-1), random.randint(0,cols-1)
        if grid[x][y] not in avoid_values:
            points.add((x,y))
    return list(points)

def create_buffer_zones(grid,features,buffer_radius=1):
    for house in features['houses']:
        mark_buffer_zone(grid,house,buffer_radius,buffer_type='house')
    for safehouse in features['safehouses']:
        mark_buffer_zone(grid,safehouse,buffer_radius,buffer_type='safehouse')

def mark_buffer_zone(grid,center,radius,buffer_type):
    cx,cy=center
    for x in range(max(0,cx-radius), min(rows,cx+radius+1)):
        for y in range(max(0,cy-radius),min(cols,cy+radius+1)):
            if grid[x][y]!=house and grid[x][y]!=safehouse:
                if buffer_type=='house':
                    grid[x][y]=buffer_zone 
                elif buffer_type=='safehouse':
                    grid[x][y]=buffer_zone  

terrain_costs={
    grass:1,     
    forest:2,    
    water:1000,  
    fire:1000000,
    hill:5,      
    house:1000,  
    safehouse:1000, 
    buffer_zone:500,
}

def a_star(grid,start,goal):
    def heuristic(a,b):
        return abs(a[0]-b[0])+abs(a[1]-b[1])
    open_set=[]
    heapq.heappush(open_set,(0+heuristic(start,goal),0,start))
    came_from={}
    g_score={start:0}
    f_score={start:heuristic(start,goal)}
    directions=[(-1,0),(1,0),(0,-1),(0,1)]
    while open_set:
        _,current_g,current=heapq.heappop(open_set)
        if current==goal:
            path=[]
            while current in came_from:
                path.append(current)
                current=came_from[current]
            path.reverse()
            return path
        for dx,dy in directions:
            neighbor=(current[0]+dx,current[1]+dy)
            if 0<=neighbor[0]<rows and 0<=neighbor[1]<cols:
                if grid[neighbor[0]][neighbor[1]]==fire or grid[neighbor[0]][neighbor[1]]==path_marker:
                    continue
                terrain=grid[neighbor[0]][neighbor[1]]
                terrain_cost=terrain_costs.get(terrain,1)
                tentative_g_score=current_g+terrain_cost
                if neighbor not in g_score or tentative_g_score<g_score[neighbor]:
                    came_from[neighbor]=current
                    g_score[neighbor]=tentative_g_score
                    f_score[neighbor]=tentative_g_score+heuristic(neighbor,goal)
                    heapq.heappush(open_set,(f_score[neighbor],tentative_g_score,neighbor))
    return [] 

def clear_old_paths(grid):
    for x in range(rows):
        for y in range(cols):
            if grid[x][y]==path_marker:
                grid[x][y]=initial_grid[x][y]  

def spread_fire(grid, features):
    directions=[(-1,0),(1,0),(0,-1),(0,1)]
    new_fire=[]
    houses_to_pathfind=[] 
    for x in range(rows):
        for y in range(cols):
            if grid[x][y]==fire:
                for dx,dy in directions:
                    nx,ny=x+dx,y+dy
                    if 0<=nx<rows and 0<=ny<cols:
                        if grid[nx][ny]==buffer_zone and is_near_house(nx,ny,features):
                            house_pos=find_nearest_house(nx,ny,features)
                            if house_pos:
                                houses_to_pathfind.append(house_pos)
    houses_to_pathfind=list(set(houses_to_pathfind))
    all_paths=[]
    for house_pos in houses_to_pathfind:
        safehouse_pos=find_nearest_safehouse(house_pos,features)
        if safehouse_pos:
            clear_old_paths(grid)
            path=a_star(grid,house_pos,safehouse_pos)
            if path:
                all_paths.append(path)
    for path in all_paths:
        draw_path_on_grid(path,grid,features)
    for x in range(rows):
        for y in range(cols):
            if grid[x][y]==fire:
                for dx,dy in directions:
                    nx,ny=x+dx,y+dy
                    if 0<=nx<rows and 0<=ny<cols:
                        if grid[nx][ny]==buffer_zone and is_in_safehouse_buffer(nx,ny):
                            continue
                        if grid[nx][ny] in [grass,forest,buffer_zone,path_marker]:
                            if random.random()<0.05: 
                                new_fire.append((nx,ny))
    for fx, fy in new_fire:
        grid[fx][fy]=fire

def is_near_house(x,y,features):
    for hx,hy in features['houses']:
        if abs(x-hx)<=1 and abs(y-hy)<=1:
            return True
    return False

def find_nearest_house(x,y,features):
    min_dist=float('inf')
    nearest_house=None
    for hx,hy in features['houses']:
        dist=abs(x-hx)+abs(y-hy)
        if dist<min_dist:
            min_dist=dist
            nearest_house=(hx,hy)
    return nearest_house

def find_nearest_safehouse(house_pos,features):
    min_dist=float('inf')
    nearest_safehouse=None
    for sx,sy in features['safehouses']:
        dist=abs(house_pos[0]-sx)+abs(house_pos[1]-sy)
        if dist<min_dist:
            min_dist=dist
            nearest_safehouse=(sx,sy)
    return nearest_safehouse

path_marker=13

def draw_path_on_grid(path,grid,features):
    for x,y in path:
        if grid[x][y] not in [buffer_zone,house,safehouse]:
            grid[x][y]=path_marker

def is_in_safehouse_buffer(x,y):
    for sx,sy in features['safehouses']:
        if abs(x-sx)<=1 and abs(y-sy)<=1:
            return True
    return False

def interactive_fire(grid):
    cell_size=20
    root=Tk()
    root.title("Evacuation during Forest Fires")
    canvas=Canvas(root,width=cols*cell_size,height=rows*cell_size)
    canvas.pack()
    def draw_grid():
        canvas.delete("all")
        color_map={
            grass:"lightgreen",
            forest:"darkgreen",
            water:"blue",
            fire:"orange",
            hill:"brown",
            house:"pink",
            safehouse:"black",
            buffer_zone:"yellow",
            path_marker:"cyan" 
        }
        for i in range(rows):
            for j in range(cols):
                x1,y1=j*cell_size,i*cell_size
                x2,y2=x1+cell_size,y1+cell_size
                color=color_map[grid[i,j]]
                canvas.create_rectangle(x1,y1,x2,y2,fill=color,outline="gray")

    def add_fire(event):
        x,y=event.y//cell_size,event.x//cell_size
        if 0<=x<rows and 0<=y<cols:
            if grid[x][y]==grass or grid[x][y]==forest:
                if grid[x][y]==grass or grid[x][y]==forest or grid[x][y]==buffer_zone and not is_in_safehouse_buffer(x,y):
                    grid[x][y]=fire
                    draw_grid()
    canvas.bind("<Button-1>",add_fire)

    def update_fire():
        spread_fire(grid,features)
        draw_grid()
        root.after(1000,update_fire)

    draw_grid()
    update_fire()
    root.mainloop()

grid=generate_grid(rows,cols)
initial_grid=np.copy(grid)
features=add_features(grid)
create_buffer_zones(grid,features)

interactive_fire(grid)