

import os
import sys
import subprocess
import argparse
from functools import reduce

import gurobilp
import glob

import output_validator as ov
import plotly.graph_objs as go
import plotly
import networkx as nx
import numpy as np

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

from gurobipy import *



def data_parser(input_data):
    number_of_locations = int(input_data[0][0])
    number_of_houses = int(input_data[1][0])
    list_of_locations = input_data[2]
    list_of_houses = input_data[3]
    starting_location = input_data[4][0]

    adjacency_matrix = [[entry if entry == 'x' else float(entry) for entry in row] for row in input_data[5:]]
    return number_of_locations, number_of_houses, list_of_locations, list_of_houses, starting_location, adjacency_matrix

def adjacency_matrix_to_graph(adjacency_matrix):
    #print("Line 29, student utils, adjacency_matrix_to_graphfunction, adjacency_matrix", adjacency_matrix)
    node_weights = [adjacency_matrix[i][i] for i in range(len(adjacency_matrix))]
    adjacency_matrix_formatted = [[0 if entry == 'x' else entry for entry in row] for row in adjacency_matrix]

    for i in range(len(adjacency_matrix_formatted)):
        adjacency_matrix_formatted[i][i] = 0

    G = nx.convert_matrix.from_numpy_matrix(np.matrix(adjacency_matrix_formatted))

    message = ''

    for node, datadict in G.nodes.items():
        if node_weights[node] != 'x':
            message += 'The location {} has a road to itself. This is not allowed.\n'.format(node)
        datadict['weight'] = node_weights[node]

    return G, message

def read_file(file):
    with open(file, 'r') as f:
        data = f.readlines()
    data = [line.replace("Â", " ").strip().split() for line in data]
    return data

def ILPtsp_solve(sloc, stas, G, donttouch=set()):
    
    # flo
    pcessors, alldists = nx.floyd_warshall_predecessor_and_distance(G)

    ta_cycle = gurobiTspCycle(sloc, stas, alldists)
    

    drop_cycle = optCycle(ta_cycle, alldists, donttouch)
    

    listdropoffs = reconDropoffs(sloc, stas, ta_cycle, drop_cycle)
    
    if (len(listdropoffs) > 1):
        drop_cycle = gurobiTspCycle(sloc, set(listdropoffs.keys()), alldists)
        
    # Reconstructs dropoff cycle

    listlocs = reconLocs(drop_cycle, pcessors)

    return listlocs, listdropoffs


"""
Here's a high-level overview of the gurobiTspCycle function:

1. Create a Gurobi model m.
2. Define the set of nodes as the union of stas and sloc.
3. Create binary variables for each edge between nodes, with objective coefficients equal to the distances between nodes.
4. Add degree-2 constraints for each node, ensuring that each node has one incoming and one outgoing edge.
5. Define the subtour elimination callback function cbSubtourElim. This function checks the current solution for subtours and adds lazy constraints to eliminate them.
6. Set the model's LazyConstraints parameter to 1 and optimize the model using the cbSubtourElim callback.
7. Extract the selected edges from the optimized model.
8.Find the single cycle in the selected edges using the getCycles function.
9. Rearrange the cycle so that it starts with sloc and ends with sloc.
10. Return the final cycle as the TSP solution.

"""
def gurobiTspCycle(sloc, stas, alldists):
    m = Model()

    nodes = set(stas).union({sloc})

    # Add variable for each edge
    e = {}
    for i in nodes:
        for j in nodes:
            e[i,j] = m.addVar(obj=alldists[i][j], vtype=GRB.BINARY, name="e{}_{}".format(i, j))
    m.update()

    # Add degree 2 constraint for all vertices
    for i in nodes:
        m.addConstr(quicksum(e[i,j] for j in nodes) == 1)
        m.addConstr(quicksum(e[j,i] for j in nodes) == 1)
        e[i,i].ub = 0
    m.update()

    # Add lazy constraint
    
    # Define the subtour elimination callback function:
    #a. Check if the current solution contains multiple cycles.
    #b. If subtours are found, add constraints to break the subtours.
    def cbSubtourElim(model, where):
        if where == GRB.callback.MIPSOL:
            ed = [] # make list of selected edges
            for i in nodes:
                for j in nodes:
                    sol = model.cbGetSolution(e[i,j]);
                    if sol > 0.5:
                        ed += [(i,j)]
            cycles = getCycles(ed)
            if len(cycles) > 1: # if more than 1 cycle, then we have a subtour
                for c in cycles:
                    expr = 0 # all edges in the subgraph must total <= |S|-1
                    for i in c:
                        for j in c:
                            expr += e[i,j]
                    model.cbLazy(expr <= len(c) - 1)
    
    # Define the subtour elimination callback function:
    #a. Check if the current solution contains multiple cycles.
    #b. If subtours are found, add constraints to break the subtours.
    
    def getCycles(edges):
        visited = {i[0]:False for i in edges}
        nexts = {i[0]:i[1] for i in edges}
        cycles = []

        while True:
            curr = -1
            for i in visited:
                if visited[i] == False:
                    curr = i
                    break
            if curr == -1:
                break
            thiscycle = []
            while not visited[curr]:
                visited[curr] = True
                thiscycle.append(curr)
                curr = nexts[curr]
            cycles.append(thiscycle)

        return cycles

    m.params.LazyConstraints = 1
    m.optimize(cbSubtourElim)

    # Get travel cycle Identify the single cycle in the final solution.
    ed = [] # get selected edges
    edec = m.getAttr('x', e)
    for i in nodes:
        for j in nodes:
            if edec[i,j] > 0.5:
                ed.append((i,j))
    cycles = getCycles(ed) # find the one cycle
    
    # Rearrange the final cycle to start with the desired starting node and return the resulting tour.
    if cycles:
        c = cycles[0] # rearrange so sloc first
        indstart = c.index(sloc)
        jumpcycle = c[indstart:] + c[:indstart] + [sloc]
    else:
        jumpcycle = [sloc]

    return jumpcycle



def optCycle(ta_cycle, alldists, donttouch):
    
    drop_cycle = ta_cycle.copy() # Find best location to dropoff corresponding TA in ta_cycle, initially just the TA's home
    
    for _ in range(len(drop_cycle)): # Doing this enough times hopefully does a good job
        for i in range(1, len(drop_cycle) - 1): # Check for better dropoff location for each TA
            if ta_cycle[i] not in donttouch:
                min_j, min_cost = -1, float('inf')
                for j in alldists[ta_cycle[i]]: # Check all other possible dropoff points, literally all nodes
                    ncost = 2/3*alldists[j][drop_cycle[i-1]] + alldists[j][ta_cycle[i]] + 2/3*alldists[j][drop_cycle[i+1]] 
                    # Cost to drive to new node (from previous in cycle), dropoff, and drive to next node
                    if ncost < min_cost:
                        min_j, min_cost = j, ncost
                drop_cycle[i] = min_j
    return drop_cycle


def reconDropoffs(sloc, stas, ta_cycle, drop_cycle):
    listdropoffs = {}
    for i in range(1, len(ta_cycle)-1):
        if drop_cycle[i] in listdropoffs:
            listdropoffs[drop_cycle[i]].append(ta_cycle[i])
        else:
            listdropoffs[drop_cycle[i]] = [ta_cycle[i]]
    # Edge case: TA home = starting location
    if sloc in stas:
        if sloc in listdropoffs:
            listdropoffs[sloc].append(sloc)
        else:
            listdropoffs[sloc] = [sloc]
    return listdropoffs

# Reconstructs dropoff cycle

def reconLocs(drop_cycle, pcessors):
    listlocs = []
    curr = -1
    for i in range(len(drop_cycle)-1):
        curr = drop_cycle[i]
        while curr != drop_cycle[i+1]:
            listlocs.append(curr)
            curr = pcessors[drop_cycle[i+1]][curr]
    listlocs.append(curr)
    return listlocs

def calculate_total_costs(infile, outfile): # driving, walking, total

    #print("infile",infile)
    #print("outfile",outfile)
    indat = read_file(infile)
    outdat = read_file(outfile)
    cost, msg = ov.tests(indat, outdat)
    return [float(s.split(" ")[-1][:-1]) for s in msg.split('\n')[:-1]]


def plotGraph(G):
    pos = nx.spring_layout(G)
    labels = nx.get_edge_attributes(G, 'weight')

    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines')

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])

    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=False,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=20,  # Increased size of the node markers
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=2)))

    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x])  # Changed the way 'x' and 'y' coordinates are being appended
        node_trace['y'] += tuple([y])

    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(f'Node {node}<br># of connections: {len(adjacencies[1])}')

    node_trace['marker']['color'] = node_adjacencies
    node_trace['text'] = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='<br>Network graph made with Plotly',
                        titlefont=dict(size=16),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            text="",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002)],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

    plotly.offline.plot(fig, filename='C:/gurobi1001/win64/examples/PlotGraph.html')
    fig.show()

def solve(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix, params=[]):

    # Create graph

    G, _ = adjacency_matrix_to_graph(adjacency_matrix)
    #print("Line 215, plotGraph commented.")
    plotGraph(G)
    # Convert locations to indices
    allowed_drop_points = set([list_of_locations.index(h) for h in list_of_homes])
    starting_loc = list_of_locations.index(starting_car_location)

    articulation_map = get_articulation_map(starting_loc, allowed_drop_points, G)
    

    listlocs, listdropoffs = tree_solve(starting_loc, allowed_drop_points, G, articulation_map)

    


    return listlocs, listdropoffs

def get_articulation_map(starting_loc, allowed_drop_points, G):
    
    # Find biconnected components and get in nice form
    bccs = list(nx.biconnected_components(G))
    articulation_points = set(nx.articulation_points(G))
    # articulation point -> (biconnected component, entire subgraph involving bcc, articulation_points in that bcc)
    
    # working of the line below:

    articulation_map = {i:[(b, set(), b.intersection(articulation_points) - {i}) for b in bccs if i in b] for i in articulation_points}


    
    if starting_loc not in articulation_map:
        
        cc = [i for i in bccs if starting_loc in i][0] # Find the biconnected component starting_loc is part of
        articulation_map[starting_loc] = [(cc, set(), cc.intersection(articulation_points))]

    # gets rid of parent in each set using BFS
    parents = {starting_loc}
    while parents:
        newparents = set()
        for p in parents:
            childs = set() # get all the children
            for t in articulation_map[p]:
                childs.update(t[2])
            for c in childs:
                articulation_map[c] = [i for i in articulation_map[c] if p not in i[0]] # remove the parent from all of them
                for t in articulation_map[c]:
                    t[2].difference_update({p})
            newparents.update(childs)
        parents = newparents

    generateSubgraphsDFS(articulation_map, starting_loc) # avoid too much recomputation and also pretty fun

    return articulation_map

def generateSubgraphsDFS(articulation_map, starting_loc):
    for bcc, subg, artps in articulation_map[starting_loc]:
        subg.update(bcc)
        for c in artps:
            generateSubgraphsDFS(articulation_map, c)
            for _, c_subg, _ in articulation_map[c]:
                subg.update(c_subg)

def tree_solve(starting_loc, allowed_drop_points, G, articulation_map):

    list_locations, list_dropoffs = [], []
    
    #print("articulation_map[starting_loc]",articulation_map[starting_loc])
    for bccset, subgset, articulation_points in articulation_map[starting_loc]:
        
        
        subg_tas = allowed_drop_points.intersection(subgset)
        
        if len(subg_tas) == 0:
            
            list_locations.append([starting_loc])
            
            list_dropoffs.append({})
            
        elif len(subg_tas) == 1 or (len(subg_tas) == 2 and starting_loc in subg_tas):
            
            list_locations.append([starting_loc])
            
            list_dropoffs.append({starting_loc:subg_tas})
            
        else:
            

            biconnected_graph = G.subgraph(bccset).copy()
            
            bccorigtas = allowed_drop_points.intersection(bccset) # so we don't dropoff fake tas
            
            bcctas = bccorigtas.copy()
            
            bcctasmap = {} # for converting fake ta back to orig
            
            bccforceta = set()
            biconnected_insert = {} # Stores recursive calls to stitch in

            # Convert articulation_points to homes and perform recursive calls if necessary
            for ap in articulation_points:
                ap_subg = reduce(lambda a,b: a.union(b), [i[1] for i in articulation_map[ap]])
                ap_tas = allowed_drop_points.intersection(ap_subg)
                
                if len(ap_tas) == 1:
                    bcctas.add(ap)
                    bcctasmap[ap] = ap_tas.pop()
                    
                elif len(ap_tas) >= 2: # if more than 2 tas then always optimal to enter subgraph
                    bcctas.add(ap)
                    bccforceta.add(ap)
                    biconnected_insert[ap] = tree_solve(ap, ap_tas, G.subgraph(ap_subg).copy(), articulation_map)

            
            ILPtsp_locs, ILPtsp_dropoffs = ILPtsp_solve(starting_loc, bcctas, biconnected_graph, donttouch=bccforceta)
            
            
            # Stitch everything together
            biconnected_locs = []
            bcc_inserted = []
            for i in ILPtsp_locs:
                if i in biconnected_insert and i not in bcc_inserted:
                    biconnected_locs.extend(biconnected_insert[i][0])
                    bcc_inserted.append(i) # don't double insert
                else:
                    biconnected_locs.append(i)
            biconnected_dropoffs = {}
            for i in ILPtsp_dropoffs:
                new_set = {bcctasmap[t] if t in bcctasmap else t for t in ILPtsp_dropoffs[i]}.intersection(allowed_drop_points)
                if len(new_set) > 0: # possible for new_set to be empty
                    biconnected_dropoffs[i] = new_set
            for i in biconnected_insert:
                for dppt in biconnected_insert[i][1]:
                    if dppt in biconnected_dropoffs:
                        biconnected_dropoffs[dppt].update(set(biconnected_insert[i][1][dppt]))
                    else:
                        biconnected_dropoffs[dppt] = set(biconnected_insert[i][1][dppt])

            # Add to our running lists
            list_locations.append(biconnected_locs)
            list_dropoffs.append(biconnected_dropoffs)

    # Reconstruct listlocs and listdropoffs
    listlocs = list_locations[0]
    listdropoffs = list_dropoffs[0]
    for l in list_locations[1:]:
        listlocs.extend(l[1:])
    for dps in list_dropoffs[1:]:
        for d in dps:
            if d in listdropoffs:
                listdropoffs[d].update(dps[d])
            else:
                listdropoffs[d] = dps[d]

    # Convert all dropoff sets to lists
    for d in listdropoffs:
        listdropoffs[d] = list(listdropoffs[d])

    return listlocs, listdropoffs
                
def convertToFile(path, dropoff_mapping, path_to_file, list_locs):
    string = ''
    for node in path:
        string += list_locs[node] + ' '
    string = string.strip()
    string += '\n'

    dropoffNumber = len(dropoff_mapping.keys())
    string += str(dropoffNumber) + '\n'
    for dropoff in dropoff_mapping.keys():
        strDrop = list_locs[dropoff] + ' '
        for node in dropoff_mapping[dropoff]:
            strDrop += list_locs[node] + ' '
        strDrop = strDrop.strip()
        strDrop += '\n'
        string += strDrop
    write_to_file(path_to_file, string)

def solve_from_file(input_file, output_directory, params=[]):
    #print('Processing {}...'.format(input_file), end="")
    sys.stdout.flush()

    input_data = read_file(input_file)
    num_of_locations, num_houses, list_locations, list_houses, starting_car_location, adjacency_matrix = data_parser(input_data)
    car_path, drop_offs = solve(list_locations, list_houses, starting_car_location, adjacency_matrix, params=params)

    basename, filename = os.path.split(input_file)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_file = input_to_output(input_file, output_directory)

    convertToFile(car_path, drop_offs, output_file, list_locations)

def get_files_with_extension(directory, extension):
    files = []
    for name in os.listdir(directory):
        if name.endswith(extension):
            files.append(f'{directory}/{name}')
    return files

def read_file(file):
    with open(file, 'r') as f:
        data = f.readlines()
    data = [line.replace("Â", " ").strip().split() for line in data]
    return data


def write_to_file(file, string, append=False):
    if append:
        mode = 'a'
    else:
        mode = 'w'
    with open(file, mode) as f:
        f.write(string)

def write_data_to_file(file, data, separator, append=False):
    if append:
        mode = 'a'
    else:
        mode = 'w'
    with open(file, mode) as f:
        for item in data:
            f.write(f'{item}{separator}')

def input_to_output(input_file, output_directory):
    return (
        os.path.join(output_directory, os.path.basename(input_file))
        .replace("input", "output")
        .replace(".in", ".out")
    )


def solve_all(input_directory, output_directory, params=[]):
    input_files = get_files_with_extension(input_directory, 'in')
    

    for input_file in input_files:
        
        solve_from_file(input_file, output_directory, params=params)


if __name__=="__main__":
    
    class Args:
        def __init__(self, all=False, input=None, output_directory='.', params=None):
            self.all = all
            self.input = input
            self.output_directory = output_directory
            self.params = params
    
    args = Args(all=True, input="1_input_practiceInsideFolder", output_directory="1_output_practiceInsideFolder", params=None)
    #args = parser.parse_args()
    #("args.all",args.all)
    output_directory = args.output_directory
    #print("args.all",args.all)
    if args.all:
        
        input_directory = args.input
        f = glob.glob(args.input + "/*.in")
        input_file = f[0]
        solve_all(input_directory, output_directory, params=args.params)
    else:
        f = glob.glob(args.input + "/*.in")
        input_file = f[0]
        solve_from_file(input_file, output_directory, params=args.params)
        
        
    outputs = "1_output_practiceInsideFolder"
    outfile = os.path.splitext(os.path.basename(input_file))[0] + ".out"
    oldfile = outputs + "/" + outfile
    
    _, _, t = calculate_total_costs(input_file,  oldfile)

    ot = calculate_total_costs(input_file, oldfile)[2] if os.path.exists(oldfile) else float('inf')
    print("total_cost", ot)



        


