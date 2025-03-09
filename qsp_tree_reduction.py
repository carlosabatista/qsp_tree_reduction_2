#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 15:37:51 2025

@author: linho
"""

import numpy as np
import random
from my_qclib_utils import *
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile

from qclib.util import get_state
from datetime import datetime
import networkx as nx

from qclib.state_preparation import CvoqramInitialize

def printvector4f(vector):
    for e in vector:
        print(np.round(e,4), " ", end =" ")
    print()
    
def printvectornf(vector,n):
    for e in vector:
        print(np.round(e,n), " ", end =" ")
    print()

def printvector4fn(vector,n):
    i = 0
    for e in vector:
        print('{:.4f}'.format(np.round(e,4)), "\t", end =" ")
        if (i+1)%n == 0: print()
        i = i + 1
    print()

def generate_random_state_n_density(num_qubits, density):
    """
    Parameters
    ----------
    num_qubits : int
        DESCRIPTION. number of qubits of the state to be created.
    density : float, , 0 <= density <= 1.
        DESCRIPTION: density of non-zero amplitudes.

    Returns
    -------
    v : TYPE: array os complex values.
        DESCRIPTION: a state of num_qubits qubits, 2**num_qubits*density of which are non-zero.

    """
    v = np.random.rand(2**num_qubits)+np.random.rand(2**num_qubits)*1j
    expulsar = int(len(v)*(1-density))
    indices = random.sample(range(len(v)), expulsar)
    for i in indices:
        v[i]=0
    norma_v = np.linalg.norm(v)
    if norma_v:
        v=v/norma_v
    return v

def generate_random_state_n_m(num_qubits, m): # m is the number of non-zero amplitudes
    """
    Parameters
    ----------
    num_qubits : int
        DESCRIPTION. number of qubits of the state to be created.
    m : int,  0 <= m <= 2**num_qubits.
        DESCRIPTION: number of non-zero amplitudes.

    Returns
    -------
    v : TYPE: array os complex values.
        DESCRIPTION: a state of num_qubits qubits, m of which are non-zero.

    """
    
    v = np.random.rand(2**num_qubits)+np.random.rand(2**num_qubits)*1j
    expulsar = len(v)-m #int(len(v)*(1-density))
    indices = random.sample(range(len(v)), expulsar)
    for i in indices:
        v[i]=0
    norma_v = np.linalg.norm(v)
    if norma_v:
        v=v/norma_v
    return v

def get_state_tree(state, Num_qubits):
  global state_tree
  data = [Amplitude(i, a) for i, a in enumerate(state)]
  state_tree = state_decomposition(Num_qubits, data)
  return state_tree

def create_angles_tree_with_dont_cares(state_tree):
  """
  Modification of the create_angles_tree routine of qclib to mark nodes with don't cares,
  represented here by the value -9.
  
  :param state_tree: state_tree is an output of state_decomposition function
  :return: tree with angles that will be used to produce the state preparation
  """
  
  dc = -9

  arg = state_tree.right.arg - state_tree.arg
  
  if state_tree.mag != 0.0:
    mag = state_tree.right.mag / state_tree.mag
    angle_z = 2 * arg
  else:
    angle_y = dc
    angle_z = dc
    mag = dc
    

  # Avoid out-of-domain value due to numerical error.

  if mag == dc:
    angle_y = mag
    angle_z = mag
  elif mag > dc and mag < -1.0:
    angle_y = -math.pi
  elif mag > 1.0:
    angle_y = math.pi
  else:
    angle_y = 2 * math.asin(mag)

  node = NodeAngleTree(state_tree.index, state_tree.level, angle_y, angle_z, None, None)

  if not is_leaf(state_tree.left):
    node.right = create_angles_tree_with_dont_cares(state_tree.right)
    node.left = create_angles_tree_with_dont_cares(state_tree.left)
  return node

def tree_to_graph_weighted_edges(root, axis):
  """

  :param root: angles tree
  :param axis: axis rotation
  :return: directed graph with weights on the edges
           indicating whether the edge is a 0/1-child
  """
  graph = nx.DiGraph()

  def traverse(node, parent=None):
    if node:
      g_node_index = 2 ** node.level + node.index
      nodeangle = 0
      if axis == 'y':
        nodeangle = node.angle_y
      if axis == 'z':
        nodeangle = node.angle_z
      g_node_weight = nodeangle
      graph.add_node(g_node_index, weight=g_node_weight, level=node.level)
      if parent:
        parent = (2 ** node.level + node.index) // 2
        edge_weight = int(((-1) ** (g_node_index + 1)+1)/2)
        graph.add_edge(parent, 2 ** node.level + node.index, weight=edge_weight)
      traverse(node.left, node)
      traverse(node.right, node)

  traverse(root)
  return graph

def display_graph(G):
  pos = nx.nx_pydot.graphviz_layout(G, prog="dot")
  labels1 = nx.get_node_attributes(G, 'weight')
  labels2 = nx.get_node_attributes(G, 'level')

  nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=8, font_weight='bold')
  nx.draw_networkx_labels(G, pos, labels={node: f"q{labels2[node]}     \n\n{np.round(labels1[node], 4)}" for node in labels1}, font_size=6)

  edge_labels = nx.get_edge_attributes(G, 'weight')
  nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

  plt.show()

def get_graph_depth(graph, no_inicial):
  profundidade = 0
  fila = [(no_inicial, 0)]
  visitados = set()

  while fila:
    no_atual, nivel_atual = fila.pop(0)
    if no_atual not in visitados:
      visitados.add(no_atual)
      profundidade = max(profundidade, nivel_atual)
      for vizinho in graph[no_atual]:
        fila.append((vizinho, nivel_atual + 1))

  return profundidade

def remove_subgraph(graph, subgraph):
  sorted_list_graph = list(graph.nodes())
  sorted_list_graph.sort()
  if subgraph in sorted_list_graph:
    raio = get_graph_depth(graph, subgraph)
    sub = list(nx.ego_graph(graph, subgraph, radius=raio))
    sub.sort(reverse=True)
    for node in sub:
      graph.remove_node(node)

def prune_graph(graph, node_1, node_2):
        
    """
    in: a graph and a subgraph root node
    out: a graph minus the subgraph
    """
    
    graphcp = graph.copy()
    radius = get_graph_depth(graph, node_2)
    parent = graphcp._pred[node_2]
    if parent:
        parentindex = list(parent)[0]
        grand_parent = graphcp._pred[parentindex]
        remove_subgraph(graphcp, node_2)
        if grand_parent:
            graphcp.add_edge(list(grand_parent)[0], node_1, weight=graphcp[list(grand_parent)[0]][parentindex]['weight'])
        graphcp.remove_edge(parentindex, node_1)  
        
    return graphcp

def reduce_graph(graph):
    """
    Mescla os procedimento MERGE e PRUNE em um único.
    
    Percorre o grafo e testa se nó é don't care. Se não for,
    pergunta se irmão é igual. se for, verifica se a sub-árvore
    irmã é igual, e poda em caso positivo.    
    nós removidos devem ser incluidos ma lista de visitados'
    
    AINDA EM CONSTRUÇÃO
    """
    
    graphcp = graph.copy() 
    
    Err = 1e-7
    
    flag = True
    
    while flag:
    
        
        sorted_list_graph = list(graphcp.nodes())
        sorted_list_graph.sort()
        
        only_parents = []
        
        for item in sorted_list_graph:
            if node_have_children(graphcp, item):
                only_parents.append(item)
        
        #visitados = [todos os removidos]
        visitados = []
        subgraph_to_eliminate = []
        
        for no_pai in only_parents:#sorted_list_graph:
            if no_pai not in visitados and node_have_children(graphcp, no_pai):
                if graphcp.nodes[no_pai]['weight'] == -9:
                    grandparentdc = graphcp._pred[no_pai]
                    listpais = list(graphcp.successors(no_pai))
                    for uncle in listpais:
                        if uncle != no_pai:
                            if graphcp.nodes[no_pai]['level'] == graphcp.nodes[uncle]['level']:
                                if graphcp[grandparentdc][np_pai]['weight'] != graphcp[grandparentdc][uncle]['weight']:
                    
                                    raio_str = get_graph_depth(graphcp, uncle)
                                    subgraph_to_remove = nx.ego_graph(graphcp, uncle, radius=raio_str)
                                    for vis in list(subgraph_to_remove): visitados.append(vis) 
                    
                                    graphcp = prune_graph(graphcp, no_pai, uncle)
                                    break
                    
                
                list_childs = list(graphcp.successors(no_pai))
                list_childs.sort()
                for indexnode, no_1 in enumerate(list_childs[:-1]):
                  for no_2 in list_childs[indexnode + 1:]:
                    if graphcp.nodes[no_1]['level'] == graphcp.nodes[no_2]['level']:
                      #if abs(graph.nodes[no_1]['weight'] - graph.nodes[no_2]['weight'])<=Err:
                      if np.isclose(graphcp.nodes[no_1]['weight'], graphcp.nodes[no_2]['weight']) or graphcp.nodes[no_2]['weight']==-9:
                          raio1 = get_graph_depth(graphcp, no_1)
                          raio2 = get_graph_depth(graphcp, no_2)
                          sub_1 = nx.ego_graph(graphcp, no_1, radius=raio1)
                          sub_2 = nx.ego_graph(graphcp, no_2, radius=raio2)
                          if check_node_weights_equal(graphcp, sub_1, sub_2):
                              for vis in list(sub_2): visitados.append(vis) 
                              subgraph_to_eliminate.append(no_2)
                              
                              graphcp = prune_graph(graphcp, no_1, no_2)
                              
          
                          #return graph
        
            if len(subgraph_to_eliminate) > 0:
              flag = True
            else:
              flag = False

    return graph

def prune_graph_dontcares_one(graph, valor):
  # Encontrar o nó com o valor especificado para don't care e substituir
  # a sub-árvore a partir desse nó pela sub-árvore irmã

  graphy = graph.copy()

  sorted_list_graph = list(graphy.nodes())
  sorted_list_graph.sort()

  list_dontcares = []

  for no in sorted_list_graph:
    if graphy.nodes[no]['weight'] == valor:
      list_dontcares.append(no)

  list_dontcares.sort()

  dc_visitados = []

  if len(list_dontcares) > 0:
    for no_dc in list_dontcares:
      if no_dc not in dc_visitados:
        raio = get_graph_depth(graphy, no_dc)
        subgrafo_dc_a_eliminar_Y = nx.ego_graph(graphy, no_dc, radius=raio)

        parent_dc = graphy._pred[no_dc]
        if parent_dc:
          children = list(graphy.successors(list(parent_dc)[0]))
          children.remove(no_dc)
          brother = children[0]

          pai = list(parent_dc)[0]
          grand_parent = graphy._pred[pai]

          remove_subgraph(graphy, no_dc)

          for n in list(subgrafo_dc_a_eliminar_Y):
            # list_dontcares.remove(n)
            dc_visitados.append(n)

          if grand_parent:
            graphy.add_edge(list(grand_parent)[0], brother, weight=graphy[list(grand_parent)[0]][pai]['weight'])

          graphy.remove_edge(pai, brother)

  return graphy

def prune_graph_dontcares(graph):
  # Encontrar o nó com o valor especificado para don't care e substituir
  # a sub-árvore a partir desse nó pela sub-árvore irmã

  valor = -9
  graphy = graph.copy()

  sorted_list_graph = list(graphy.nodes())
  sorted_list_graph.sort()

  list_dontcares = []

  for no in sorted_list_graph:
    if graphy.nodes[no]['weight'] == valor:
      list_dontcares.append(no)

  list_dontcares.sort()

  dc_visitados = []

  if len(list_dontcares) > 0:
    for no_dc in list_dontcares:
      if no_dc not in dc_visitados:
        raio = get_graph_depth(graphy, no_dc)
        subgrafo_dc_a_eliminar_Y = nx.ego_graph(graphy, no_dc, radius=raio)

        parent_dc = graphy._pred[no_dc]
        if parent_dc:
          children = list(graphy.successors(list(parent_dc)[0]))
          children.remove(no_dc)
          brother = children[0]

          pai = list(parent_dc)[0]
          grand_parent = graphy._pred[pai]

          remove_subgraph(graphy, no_dc)

          for n in list(subgrafo_dc_a_eliminar_Y):
            # list_dontcares.remove(n)
            dc_visitados.append(n)

          if grand_parent:
            graphy.add_edge(list(grand_parent)[0], brother, weight=graphy[list(grand_parent)[0]][pai]['weight'])

          graphy.remove_edge(pai, brother)

  return graphy

def remove_subtrees_zero(graph):
    Err = 1e-6
    again = True
    while again:
        again = False
        graph = graph.copy()
        sorted_list_graph = list(graph.nodes())
        sorted_list_graph.sort()
        for node in sorted_list_graph:
            if abs(graph.nodes[node]["weight"]) <= Err:
                raio = get_graph_depth(graph, node)
                sub = nx.ego_graph(graph, node, radius=raio)
                listsub = list(sub)
                onlyzeros = True
                for no in listsub:
                    if abs(graph.nodes[no]["weight"]) > Err:
                        onlyzeros = False
                if onlyzeros:
                    remove_subgraph(graph, node)
    return graph

def remove_first_zero_node(graph):

  graph = graph.copy()

  list_nodes = list(graph)
  list_nodes.sort()
  
  dead_nodes = []

  for node in list_nodes:
      if node not in dead_nodes:
        if not (graph._pred[node]):
    
          # sorted_list_graph = list(graph.nodes())
          # sorted_list_graph.sort()
    
          # list_nodes = list(graph.nodes())
          # se no nao tem pai e tem valor zero
          #  se aresta filho = 1, elimina toda o subgrafo
          #  se -1, elimina apenas o nó de valor zero
          # if 1 in list_nodes:
          if graph.nodes[node]['weight'] == 0:
            dead_nodes.append(node)
            lista_filhos = list(graph.successors(node))
            lista_filhos.sort()
            #node_to_dead = node
    
            
    
            for subgraph in lista_filhos:
              if graph[node][subgraph]['weight'] == 1:
                #sorted_list_graph = list(graph.nodes())
                #if subgraph in sorted_list_graph:
                raio = get_graph_depth(graph, subgraph)
                sub_dead = nx.ego_graph(graph, subgraph, radius=raio)
                for no in sub_dead:
                  #graph.remove_node(no)
                  dead_nodes.append(no)
                    
            
  #remanescentgraph = list(graph)
  for node in dead_nodes: 
      if node in graph:
          graph.remove_node(node)
  #graph.remove_node(node_to_dead)
  return graph

def check_node_weights_equal(graph, subgraph1, subgraph2):
  """
  Verifica se os pesos dos nós em um subgraph são iguais aos pesos dos nós
  no subgraph irmão, respectivamente, para cada nó.

  As entradas são:
    graph: O graph principal.
    subgraph1: O primeiro subgraph.
    subgraph2: O segundo subgraph (irmão).

  e retorna True se os pesos dos nós forem iguais, False caso contrário.
  """

  Err = 1e-6

  if len(subgraph1.nodes()) != len(subgraph2.nodes()):
    return False

  list_subgraph1 = list(subgraph1.nodes)
  list_subgraph1.sort()
  list_subgraph2 = list(subgraph2.nodes)
  list_subgraph2.sort()

  node_map = {node: i for i, node in enumerate(list_subgraph1)}
  for node in list_subgraph1:
    if 'weight' not in graph.nodes[node] or 'weight' not in graph.nodes[list_subgraph2[node_map[node]]]:
      return False
    #if graph.nodes[node]['weight'] != graph.nodes[list_subgraph2[node_map[node]]]['weight']:
    if abs(graph.nodes[node]['weight'] - graph.nodes[list_subgraph2[node_map[node]]]['weight'])>Err:
      return False

  return True

def merge_subgraphs(graph):
  graph = graph.copy()

  #Err = 0.0001

  if len(list(graph.nodes())) <= 1:
    return graph, False

  sorted_list_graph = list(graph.nodes())
  sorted_list_graph.sort()

  subgraph_to_eliminate = []
  # lembrar de atualizar a lista de childs
  # varrer os nós e retirar os excluidos da lista

  for no_pai in sorted_list_graph:
    list_childs = list(graph.successors(no_pai))
    list_childs.sort()
    for indexnode, no_1 in enumerate(list_childs[:-1]):
      for no_2 in list_childs[indexnode + 1:]:
        if graph.nodes[no_1]['level'] == graph.nodes[no_2]['level']:
          #if abs(graph.nodes[no_1]['weight'] - graph.nodes[no_2]['weight'])<=Err:
          if np.isclose(graph.nodes[no_1]['weight'],graph.nodes[no_2]['weight']):
            raio1 = get_graph_depth(graph, no_1)
            raio2 = get_graph_depth(graph, no_2)
            sub_1 = nx.ego_graph(graph, no_1, radius=raio1)
            sub_2 = nx.ego_graph(graph, no_2, radius=raio2)
            if check_node_weights_equal(graph, sub_1, sub_2):
              subgraph_to_eliminate.append(no_2)

              graph.remove_edge(no_pai, no_1)
              grandparent = graph._pred[no_pai]
              if grandparent:
                graph.add_edge(list(grandparent)[0], no_1, weight=graph[list(grandparent)[0]][no_pai]['weight'])
              remove_subgraph(graph, no_2)
              return graph, True

  if len(subgraph_to_eliminate) > 0:
    flag = True
  else:
    flag = False

  return graph, flag

def merge_equal_sibling_subtrees(graph):
    graphs_merged = True
    while graphs_merged:
        graph, graphs_merged = merge_subgraphs(graph)
    return graph

def dontcare_condition(state):
    Err = 1e-6
    dc_condition = False
    for i in range(int(len(state)/2)):
        if abs(state[2*i])<=Err and abs(state[2*i+1])<=Err:
            dc_condition = True
            break
    return dc_condition

def find_paths(graph):
  node_list = list(graph)
  node_list.sort()
  list_paths = []
  if node_list:
    list_paths.append([node_list[0]])
    for i, no_1 in enumerate(node_list):
      parent = list(graph.predecessors(no_1))
      if not parent:
        if not [no_1] in list_paths: 
          list_paths.append([no_1])
        for no_2 in node_list[1:]:  
          if no_1 != no_2:
            if nx.has_path(graph, no_1, no_2):
              for path in nx.all_simple_paths(graph, no_1, no_2):
                list_paths.append(list(path))
  #ordenar pelo niǘel do ultimo elemento
  return list_paths

def sort_paths(paths):
  # Ordena a lista primeiro pelo tamanho dos subelementos e depois pelos valores dos subelementos
  paths_ordenado = sorted(paths, key=lambda x: (len(x), x[-1]))
  # def custom_sort(lst):
  # Ordena primeiro pelo valor do último subelemento e depois pelo tamanho do subelemento
  return sorted(paths, key=lambda x: (x[-1], len(x)))


def PathEdgesWeights(graph, path):
    path = list(path)
    lenPath = len(path)
    pew = []
    if lenPath>1:
        for i in range(lenPath-1):
            pew.append(graph[path[i]][path[i+1]]['weight'])
    return pew

def vetor_para_string_binaria(vetor):
    # Verifica se todos os elementos do vetor são 0 ou 1
    if all(elemento in [0, 1] for elemento in vetor):
        # Converte cada elemento do vetor para string e junta todos em uma única string
        string_binaria = ''.join(map(str, vetor))
        return string_binaria
    else:
        raise ValueError("O vetor deve conter apenas zeros e uns.")

def PathEdgesWeightsToControl(graph, path):
    path = list(path)
    lenPath = len(path)
    pew = []
    if lenPath>0:#1:
        for i in range(lenPath-1):
            pew.append(graph[path[i]][path[i+1]]['weight'])
        pewtobin = "'"+vetor_para_string_binaria(pew)+"'"
        print(pew, pewtobin)
    return pewtobin

def graph_to_multicontrolled_circuit_(graph, r_axis, num_qubits):
    Err = 1e-6
    qc = QuantumCircuit(num_qubits)
    paths = find_paths(graph)
    paths = sort_paths(paths)

    for path in paths:
        PEW = PathEdgesWeightsToControl(graph, path)
        print("pew:",PEW)

        ctrls = path[:-1]
        ctargets = path[1:]  # para observar o tipo de aresta entre um nó (target) e o anterior (ctarget), control
        qubits = []
        for p in path:
            qubits.append(graph.nodes[p]['level'])


        print("ctrls:", ctrls)
        print("qubits:", qubits)
        print(qc)

        if len(path) == 1:
            qubit_alvo = graph.nodes[path[0]]['level']
            angle = graph.nodes[path[0]]['weight']
            if abs(angle) > Err:
                if r_axis == 'y':
                    qc.ry(angle, qubit_alvo)
                if r_axis == 'z':
                    qc.rz(angle, qubit_alvo)
        else:
            angle = graph.nodes[path[-1]]['weight']
            if abs(angle) >= Err:
                num_ctrls = len(ctrls)
                if r_axis == 'y':
                    u = RYGate(angle).control(num_ctrls, None, PEW)
                if r_axis == 'z':
                    u = RZGate(angle).control(num_ctrls, None, PEW)
                qc.append(u, qubits)


    return qc

def CircuitParams(graph):
    """
    param: graph
    return:
     - All Paths on graph
     - All Edges to each path
     - All Nodes angles
     - All node levels
    """
    AllPaths = find_paths(graph)
    AllPaths = sort_paths(AllPaths)#*********************

    AllEdgesWeights = []
    AllNodesLevels = []
    AllAngles = []
    for path in AllPaths:

        AllEdgesWeights.append(PathEdgesWeights(graph, path))

        pathnodesLVL = []

        for node in path:
            pathnodesLVL.append(graph.nodes[node]['level'])
        AllNodesLevels.append(pathnodesLVL)

        #AllAngles = NodesWeight(graph)
        AllAngles.append(graph.nodes[path[-1]]['weight'])
    return AllPaths, AllEdgesWeights, AllAngles, AllNodesLevels

def graph_to_multicontrolled_circuit(graph, r_axis, Num_qubits):
    """
    Converts a graph to a quantum circuit using multicontrolled gates
    Input: graph
    Process:
    - get all paths, edge weights, angles and node levels from a graph
    - forms groups with all paths that have the same levels
    - each group corresponds to a multiplexer
    - build each multicontrolled gate and append to the quantum circuit
    :return: a quantum circuit
    """
    Err = 1.0e-07
    AllPaths, AllEdgesWeights, AllAngles, AllNodesLVLs = CircuitParams(graph)

    qc = QuantumCircuit(Num_qubits)
    for item in range(len(AllAngles)):
        lvls = AllNodesLVLs[item]
        ctrl = lvls[:-1]
        ctrl = ctrl[::-1]
        ctrlbin = vetor_para_string_binaria(AllEdgesWeights[item])
        trgt = [lvls[-1]]
        angle = AllAngles[item]
        if abs(angle)>=Err and angle!=-9:
            if len(ctrl) == 0:
                if r_axis == 'y':
                    qc.ry(angle, trgt[0])
                if r_axis == 'z':
                    qc.rz(angle, trgt[0])
            else:
                if r_axis == 'y':
                    u = RYGate(angle).control(len(ctrl), None, ctrlbin)
                    qc.append(u, ctrl+trgt)
                if r_axis == 'z':
                    u = RZGate(angle).control(len(ctrl), None, ctrlbin)
                    qc.append(u.reverse_ops(), ctrl+trgt)
    return qc

def subtractpaths(pair, AllPaths, AllEdgesWeights, AllAngles, AllNodesLevels):
    """
    Subtract a pair of paths
    """
    Err = 0.00001

    Path1 = AllPaths[pair[0]].copy()
    Edges1 = AllEdgesWeights[pair[0]].copy()
    Weight1 = AllAngles[pair[0]].copy()
    Weight2 = AllAngles[pair[1]].copy()
    Path1LVLs = AllNodesLevels[pair[0]].copy()

    EW1 = AllEdgesWeights[pair[0]].copy()
    EW2 = AllEdgesWeights[pair[1]].copy()

    index = FindEdgeIndextoChange(EW1, EW2)

    Path1.pop(index)
    Edges1.pop(index)
    Path1LVLs.pop(index)

    Weight2 = Weight2-Weight1

    AllPaths[pair[0]] = Path1
    AllEdgesWeights[pair[0]] = Edges1
    AllNodesLevels[pair[0]] = Path1LVLs

    if abs(Weight2) <= Err:
        AllPaths[pair[1]] = [0]
        AllEdgesWeights[pair[1]]=[0]
        AllNodesLevels[pair[1]]=[0]
        AllAngles[pair[1]]=0
    else:
        AllAngles[pair[1]] = Weight2

    return AllPaths, AllEdgesWeights, AllAngles, AllNodesLevels

def boolCheckCompatibleEdgesWeights(EW1,EW2):
    """
    Check if two arrays of edges are compatible with subtract paths
    """
    vE = 0
    for i, j in zip(EW1, EW2):
        if i != j:
            vE = vE + 1
    if vE == 1:
        return True
    else:
        return False

def boolCheckCompalitblePaths(Path1, Path2, NodesLVL1, NodesLVL2, EW1,EW2):
    # Check if two paths are compatible with subtract
    if len(Path1) == len(Path2):
        if Path1[0] == Path2[0]:
            if NodesLVL1 == NodesLVL2:
                if boolCheckCompatibleEdgesWeights(EW1,EW2):
                    return True
                else:
                    return False

def vPathsPairs(AllPaths, AllNodesLVLs, AllEdgesWeights):
    """
    Obtain all the pairs of paths are compatible with subtract
    """
    lenPath = len(AllPaths)
    PathPairs = []
    for v1 in range(lenPath-1):
        for v2 in range(v1+1, lenPath):
            Path1 = AllPaths[v1]
            Path2 = AllPaths[v2]
            if len(Path1) == len(Path2):
                NodesLVL1 = AllNodesLVLs[v1]
                NodesLVL2 = AllNodesLVLs[v2]
                EW1 = AllEdgesWeights[v1]
                EW2 = AllEdgesWeights[v2]

                if  boolCheckCompalitblePaths(Path1, Path2, NodesLVL1, NodesLVL2, EW1,EW2):
                    PathPairs.append((v1,v2))
    return PathPairs

def unify_sets(pares):
    """
    Unifica os conjuntos de números que tem elementos comuns para evitar quebra de multicontrolados.

    Args:
        pares: Uma lista de tuplas, onde cada tupla representa um conjunto de números.

    Returns:
        A lista de tuplas, onde cada tupla representa um conjunto unificado.
    """

    unified_sets = []
    seen_numbers = set()

    for pair in pares:
        unified_set = set(pair)
        merged = False
        for i, existing_set in enumerate(unified_sets):
            if any(number in existing_set for number in unified_set):
                unified_sets[i] = existing_set.union(unified_set)
                merged = True
                break  # Exit the inner loop after merging
        if not merged:
            unified_sets.append(unified_set)

    # Convert sets back to tuples and return
    return [tuple(sorted(s)) for s in unified_sets]

def premuxes(allpaths, pairs):
    """
    :param allpaths:
    :param pairs:
    :return: vector of paths to build the multiplexors
    """

    lenA = len(allpaths)

    eB = []
    for i in pairs:
        for j in i:
            eB.append(j)

    C = []

    for i in range(lenA):
        if i not in eB and i not in C:
            C.append(i)
        else:
            for j in pairs:
                if i in j and j not in C:
                    C.append(j)
    return C

def edgesweightstodecimal(edgesweight):
    dec = 0
    for i in range(len(edgesweight)):
        if edgesweight[i] == 1:
            dec = dec + 2**i
    dec = int(dec)
    return dec

def graph_to_multiplexor(graph, r_axis, Num_qubits):
    """
    Converts a graph to a quantum circuit
    Input: graph
    Process:
    - get all paths, edge weights, angles and node levels from a graph
    - forms groups with all paths that have the same levels
    - each group corresponds to a multiplexer
    - build each multiplexer and append to the quantum circuit
    :return: a quantum circuit
    """

    AllPaths, AllEdgesWeights, AllAngles, AllNodesLVLs = CircuitParams(graph)
    pares = vPathsPairs(AllPaths, AllNodesLVLs, AllEdgesWeights)
    unified = unify_sets(pares)
    premuxe = premuxes(AllPaths, unified)
    allew = []
    for ew in AllEdgesWeights:
        allew.append(edgesweightstodecimal(ew))
    allew = np.array(allew, dtype=int)

    redone = []
    for p in premuxe:
        if isinstance(p, int):
            redone.append([p])
        else:
            redone.append(list(p))

    qc = QuantumCircuit(Num_qubits)
    for item in redone:
        lvls = AllNodesLVLs[item[0]]
        szMux = 2 ** (len(lvls) - 1)
        ctrl = lvls[:-1]
        trgt = [lvls[-1]]
        if len(ctrl) == 0:
            angle = AllAngles[item[-1]]
            if r_axis == 'y':
                qc.ry(angle, trgt)
            if r_axis == 'z':
                qc.rz(angle, trgt)
        else:
            angles = np.zeros([szMux])
            for index in item:
                cleanangle = AllAngles[index]
                if cleanangle == -9:
                    angles[allew[index]] = 0
                else:
                    angles[allew[index]] = AllAngles[index]
            if r_axis == 'y':
                ucry = multiplexor(RYGate, angles)
                qc.append(ucry, trgt + ctrl)
            if r_axis == 'z':
                ucry = multiplexor(RZGate, angles)
                qc.append(ucry.reverse_ops(), trgt + ctrl)
    return qc

def SubtractAllPaths(AllPaths_, AllEdgesWeights_, AllAngles_, AllNodesLevels_):
    """
    Subtract all paths
    """

    num_unique_pairs = True
    while num_unique_pairs:
        pares = vPathsPairs(AllPaths_, AllNodesLevels_, AllEdgesWeights_)
        unique_pairs = remove_duplicates(pares)
        if len(unique_pairs) == 0:
            #print(unique_pairs)
            num_unique_pairs = False
        for pair in unique_pairs:
            AllPaths_, AllEdgesWeights_, AllAngles_, AllNodesLevels_ = subtractpaths(pair, AllPaths_, AllEdgesWeights_, AllAngles_, AllNodesLevels_)
    return AllPaths_, AllEdgesWeights_, AllAngles_, AllNodesLevels_

def node_have_children(graph, no):
  return len(list(graph.successors(no))) > 0

def FindEdgeIndextoChange(EW1, EW2):
    """
    Find the index in edges to apply controls modifications
    """

    for i in range(len(EW1)):
        if EW1[i] != EW2[i]:
            return i
    return -1

def subtract_subgraphs_brothers_down_top_original(graph):
    graph = graph.copy()

    sorted_list_graph = list(graph.nodes())
    sorted_list_graph.sort(reverse=True)

    #remove leafs
    RemoveNodesList  = []
    for node in sorted_list_graph:
        if not(node_have_children(graph, node)):
          RemoveNodesList.append(node)  
    ListParent = list(set(sorted_list_graph)-set(RemoveNodesList))
    ListParent.sort(reverse=True)
    sorted_list_graph = ListParent
    
    for no_pai in sorted_list_graph:
        list_childs = list(graph.successors(no_pai))
        list_childs.sort()
        for indexnode, no_1 in enumerate(list_childs[:-1]):
            for no_2 in list_childs[indexnode + 1:]:
                if graph.nodes[no_1]['level'] == graph.nodes[no_2]['level']: #same level?
                    if graph[no_pai][no_1]['weight'] != graph[no_pai][no_2]['weight']: #different values edges?

                        raio1 = get_graph_depth(graph, no_1)
                        raio2 = get_graph_depth(graph, no_2)
                        sub_1 = nx.ego_graph(graph, no_1, radius=raio1)
                        sub_2 = nx.ego_graph(graph, no_2, radius=raio2)

                        if raio1 == raio2:

                            list_sub_1 = list(sub_1)
                            list_sub_1.sort()

                            list_sub_2 = list(sub_2)
                            list_sub_2.sort()

                            for n1, n2 in zip(list_sub_1, list_sub_2):
                                graph.nodes[n2]['weight'] = graph.nodes[n2]['weight'] - graph.nodes[n1]['weight']
                                grandparent = graph._pred[no_pai]
                                if grandparent:
                                    grandparent_edge = graph[list(grandparent)[0]][no_pai]['weight']
                                    graph.add_edge(list(grandparent)[0], no_1, weight=grandparent_edge)

                                if graph.has_edge(no_pai,no_1):

                                    graph.remove_edge(no_pai, no_1)

    reverse_list = sorted_list_graph
    reverse_list.sort(reverse=True)
    return graph

def subtract_subgraphs_brothers_down_top_10fev(graph):
    graph = graph.copy()

    sorted_list_graph = list(graph.nodes())
    sorted_list_graph.sort(reverse=True)

    #remove leafs
    RemoveNodesList  = []
    for node in sorted_list_graph:
        if not(node_have_children(graph, node)):
          RemoveNodesList.append(node)  
    ListParent = list(set(sorted_list_graph)-set(RemoveNodesList))
    ListParent.sort(reverse=True)
    sorted_list_graph = ListParent

    for no_pai in sorted_list_graph:
        list_childs = list(graph.successors(no_pai))
        list_childs.sort()
        for indexnode, no_1 in enumerate(list_childs[:-1]):
            for no_2 in list_childs[indexnode + 1:]:
                if graph.nodes[no_1]['level'] == graph.nodes[no_2]['level']: #same level?
                    if graph[no_pai][no_1]['weight'] != graph[no_pai][no_2]['weight']: #different values edges?

                        raio1 = get_graph_depth(graph, no_1)
                        raio2 = get_graph_depth(graph, no_2)
                        sub_1 = nx.ego_graph(graph, no_1, radius=raio1)
                        sub_2 = nx.ego_graph(graph, no_2, radius=raio2)

                        if raio1 == raio2:

                            list_sub_1 = list(sub_1)
                            list_sub_1.sort()

                            list_sub_2 = list(sub_2)
                            list_sub_2.sort()
                            
                            subAllPaths1, subAllEdgesWeights1, subAllAngles1, subAllNodesLevels1 = CircuitParams(sub_1)
                            subAllPaths2, subAllEdgesWeights2, subAllAngles2, subAllNodesLevels2 = CircuitParams(sub_2)
                            
                            if subAllNodesLevels1 == subAllNodesLevels2:
                            #if len(list_sub_1) == len(list_sub_1): 

                                for n1, n2 in zip(list_sub_1, list_sub_2):
                                    graph.nodes[n2]['weight'] = graph.nodes[n2]['weight'] - graph.nodes[n1]['weight']
                                    grandparent = graph._pred[no_pai]
                                    if grandparent:
                                        grandparent_edge = graph[list(grandparent)[0]][no_pai]['weight']
                                        graph.add_edge(list(grandparent)[0], no_1, weight=grandparent_edge)
    
                                    if graph.has_edge(no_pai,no_1):
                                        graph.remove_edge(no_pai, no_1)

    reverse_list = sorted_list_graph
    reverse_list.sort(reverse=True)
    return graph

#pra cada subárvore:
#    1 - tem raios iguais?
#    2 - tem comprimentos iguais?
#    3 - levanta os paths de cada uma
#    4 - cada path de um lado tem um correspondente do outro difereindo apenas do peso de uma aresta?

def GraphEdgesWeights(graph):
    alledges = graph.edges()
    alledgesweights = []
    for edge in alledges:
        alledgesweights.append(graph[edge[0]][edge[1]]['weight'])
    return alledgesweights

def subtract_subgraphs_brothers_down_top(graph):
    graph = graph.copy()

    sorted_list_graph = list(graph.nodes())
    sorted_list_graph.sort(reverse=True)

    #remove leafs
    remove_node_list = []
    for node in sorted_list_graph:
        if not (node_have_children(graph, node)):
            remove_node_list.append(node)
    for node in remove_node_list:
        sorted_list_graph.remove(node)

    for no_pai in sorted_list_graph:
        list_childs = list(graph.successors(no_pai))
        list_childs.sort()
        if len(list_childs) > 1:
            for indexnode, no_1 in enumerate(list_childs[:-1]):
                for no_2 in list_childs[indexnode + 1:]:
                    if graph.nodes[no_1]['level'] == graph.nodes[no_2]['level']:  # same level?
                        if graph[no_pai][no_1]['weight'] != graph[no_pai][no_2]['weight']:  # different values edges?

                            raio1 = get_graph_depth(graph, no_1)
                            raio2 = get_graph_depth(graph, no_2)

                            if raio1 == raio2:
                                sub_1 = nx.ego_graph(graph, no_1, radius=raio1)
                                sub_2 = nx.ego_graph(graph, no_2, radius=raio2)

                                EdgesWeights1 = GraphEdgesWeights(sub_1)
                                EdgesWeights2 = GraphEdgesWeights(sub_2)
                                if EdgesWeights1 == EdgesWeights2:

                                    list_sub_1 = list(sub_1)
                                    list_sub_1.sort()

                                    list_sub_2 = list(sub_2)
                                    list_sub_2.sort()

                                    for n1, n2 in zip(list_sub_1, list_sub_2):
                                        graph.nodes[n2]['weight'] = graph.nodes[n2]['weight'] - graph.nodes[n1][
                                            'weight']
                                        grandparent = graph._pred[no_pai]
                                        if grandparent:
                                            grandparent_edge = graph[list(grandparent)[0]][no_pai]['weight']
                                            graph.add_edge(list(grandparent)[0], no_1, weight=grandparent_edge)

                                        if graph.has_edge(no_pai, no_1):
                                            graph.remove_edge(no_pai, no_1)

    return graph

def Graph_to_Reduced_Paths(graph):
    """
    Produces circuit params from graph
    """
    AllPaths, AllEdgesWeights, AllAngles, AllNodesLevels = CircuitParams(graph)
    AllPaths_, AllEdgesWeights_, AllAngles_, AllNodesLevels_ = SubtractAllPaths(AllPaths, AllEdgesWeights, AllAngles, AllNodesLevels)
    return AllPaths_, AllEdgesWeights_, AllAngles_, AllNodesLevels_

def remove_duplicates(pairs):
    """
    Remove duplicate nodes from pairs paths
    """
    seen = set()
    result = []
    for pair in pairs:
        if pair[0] not in seen and pair[1] not in seen:
            seen.add(pair[0])
            seen.add(pair[1])
            result.append(pair)
    return result

def BuildQuantumCircuit_From_Paths_(num_qubits, graph, r_axis):
    """
    Build circuit from paths, the paths  are obtained from graph
    """
    Err = 0.0001

    AllPaths, AllEdegsWeight, AllAngles, AllNodesLVL = Graph_to_Reduced_Paths(graph)

    qc = QuantumCircuit(num_qubits)

    for i in range(len(AllPaths)):
        Path = AllPaths[i]
        EdgesWeight = AllEdegsWeight[i]
        Angle = AllAngles[i]
        Qubits = AllNodesLVL[i]

        ctrls = Qubits[:-1]
        target = Qubits[-1]

        if abs(Angle) >= Err:
            if len(Path) == 1:
                if r_axis == 'y':
                    qc.ry(Angle, target)
                if r_axis == 'z':
                    qc.rz(Angle, target)
            else:
                num_ctrls = len(ctrls)
                if r_axis == 'y':
                    u = RYGate(Angle).control(num_ctrls)
                if r_axis == 'z':
                    u = RZGate(Angle).control(num_ctrls)
                for q, ew in zip(ctrls, EdgesWeight):
                    if ew == -1:
                        qc.x(q)

                qc.append(u, Qubits)
                for q, ew in zip(ctrls, EdgesWeight):
                    if ew == -1:
                        qc.x(q)
    return qc

def BuildQuantumCircuit_From_Paths__(num_qubits, graph, r_axis):
    """
    Build circuit from paths, the paths  are obtained from graph
    """
    Err = 0.0001

    AllPaths, AllEdegsWeight, AllAngles, AllNodesLVL = Graph_to_Reduced_Paths(graph)

    qc = QuantumCircuit(num_qubits)

    for i in range(len(AllPaths)):
        Path = AllPaths[i]
        EdgesWeight = AllEdegsWeight[i]
        Angle = AllAngles[i]
        Qubits = AllNodesLVL[i]

        ctrls = Qubits[:-1]
        target = Qubits[-1]

        if abs(Angle) >= Err:
            if len(Path) == 1:
                if r_axis == 'y':
                    qc.ry(Angle, target)
                if r_axis == 'z':
                    qc.rz(Angle, target)
            else:
                num_ctrls = len(ctrls)
                if r_axis == 'y':
                    u = RYGate(Angle).control(num_ctrls)
                if r_axis == 'z':
                    u = RZGate(Angle).control(num_ctrls)
                for q, ew in zip(ctrls, EdgesWeight):
                    if ew == 0:
                        qc.x(q)

                qc.append(u, Qubits)
                for q, ew in zip(ctrls, EdgesWeight):
                    if ew == 0:
                        qc.x(q)
    return qc

def BuildQuantumCircuit_From_Paths(Num_qubits, graph, r_axis):
    """
    Build circuit from paths, the paths  are obtained from graph
    """
    Err = 0.0001

    AllPaths, AllEdgesWeights, AllAngles, AllNodesLVLs = Graph_to_Reduced_Paths(graph)

    
    qc = QuantumCircuit(Num_qubits)
    for item in range(len(AllAngles)):
        lvls = AllNodesLVLs[item]
        ctrl = lvls[:-1]
        ctrl = ctrl[::-1]
        ctrlbin = vetor_para_string_binaria(AllEdgesWeights[item])
        trgt = [lvls[-1]]
        angle = AllAngles[item]
        if abs(angle)>=Err and angle!=-9:
            if len(ctrl) == 0:
                if r_axis == 'y':
                    qc.ry(angle, trgt[0])
                if r_axis == 'z':
                    qc.rz(angle, trgt[0])
            else:

                if r_axis == 'y':
                    u = RYGate(angle).control(len(ctrl), None, ctrlbin)
                    qc.append(u, ctrl+trgt)
                if r_axis == 'z':
                    u = RZGate(angle).control(len(ctrl), None, ctrlbin)
                    qc.append(u.reverse_ops(), ctrl+trgt)

    return qc

def GyGz_to_mc(graphy,graphz, global_phase, num_qubits):
  qcy = graph_to_multicontrolled_circuit(graphy, 'y', num_qubits)
  qcz = graph_to_multicontrolled_circuit(graphz, 'z', num_qubits)
  qc = QuantumCircuit.compose(qcy,qcz)
  qc.global_phase = global_phase
  return qc
  
def GyGz_to_mx(graphy,graphz, global_phase, num_qubits): 
  qcy = graph_to_multiplexor(graphy, 'y', num_qubits)
  qcz = graph_to_multiplexor(graphz, 'z', num_qubits)
  qc = QuantumCircuit.compose(qcy,qcz)
  qc.global_phase = global_phase
  return qc

def CNOTS_QC_MC_with_params(graphy, graphx, GP, NQ, ShowCircuits):
  qc = GyGz_to_mc(graphy, graphx, GP, NQ)
  if ShowCircuits:
    qc.draw('mpl')
    plt.show()
  state = get_state(qc.reverse_bits())
  tqc_mc = transpile(qc, basis_gates=['u', 'cx'], optimization_level=0)
  numcnots_mc = tqc_mc.count_ops().get('cx', 0)
  return state, numcnots_mc

def CNOTS_QC_MX_with_params(graphy, graphx, GP, NQ, ShowCircuits):
  qc = GyGz_to_mx(graphy, graphx, GP, NQ)
  qcd = qc.decompose()
  if ShowCircuits:
    qcd.draw('mpl')
  state = get_state(qc.reverse_bits())
  tqc_mx = transpile(qc, basis_gates=['u', 'cx'], optimization_level=0)
  numcnots_mx = tqc_mx.count_ops().get('cx', 0)
  return state, numcnots_mx

def display_all_graph(graphy, graphz, showgraph, name):
  if showgraph:
        plt.figure(name + "Ry")
        display_graph(graphy)
        plt.figure(name + "Rz")
        display_graph(graphz)

def removes_isolated_zero_nodes_from_the_graph(graph):
    
    graph = graph.copy()
    list_nodes = list(graph)
        
    for node in list_nodes:
        if graph.nodes[node]['weight'] == 0:
            if not (graph._pred[node]):
                if len(list(graph.successors(node)))==0:
                    graph.remove_node(node)

    return graph

#added in 07mar25    

def combine_vertically_quantum_circuits(vqc):
    """
    Parameters
    ----------
    vqc : array of quantum circuits 
           

    Returns Vertically combined circuits
    -------
    None.

    """
    nqs = []
    for qc in vqc:
        nqs.append(qc.num_qubits)
    
    total_qubits = sum(nqs)
    
    combined_circuit = QuantumCircuit(total_qubits)

    combined_circuit.compose(vqc[0], qubits = range(nqs[0]), inplace = True)
    for i in range(1,len(nqs)):
        combined_circuit.compose(vqc[i], qubits = range(sum(nqs[:i]), sum(nqs[:i+1])), inplace = True)
        
    return combined_circuit

def int_para_binario(numero):
    if numero == 0:
        return "0"
    binario = ""
    while numero > 0:
        binario = str(numero % 2) + binario
        numero = numero // 2
    return binario

def extend_number_of_bits(BinaryNumber, RequiredDigitsNumber):
    while len(BinaryNumber) < RequiredDigitsNumber:
        BinaryNumber = '0' + BinaryNumber
    BinaryNumber = BinaryNumber +'b'
    return BinaryNumber

def state_vector_to_dict_bin_amplitudes(state_vector):
    """
    Dicionário para armazenar as amplitudes diferentes de zero e os seus respectvos
    índices em binário.
    """        
    
    dictNonZeroBinaryIndexAndAmplitudes = {}
    iNumQubits = int(np.log2(len(state_vector)))
    
    for index, amplitude in enumerate(state_vector):
        if not np.isclose(abs(amplitude), 0):
            dictNonZeroBinaryIndexAndAmplitudes[extend_number_of_bits(int_para_binario(index), iNumQubits)] = amplitude
    return dictNonZeroBinaryIndexAndAmplitudes
   
def find_clusters(graph):
    """
    Parameters
    ----------
    graph : directed  graph
    Returns
    -------
    clusters : root nodes
    The custers are defined by root nodes.
    """
    
    clusters = []
    graphnodeslist = list(graph)
    for no in graphnodeslist:
        if not graph._pred[no]:
            clusters.append(no)
    return clusters

def flatten_list(list_of_lists):
    # transformando uma lista de listas em uma lista única, com elementos únicos
    flattened_list = [item for sublist in list_of_lists for item in sublist]
    flattened_list = list(set(flattened_list))
    return flattened_list

def get_subgraph(graph, root):
    raio = get_graph_depth(graph, root)
    subgraph = nx.ego_graph(graph, root, radius=raio)
    return subgraph
   
    raio = get_graph_depth(graph, root)
    subgraph = nx.ego_graph(graph, root, radius=raio)
    return subgraph

def levelsmap(list_of_lists):
    """
    Mapping of the list of levels to sequence of integers from zero, so that 
    the value of the highest level corresponds to the largest qubit amount.
    
    1 - Extract all unique values from the vector
    2 - Create a mapping dictionary
    3 - Replace the values in the original vector
    """
    newvalues = flatten_list(list_of_lists)
    map = {value: i for i, value in enumerate(newvalues)}
    Newlist_of_lists = [[map[value] for value in sublist] for sublist in list_of_lists]
    return Newlist_of_lists

def graph_to_multiplexor_newlvls(graph, r_axis, Num_qubits):#, NewLVLs):
    """
    Converts a graph to a quantum circuit
    Input: graph
    Process:
    - get all paths, edge weights, angles and node levels from a graph
    - forms groups with all paths that have the same levels
    - each group corresponds to a multiplexer
    - build each multiplexer and append to the quantum circuit
    :return: a quantum circuit
    """
    AllPaths, AllEdgesWeights, AllAngles, AllNodesLVLs = CircuitParams(graph)
    AllNodesLVLs = levelsmap(AllNodesLVLs)
    pares = vPathsPairs(AllPaths, AllNodesLVLs, AllEdgesWeights)
    unified = unify_sets(pares)
    premuxe = premuxes(AllPaths, unified)
    allew = []
    for ew in AllEdgesWeights:
        allew.append(edgesweightstodecimal(ew))
    allew = np.array(allew, dtype=int)
    
    redone = []
    for p in premuxe:
        if isinstance(p, int):
            redone.append([p])
        else:
            redone.append(list(p))
    
    qc = QuantumCircuit(Num_qubits)
    for item in redone:
        lvls = AllNodesLVLs[item[0]]
        szMux = 2 ** (len(lvls) - 1)
        ctrl = lvls[:-1]
        trgt = [lvls[-1]]
        if len(ctrl) == 0:
            angle = AllAngles[item[-1]]
            if r_axis == 'y':
                qc.ry(angle, trgt)
            if r_axis == 'z':
                qc.rz(angle, trgt)
        else:
            angles = np.zeros([szMux])
            for index in item:
                cleanangle = AllAngles[index]
                if cleanangle == -9:
                    angles[allew[index]] = 0
                else:
                    angles[allew[index]] = AllAngles[index]
            if r_axis == 'y':
                ucry = multiplexor(RYGate, angles)
                qc.append(ucry, trgt + ctrl)
            if r_axis == 'z':
                ucry = multiplexor(RZGate, angles)
                qc.append(ucry.reverse_ops(), trgt + ctrl)
    return qc

def qcbuilder_real_amplitudes_09mar25(graph):
    vQCs = []
    roots = find_clusters(graph)
    for root in roots:
        cluster = get_subgraph(graph, root)
        AllLevels = CircuitParams(cluster)[3]
        vSGLvls = flatten_list(AllLevels)
        iClusterNumQubits = len(vSGLvls)
        QC_ = graph_to_multiplexor_newlvls(cluster, 'y', iClusterNumQubits)
        QC_ = transpile(QC_, basis_gates=['u', 'cx'], optimization_level=1)
        vStatei = get_state(QC_.reverse_bits())
        for i in range(len(vStatei)):
            if abs(np.round(vStatei[i],8)) == 0.0: 
                vStatei[i] = 0
        NonZeroDensity = np.count_nonzero(vStatei)/len(vStatei)
        print("nonzerodensity:", NonZeroDensity)
        #cvqoram: funciona bem se m<<2^n e t<<n, onde t é o número de digitos iguais 1 numa string binária
        # Vamos setar para NonZeroDensity <=0.3 e densidade de 1's na string binária for <=0.3 em média
        AverageCounts1 = []
        statedict = state_vector_to_dict_bin_amplitudes(vStatei)
        for NonZeroIndex in statedict:
            print("NonZeroIndex:", NonZeroIndex)
            AverageCounts1.append(NonZeroIndex.count('1'))
        AverageCounts1 = sum(AverageCounts1)/((len(NonZeroIndex)-1)*len(AverageCounts1))
        print("average counts 1:",AverageCounts1)
        print()
        if NonZeroDensity <= 0.4 and AverageCounts1 <=0.4:
           vQCs.append(CvoqramInitialize(statedict, opt_params={'with_aux': False}).definition)            
        else:
            QCi = QuantumCircuit(iClusterNumQubits)
            QCi.initialize(vStatei)
            QCi = transpile(QCi, basis_gates=['u', 'cx'], optimization_level=0)
            QCi = QCi.reverse_bits()
            vQCs.append(QCi)
    
    return combine_vertically_quantum_circuits(vQCs)
   
def build_circuit_QSPmethod_from_split_graph(graph, qspmethod):
    """
    Build a circuit from a split graph
    
    Parameters
    ----------
    graph : a graph split in smaller subgraphs
    
    qspmethod : QSP method to buid the circuits: multiplexor ou cvoqram
      

    Returns
    -------
    The circuit complete

    """
    
    vQCs = []
    roots = find_clusters(graph)
    for root in roots:
        cluster = get_subgraph(graph, root)
        AllLevels = CircuitParams(cluster)[3]
        vSGLvls = flatten_list(AllLevels)
        iClusterNumQubits = len(vSGLvls)
        QC_ = graph_to_multiplexor_newlvls(cluster, 'y', iClusterNumQubits)
        QC_ = transpile(QC_, basis_gates=['u', 'cx'], optimization_level=1)
        vStatei = get_state(QC_.reverse_bits())
        for i in range(len(vStatei)):
            if abs(np.round(vStatei[i],8)) == 0.0: 
                vStatei[i] = 0
        
        if qspmethod == 'cvoqram':
            statedict = state_vector_to_dict_bin_amplitudes(vStatei)
            vQCs.append(CvoqramInitialize(statedict, opt_params={'with_aux': False}).definition)            
        elif qspmethod == 'multiplexor':
            QCi = QuantumCircuit(iClusterNumQubits)
            QCi.initialize(vStatei)
            QCi = transpile(QCi, basis_gates=['u', 'cx'], optimization_level=0)
            QCi = QCi.reverse_bits()
            vQCs.append(QCi)
        else:
            raise(ValueError("QSP method not avaliable."))
            
    
    return combine_vertically_quantum_circuits(vQCs)
    
def PrintNonzeroIndexAmplitudes(statevector, fractionsize, lenline):
    print()
    print('index:\t amplitude\n')
    for i in range(len(statevector)):
        if not np.isclose(statevector[i],0):
            print(int_para_binario(i), round(statevector[i], fractionsize), end =" ")
            if (i+1)%lenline == 0: print()
    print()
    