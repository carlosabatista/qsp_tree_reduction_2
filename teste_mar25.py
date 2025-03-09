#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 08:54:54 2025

@author: linho
"""

import numpy as np
from qsp_tree_reduction import *
from qclib.state_preparation import CvoqramInitialize
from qiskit import QuantumCircuit
import matplotlib.pyplot as plt

"""
Fazer primeiro para amplitudes reais, depois com complexas
Para complexas, ajustar qcbuilder_real_amplitudes_07mar25 para casos complexos, ou seja argumentos passam de (graph) para (graphy, graphz)

Testes
0 - É gerado um estado produto a partir de um estado denso e um esparso
1 - Prepara-se árvore sem don't cares' 
2 - Prepara-se o estado usando multiplexor, conta-se o número de CNOTs e a profundidade
3 - Prepara-se o estado usando CVOQRAM, conta-se o número de CNOTs e a profundidade
4 - Prepara-se a árvore com don't cares
5 - Produz o circuito misto (multiplexor para denso e CVQORAM para esparso), conta-se o número de CNOTs e a profundidade
"""

show_circuits = True
show_graphs = True
RTOL = 1e-6
fractionsize = 6

#np.random.seed(3)
state1 = np.random.rand(4)
state1 = state1 / np.linalg.norm(state1)
state2 = generate_random_state_n_density(4, 0.3)
state2=abs(state2)
state = abs(np.kron(state1, state2))
state = state/np.linalg.norm(state)

num_qubits = int(np.log2(len(state)))

state_tree = get_state_tree(state, num_qubits)
global_phase = state_tree.arg
angle_tree = create_angles_tree(state_tree)

Gy = tree_to_graph_weighted_edges(angle_tree, 'y')
#Gz = tree_to_graph_weighted_edges(angle_tree, 'z')
#display_all_graph(Gy, Gz, show_graphs, 'original')
display_graph(Gy) 


#Multiplexor Circuit
qc_mx = QuantumCircuit(num_qubits)
qc_mx.initialize(state)
qc_mx = transpile(qc_mx, basis_gates=['u', 'cx'], optimization_level=0)
qc_mx = qc_mx.reverse_bits()
NumCnots_mx = qc_mx.count_ops().get('cx', 0)
Depth_mx = qc_mx.depth()

#CVQOQRAM Circuit
statedict = state_vector_to_dict_bin_amplitudes(state)
qc_cvoqram = CvoqramInitialize(statedict, opt_params={'with_aux': False}).definition
qc_cvoqram = transpile(qc_cvoqram, basis_gates=['u', 'cx'], optimization_level=0)
NumCnots_cvoqram = qc_cvoqram.count_ops().get('cx', 0)
Depth_cvoqram = qc_cvoqram.depth()

#Grafo with don't cares and reductions
angle_tree_dc = create_angles_tree_with_dont_cares(state_tree)
Gy_dc = tree_to_graph_weighted_edges(angle_tree_dc, 'y')
display_graph(Gy_dc)
Gy_dc = prune_graph_dontcares(Gy_dc)
Gy_dc = merge_equal_sibling_subtrees(Gy_dc)
Gy_dc = remove_first_zero_node(Gy_dc)
Gy_dc=removes_isolated_zero_nodes_from_the_graph(Gy_dc)
display_graph(Gy_dc)

#reduced circuits
qc_reduc_mx = build_circuit_QSPmethod_from_split_graph(Gy_dc, 'multiplexor')
qc_reduc_cvoqram = build_circuit_QSPmethod_from_split_graph(Gy_dc, 'cvoqram')

NumCnots_reduc_mx = qc_reduc_mx.count_ops().get('cx', 0)
Depth_reduc_mx = qc_reduc_mx.depth()

NumCnots_reduc_cvoqram = qc_reduc_cvoqram.count_ops().get('cx', 0)
Depth_reduc_cvoqram = qc_reduc_cvoqram.depth()

#better circuit
qc = qcbuilder_real_amplitudes_09mar25(Gy_dc)
qc = transpile(qc, basis_gates=['u', 'cx'], optimization_level=0)
NumCnots = qc.count_ops().get('cx', 0)
Depth = qc.depth()

qc_mx.draw('mpl')
qc_cvoqram.draw('mpl')
qc_reduc_mx.draw('mpl')
qc_reduc_cvoqram.draw('mpl')
qc.draw('mpl')

out_state_mx = get_state(qc_mx.reverse_bits())
out_state_cvoqram = get_state(qc_cvoqram)
out_state_reduc_mx = get_state(qc_reduc_mx.reverse_bits())
out_state_reduc_cvoqram = get_state(qc_reduc_cvoqram)
out_state_qc = get_state(qc.reverse_bits())
print()
print("state:")
print(state)
PrintNonzeroIndexAmplitudes(state,4,4)
print()

print("out state mx:")
#print(out_state_mx)
PrintNonzeroIndexAmplitudes(out_state_mx,4,4)
print()

print("out state cvoqram:")
#print(out_state_cvoqram)
PrintNonzeroIndexAmplitudes(out_state_cvoqram,4,4)
print()

print("out state reduced mx:")
#print(out_state_reduc_mx)
PrintNonzeroIndexAmplitudes(out_state_reduc_mx,4,4)
print()

print("out state reduced cvoqram:")
#print(out_state_reduc_cvoqram)
PrintNonzeroIndexAmplitudes(out_state_reduc_cvoqram, 4,4)
print()

print("out state reduced qc:")
#print(out_state_qc)
PrintNonzeroIndexAmplitudes(out_state_qc, 4, 4)
print()

print("CNOT counts and Depth: ")
print("mx            : ", NumCnots_mx, '\t', Depth_mx)
print("cvoqram       : ", NumCnots_cvoqram , '\t', Depth_cvoqram)
print("reduc mx      : ", NumCnots_reduc_mx , '\t', Depth_reduc_mx)
print("reduc cvoqram : ", NumCnots_reduc_cvoqram, '\t', Depth_reduc_cvoqram)
print("reduc qc      : ", NumCnots, '\t', Depth)

