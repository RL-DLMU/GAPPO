import pickle
import numpy as np
import json
import os
from collections import Counter
import random
import sys
import yaml
import copy
import logging
from itertools import groupby
import pandas as pd
from datetime import datetime
from collections import defaultdict
from tensorflow.keras.utils import to_categorical
from common.registry import Registry
import world

x_is_color_list = []
solution_found = False


def get_road_dict(roadnet_dict, road_id):
    for item in roadnet_dict['roads']:
        if item['id'] == road_id:
            return item
    raise KeyError("environment and graph setting mapping error, no such road exists")

def adjacency_index2matrix_onehot(adjacency_index, num, self=True):
    # adjacency_index(the nearest K neighbors):[1,2,3]
    """
    if in 1*6 aterial and
        - the 0th intersection,then the adjacency_index should be [0,1,2,3]
        - the 1st intersection, then adj [0,3,2,1]->[1,0,2,3]
        - the 2nd intersection, then adj [2,0,1,3]

    """
    # [batch,agents,neighbors]
    if self:
        for id,node_list in enumerate(adjacency_index):
            node_list.append(id)
    # adjacency_index = np.array(pd.DataFrame(adjacency_index))
    out_dim = len(adjacency_index)
    padd = np.zeros(out_dim, dtype=np.float32)
    # n = 0
    for i, classes in enumerate(adjacency_index):
        one_hot = to_categorical(np.array(classes), num_classes=out_dim)
        while one_hot.shape[0] < num:
            one_hot = np.vstack([one_hot, padd])

        out_matrix = one_hot if i == 0 else np.concatenate([out_matrix, one_hot], axis=0)
    out_matrix = out_matrix.reshape(out_dim,num,out_dim)
    return out_matrix
def adjacency_index2matrix(adjacency_index, num, self=True):
    # adjacency_index(the nearest K neighbors):[1,2,3]
    """
    if in 1*6 aterial and
        - the 0th intersection,then the adjacency_index should be [0,1,2,3]
        - the 1st intersection, then adj [0,3,2,1]->[1,0,2,3]
        - the 2nd intersection, then adj [2,0,1,3]

    """
    if self:
        for id,node_list in enumerate(adjacency_index):
            node_list.append(id)
    out_dim = len(adjacency_index)
    padd = np.zeros(out_dim, dtype=np.float32)
    for i, classes in enumerate(adjacency_index):
        padd = np.zeros(out_dim, dtype=np.float32)
        padd[classes] = 1
        out_matrix = padd if i == 0 else np.concatenate([out_matrix, padd], axis=0)
    out_matrix = out_matrix.reshape(num,out_dim)
    return out_matrix

def adjacency_index2matrix1(adjacency_index, num, out_dim, flag_self, flag_even):
    """
    if in 1*6 aterial and
        - the 0th intersection,then the adjacency_index should be [0,1,2,3]
        - the 1st intersection, then adj [0,3,2,1]->[1,0,2,3]
        - the 2nd intersection, then adj [2,0,1,3]

    """
    if flag_self:
        for id,node_list in enumerate(adjacency_index):
            node_list.append(id)
    if flag_even:
        for id, node_list in enumerate(adjacency_index):
            if -1 in node_list:
                adjacency_index.remove(node_list)
    padd = np.zeros(out_dim, dtype=np.float32)
    out_matrix = []
    for i, classes in enumerate(adjacency_index):
        one_hot = to_categorical(np.array(classes), num_classes=out_dim)
        while one_hot.shape[0] < num:
            one_hot = np.vstack([one_hot, padd])
        out_matrix = one_hot if i == 0 else np.concatenate([out_matrix, one_hot], axis=0)
    out_matrix = out_matrix.reshape(-1, num, out_dim)
    return out_matrix
def calculate_max_route(flow_dict, valid_intersection_id):
    inter_flow_max = {valid_intersection_id[k]: 0 for k in range(0, len(valid_intersection_id))}
    print(valid_intersection_id)
    column = int(valid_intersection_id[-1][15])
    row = int(len(valid_intersection_id) / column)
    for col, inter in enumerate(valid_intersection_id):
        if inter[13] == '2':
            column = col
            row = int(len(valid_intersection_id)/column)
            break

    routes = [d['route'] for d in flow_dict]
    if isinstance(routes[0], str):
        routes = [route.split() for route in routes]
    route_max = [list1 for list1 in routes if len(list1)>3]
    counter = Counter(tuple(sorted(l)) for l in route_max)
    most_common_list1 = counter.most_common(1)

    for route in routes:
        for road in route: 
            if road[5]!='0' and road[7]!='0' and road[5]!=str(row+1) and road[7]!=str(column+1):
                inter_flow_max['intersection_'+road[5]+'_'+road[7]] += 1
    sorted_items = sorted(inter_flow_max.items(), key=lambda item: item[1], reverse=True)
    top_k_inters = [item[0] for item in sorted_items[:3]]
    return top_k_inters,row,column


def build_index_intersection_map(roadnet_file, flow_path):
    """
    generate the map between identity ---> index ,index --->identity
    generate the map between int ---> roads,  roads ----> int
    generate the required adjacent matrix
    generate the degree vector of node (we only care the valid roads(not connect with virtual intersection), and intersections)
    return: map_dict, and adjacent matrix
    res = [net_node_dict_id2inter,net_node_dict_inter2id,net_edge_dict_id2edge,net_edge_dict_edge2id,
        node_degree_node,node_degree_edge,node_adjacent_node_matrix,node_adjacent_edge_matrix,
        edge_adjacent_node_matrix]
    """
    roadnet_dict = json.load(open(roadnet_file, "r"))
    flow_dict = json.load(open(flow_path, "r"))

    virt = "virtual" # judge whether is virtual node, especially in convert_sumo file
    if "gt_virtual" in roadnet_dict["intersections"][0]:
        virt = "gt_virtual"
    valid_intersection_id = [node["id"] for node in roadnet_dict["intersections"] if not node[virt]]
    # traffic_inter_max,row,col = calculate_max_route(flow_dict, valid_intersection_id)

    print(valid_intersection_id)


    node_id2idx = {}
    node_idx2id = {}
    edge_id2idx = {}
    edge_idx2id = {}
    node_id2lead = {}
    node_degrees = []  

    edge_list = []  
    node_list = [] 
    sparse_adj = [] 
    invalid_roads = []
    node_leadid = []
    Heterogeneous_id = [] 
    Heterogeneous_idx = [] 
    lead_id = []
    cur_num = 0 
    num = 0 
    pos_dict = {}

    for node_dict in roadnet_dict["intersections"]:
        if node_dict[virt]:
            for node in node_dict["roads"]:
                invalid_roads.append(node)
            continue
        cur_id = node_dict["id"]

        num += 1
        if len(node_dict['roads']) == 8 :
            node_idx2id[cur_num] = cur_id
            node_id2idx[cur_id] = cur_num
            x = node_dict["point"]["x"]
            y = node_dict["point"]["y"]
            pos_dict[cur_num] = (x, y)
            cur_num += 1
        else: 
            Heterogeneous_id.append(cur_id)
        if cur_num == 1:
            node_id2lead[cur_id] = 1
            node_leadid.append(1)
            lead_id.append(cur_num - 1)
        else:
            node_id2lead[cur_id] = 0
            node_leadid.append(0)

    if cur_num + len(Heterogeneous_id) != len(valid_intersection_id):
        raise ValueError("environment and graph setting mapping error, node 1 to 1 mapping error")

    cur_num = 0
    for edge_dict in roadnet_dict["roads"]:
        edge_id = edge_dict["id"]
        if edge_id in invalid_roads:
            continue
        else:
            edge_idx2id[cur_num] = edge_id
            edge_id2idx[edge_id] = cur_num
            cur_num += 1
            input_node_id = edge_dict['startIntersection']
            output_node_id = edge_dict['endIntersection']
            input_node_idx = node_id2idx[input_node_id]
            output_node_idx = node_id2idx[output_node_id]
            sparse_adj.append([input_node_idx, output_node_idx])


    for node_dict in roadnet_dict["intersections"]:
        if node_dict[virt]:
            continue        
        node_id = node_dict["id"]
        road_links = node_dict['roads']
        input_nodes = []  
        input_edges = []  
        for road_link_id in road_links:
            road_link_dict = get_road_dict(roadnet_dict, road_link_id)
            if road_link_dict['endIntersection'] == node_id:
                if road_link_id in edge_id2idx.keys():
                    input_edge_idx = edge_id2idx[road_link_id]
                    input_edges.append(input_edge_idx)
                else:
                    continue
                start_node = road_link_dict['startIntersection']
                if start_node in node_id2idx.keys():
                    start_node_idx = node_id2idx[start_node]
                    input_nodes.append(start_node_idx)
        if len(input_nodes) != len(input_edges):
            raise ValueError(f"{node_id} : number of input node and edge not equals")
        node_degrees.append(len(input_nodes))
        edge_list.append(input_edges)
        node_list.append(input_nodes)
    [l.sort() for l in node_list]
    node_degrees = np.array(node_degrees) 
    sparse_adj = np.array(sparse_adj)  

    two_jump_nei_adj,one_two_nei_straight_adj,one_two_nei_all_adj = generate_two_hop_neighbors(node_list)

    nei_matrix = copy.deepcopy(node_list)
    Adj_matrix_onehot = adjacency_index2matrix1(nei_matrix, 5, len(nei_matrix), True, False)
    Adj_matrix = np.sum(Adj_matrix_onehot, -2)
    nei_pos_matrix = generate_pos(nei_matrix, Adj_matrix)  

    matrix = copy.deepcopy(node_list)
    Adj_matrix_noself = adjacency_index2matrix(matrix, len(matrix), self=False)
    Adj_matrix = adjacency_index2matrix_onehot(matrix, 5)
    node_BFS = [] 
    nodelayer_BFS = []
    node_precursor = [[-1] for i in range(len(node_list))]
    flag = [0]*len(node_idx2id)
    flag_p = [10000] * len(node_idx2id)
    queue = lead_id 
    level = 0
    for id in queue:
        flag[id] = 1
        flag_p[id] = level
    while queue:  
        n = len(queue)  
        temp = []  
        for i in range(n): 
            a = queue.pop(0)  
            temp.append(a)
            a_n = node_list[a]
            flag_p[a] = level
            for b in a_n:
                if flag[b] == 0:
                    if flag_p[b] > flag_p[a]:
                        node_precursor[b][0] = a
                    queue.append(b)
                    flag[b] = 1
                else:
                    if flag_p[b] > flag_p[a]:
                        node_precursor[b].append(a)
        level += 1
        for i in temp:
            node_BFS.append(i)
        for id in queue:
            flag_p[id] = level
        nodelayer_BFS.append(temp)
    group_num = Registry.mapping['model_mapping']['setting'].param['group_num']
    [l.sort() for l in node_precursor]
    [l.sort() for l in nodelayer_BFS]
    grouping = solve_graph_coloring(len(pos_dict),Adj_matrix_noself, group_num, node_BFS, pos_dict)
    his_pos_matrix = generate_pos_his_vari(node_list, Adj_matrix_noself)
    result = {'node_idx2id': node_idx2id, 'node_id2idx': node_id2idx, 'node_id2lead': node_id2lead,
              'edge_idx2id': edge_idx2id, 'edge_id2idx': edge_id2idx, 'node_BFS': node_BFS, 'nodelayer_BFS': nodelayer_BFS, 'his_pos_matrix': his_pos_matrix,
              'node_degrees': node_degrees, 'sparse_adj': sparse_adj, 'node_precursor': node_precursor, 'grouping': grouping, 'nei_pos_matrix': nei_pos_matrix,
              'node_list': node_list, 'Adj_matrix': Adj_matrix, 'Adj_matrix_noself':Adj_matrix_noself, 'edge_list': edge_list, 'node_leadid': node_leadid,
              'two_jump_nei_adj': two_jump_nei_adj, 'one_two_nei_straight_adj': one_two_nei_straight_adj, 'one_two_nei_all_adj': one_two_nei_all_adj}
    return result

def generate_two_hop_neighbors(node_list):
    one_jump_nei = copy.deepcopy(node_list)
    two_hop_neighbors = [[] for _ in range(48)]
    for node, first_hop_neighbors in enumerate(one_jump_nei):
        for neighbor in first_hop_neighbors:
            two_hop_neighbors[node].append(one_jump_nei[neighbor])
    merged_list = [[item for sublist in sublist_list for item in sublist] for sublist_list in two_hop_neighbors]
    two_jump_nei = [sorted(set(sublist) - {idx}) for idx, sublist in enumerate(merged_list)]  

    two_jump_straight = []
    for sublist in merged_list:
        counts = Counter(sublist)
        unique_elements = [element for element, count in counts.items() if count == 1]
        unique_elements.sort()
        two_jump_straight.append(unique_elements)

    one_two_nei_all = [[*sorted(set(sublist1) | set(sublist2))] for sublist1, sublist2 in
                       zip(one_jump_nei, two_jump_nei)]  
    one_two_nei_straight = [[*sorted(set(sublist1) | set(sublist2))] for sublist1, sublist2 in
                            zip(one_jump_nei, two_jump_straight)]  

    two_jump_nei_matrix = copy.deepcopy(two_jump_nei)
    Adj_matrix_onehot = adjacency_index2matrix1(two_jump_nei_matrix, 1 + max(len(item) for item in two_jump_nei),
                                                len(two_jump_nei_matrix), True, False)
    two_jump_nei_adj = np.sum(Adj_matrix_onehot, -2)

    one_two_nei_straight_matrix = copy.deepcopy(one_two_nei_straight)
    Adj_matrix_onehot = adjacency_index2matrix1(one_two_nei_straight_matrix,
                                                1 + max(len(item) for item in one_two_nei_straight),
                                                len(one_two_nei_straight_matrix), True, False)
    one_two_nei_straight_adj = np.sum(Adj_matrix_onehot, -2)

    one_two_nei_all_matrix = copy.deepcopy(one_two_nei_all)
    Adj_matrix_onehot = adjacency_index2matrix1(one_two_nei_all_matrix, 1 + max(len(item) for item in one_two_nei_all),
                                                len(one_two_nei_all_matrix), True, False)
    one_two_nei_all_adj = np.sum(Adj_matrix_onehot, -2)
    return two_jump_nei_adj, one_two_nei_straight_adj, one_two_nei_all_adj

def generate_pos(matrix, precursor_matrix, leader=None):
    pos_list = (np.zeros_like(precursor_matrix)).tolist()
    for i, nodes in enumerate(matrix):
        for node in nodes:
            if node == -1:
                break
            elif node == i-1: 
                pos_list[i][node] = 3
            elif node == i+1: 
                pos_list[i][node] = 5
            elif node < i: 
                pos_list[i][node] = 7
            elif node > i: 
                pos_list[i][node] = 11
            else: 
                pos_list[i][node] = 13
    return pos_list
def generate_pos_his_vari(matrix, precursor_matrix, leader=None):
    pos_list = (np.zeros_like(precursor_matrix)).tolist()
    his_length = Registry.mapping['trainer_mapping']['setting'].param['his_length']
    pos_list = np.tile(pos_list, (1, his_length))
    n_agent = len(matrix)
    for i, nodes in enumerate(matrix):
        for node in nodes:
            for j in range(his_length):
                if node == -1:
                    break
                elif node == i-1: 
                    pos_list[i][n_agent * j + node] = 3
                elif node == i+1: 
                    pos_list[i][n_agent * j + node] = 5
                elif node < i:
                    pos_list[i][n_agent * j + node] = 7
                elif node > i: 
                    pos_list[i][n_agent * j + node] = 11
                else: 
                    pos_list[i][node] = 5
    return pos_list
def same_color(node, G, x_is_color, n):
    for j in range(n):
        if G[j] == 1 and x_is_color[node] == x_is_color[j]:
            return False
    return True


def distribute_elements(total, num_groups):
    min_elements_per_group = total // num_groups
    elements_per_group = [min_elements_per_group] * num_groups
    remaining_elements = total - min_elements_per_group * num_groups
    for i in range(remaining_elements):
        elements_per_group[i] += 1
    return elements_per_group

def check_group_size(solution, m):
    color_count = Counter(solution)
    n = len(solution)
    ideal_distribution = distribute_elements(n, m)
    actual_distribution = [color_count[i] for i in range(1, m+1)]
    return actual_distribution == ideal_distribution


def check_uniform_distribution(solution, pos_dict):    
    group_sums = {}
    group_counts = {}

    for i, color_val in enumerate(solution):
        x_i, y_i = pos_dict[i]  

        if color_val in group_sums:
            group_sums[color_val][0] += x_i
            group_sums[color_val][1] += y_i
            group_counts[color_val] += 1
        else:
            group_sums[color_val] = [x_i, y_i]
            group_counts[color_val] = 1

    sumx, sumy = 0.0, 0.0
    n = len(pos_dict)
    for idx in pos_dict:
        x_i, y_i = pos_dict[idx]
        sumx += x_i
        sumy += y_i
    center_x = sumx / n
    center_y = sumy / n

    total_dist = 0.0

    for color_val, (sum_x, sum_y) in group_sums.items():
        count = group_counts[color_val]
        mean_x = sum_x / count
        mean_y = sum_y / count

        dist = ((mean_x - center_x)**2 + (mean_y - center_y)**2) ** 0.5
        total_dist += dist


    return total_dist


def check_group_num(solution, m):
    used_colors = set(solution)
    if len(used_colors) != m:
        return False

    return True


def calculate_first_group(groups_list):
    scheme_list = []
    scheme_list1 = []
    num = 0
    for groups in groups_list:
        first_sizes = len(groups[0])
        if first_sizes == num:
            scheme_list.append(groups)
        elif first_sizes > num:
            num = first_sizes
            scheme_list = [groups]
    num = 0
    for groups in scheme_list:
        first_sizes = len(groups[1])
        if first_sizes == num:
            scheme_list1.append(groups)
        elif first_sizes > num:
            num = first_sizes
            scheme_list1 = [groups]
    return scheme_list1

def show(x_is_color):
    print(f'第个着色方案：{x_is_color[0:16]}')

def dfs_color(node, x_color, n, G, m, pos_dict, ideal_distribution, color_usage, top_solutions, k=99):
    if node == n:
        if all(color_usage[i] == ideal_distribution[i] for i in range(m)):
            dist_val = check_uniform_distribution(x_color, pos_dict)  
            add_solution_to_top(dist_val, x_color, top_solutions, k)
        return
    
    left = n - node

    for color_val in range(1, m+1):
        
        cidx = color_val - 1
        if color_usage[cidx] >= ideal_distribution[cidx]:
            continue

        if not is_safe(node, color_val, x_color, G):
            continue

        x_color[node] = color_val
        color_usage[cidx] += 1
        if feasible(color_usage, ideal_distribution, left - 1):
            dfs_color(node+1, x_color, n, G, m, pos_dict,
                      ideal_distribution, color_usage,
                      top_solutions, k)
        x_color[node] = 0
        color_usage[cidx] -= 1


def is_safe(u, color_val, x_color, G):
    for v in range(len(x_color)):
        if G[u][v] != 0 and x_color[v] == color_val:
            return False
    return True


def add_solution_to_top(dist_val, solution, top_solutions, k=99):
    top_solutions.append((dist_val, solution[:]))
    if len(top_solutions) > k:
        max_index = max(range(len(top_solutions)), key=lambda i: top_solutions[i][0])
        top_solutions.pop(max_index)


def feasible(color_usage, ideal_distribution, left):
    needed = 0
    for i in range(len(color_usage)):
        needed += max(0, ideal_distribution[i] - color_usage[i])
    return needed <= left




def check_constraints(solution, m):

    if not check_group_num(solution, m):
        return False
    
    if not check_group_size(solution, m):
        return False
    
    return True


def solve_graph_coloring(n, G, m, s, pos_dict):

    ideal_distribution = distribute_elements(n, m)
    color_usage = [0]*m


    x_color = [0]*n
    top_solutions = []

    dfs_color(0, x_color, n, G, m, pos_dict, ideal_distribution, color_usage, top_solutions, k=99)
    
    top_solutions.sort(key=lambda x: x[0])
    best_dist, best_sol = top_solutions[0]


    groups = [[] for _ in range(m)]
    for i, color_val in enumerate(best_sol):
        groups[color_val - 1].append(i)

    return groups


def is_valid(v, color):
    for i in range(v):
        if graph[v][i] == 1 and color[i] == color[v]:
            return False
    return True


def graph_coloring(m, color, start, v):
    if v == len(graph):
        return True
    for c in range(1, m + 1):
        color[v] = c
        if is_valid(v, color):
            if graph_coloring(m, color, start, v + 1):
                return True
        color[v] = 0
    return False

def solve_graph_colorin(m, n, k, start, graph):
    color = [0] * n
    color[start] = 1

    if not graph_coloring(m, color, start, 0):
        return False
    result = []
    for c in range(1, m + 1):
        result.append([i for i in range(n) if color[i] == c])
    return result

def analyse_vehicle_nums(file_path):
    replay_buffer = pickle.load(open(file_path, "rb"))
    observation = [i[0] for i in replay_buffer]
    observation = np.array(observation)
    observation = observation.reshape([-1])
    print("the mean of vehicle nums is ", observation.mean())
    print("the max of vehicle nums is ", observation.max())
    print("the min of vehicle nums is ", observation.min())
    print("the std of vehicle nums is ", observation.std())


def get_output_file_path(task, model, prefix):
    path = os.path.join('./data/output_data', task, model, prefix)
    return path


def load_config(path, previous_includes=[]):
    if path in previous_includes:
        raise ValueError(
            f"Cyclic configs include detected. {path} included in sequence {previous_includes}."
        )
    previous_includes = previous_includes + [path]

    direct_config = yaml.load(open(path, "r"), Loader=yaml.Loader)

    # Load configs from included files.
    if "includes" in direct_config:
        includes = direct_config.pop("includes")
    else:
        includes = []
    if not isinstance(includes, list):
        raise AttributeError(
            "Includes must be a list, '{}' provided".format(type(includes))
        )

    config = {}
    duplicates_warning = []
    duplicates_error = []

    # TODO: Need test duplication here
    for include in includes:
        include_config, inc_dup_warning, inc_dup_error = load_config(
            include, previous_includes
        )
        duplicates_warning += inc_dup_warning
        duplicates_error += inc_dup_error

        # Duplicates between includes causes an error
        config, merge_dup_error = merge_dicts(config, include_config)
        duplicates_error += merge_dup_error

    # Duplicates between included and main file causes warnings
    config, merge_dup_warning = merge_dicts(config, direct_config)
    duplicates_warning += merge_dup_warning

    return config, duplicates_warning, duplicates_error


def merge_dicts(dict1: dict, dict2: dict):
    """Recursively merge two dictionaries.
    Values in dict2 override values in dict1. If dict1 and dict2 contain a dictionary as a
    value, this will call itself recursively to merge these dictionaries.
    This does not modify the input dictionaries (creates an internal copy).
    Additionally returns a list of detected duplicates.
    Adapted from https://github.com/TUM-DAML/seml/blob/master/seml/utils.py
    Parameters
    ----------
    dict1: dict
        First dict.
    dict2: dict
        Second dict. Values in dict2 will override values from dict1 in case they share the same key.
    Returns
    -------
    return_dict: dict
        Merged dictionaries.
    """
    if not isinstance(dict1, dict):
        raise ValueError(f"Expecting dict1 to be dict, found {type(dict1)}.")
    if not isinstance(dict2, dict):
        raise ValueError(f"Expecting dict2 to be dict, found {type(dict2)}.")

    return_dict = copy.deepcopy(dict1)
    duplicates = []

    for k, v in dict2.items():
        if k not in dict1:
            return_dict[k] = v
        else:
            if isinstance(v, dict) and isinstance(dict1[k], dict):
                return_dict[k], duplicates_k = merge_dicts(dict1[k], dict2[k])
                duplicates += [f"{k}.{dup}" for dup in duplicates_k]
            else:
                return_dict[k] = dict2[k]
                duplicates.append(k)

    return return_dict, duplicates


def build_config(args):
    # configs file of specific agents is loaded from configs/agents/{agent_name}
    agent_name = os.path.join('./configs/agents', args.task, f'{args.agent}.yml')
    config, duplicates_warning, duplicates_error = load_config(agent_name)
    if len(duplicates_warning) > 0:
        logging.warning(
            f"Overwritten configs parameters from included configs "
            f"(non-included parameters take precedence): {duplicates_warning}"
        )
    if len(duplicates_error) > 0:
        raise ValueError(
            f"Conflicting (duplicate) parameters in simultaneously "
            f"included configs: {duplicates_error}"
        )
    args_dict = vars(args)
    for key in args_dict:
        config.update({key: args_dict[key]})  # short access for important param

    # add network(for FRAP and MPLight)
    cityflow_setting = json.load(open(config['path'], 'r'))
    config['traffic']['network'] = cityflow_setting['network'] if 'network' in cityflow_setting.keys() else None
    return config

