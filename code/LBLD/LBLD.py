import time
from itertools import groupby
from math import ceil
from operator import itemgetter
import numpy as np
from numpy import loadtxt
from pandas import DataFrame
import collections
import networkx as nx
from sklearn.metrics.cluster import normalized_mutual_info_score

import matplotlib.pyplot as plt

# ---------------------------------- 加载数据集 -------------------------------------------
dataset_name = "karate"  # 数据集名称
path = "../data/datasets/" + dataset_name + ".txt"  # 数据集路径

iteration = 1  # 标签选择步骤的迭代次数（通常设置为1或2）
merge_flag = 1  # 是否合并社区：0 -> 不合并，1 -> 合并
write_flag = 1  # 是否将节点标签写入文件：1 -> 写入，0 -> 不写入
modularity_flag = 1  # 是否计算模块度：1 -> 计算，0 -> 不计算
NMI_flag = 1  # 是否计算归一化互信息（NMI）：1 -> 计算，0 -> 不计算

# ------------------------- 计算节点邻居和节点度数 --------------------------
nodes_neighbors = {}  # 存储每个节点的邻居和度数

i = 0
with open(path) as f:
    for line in f:
        row = str(line.strip()).split('\n')[0].split('\t')  # 去除首尾空白字符并按制表符分割
        temp_array = []

        for j in row:
            if j == '':  # 如果元素为空则添加-1
                temp_array.append(-1)
            else:
                temp_array.append(int(j))  # 转换为整数并添加到临时列表中

        nodes_neighbors.setdefault(i, []).append(temp_array)  # 将处理后的临时列表添加到字典中

        if nodes_neighbors[i][0][0] != -1:  # 添加邻居数量
            nodes_neighbors.setdefault(i, []).append(len(nodes_neighbors[i][0]))
        else:
            nodes_neighbors.setdefault(i, []).append(0)

        i += 1

N = i  # 节点总数
start_time = time.time()

# ----------------------------- 计算节点重要性 ------------------------------
for i in range(N):
    CN_sum = 0
    d = {}
    if nodes_neighbors[i][1] > 1:
        for neighbor in nodes_neighbors[i][0]:
            intersect = len(list(set(nodes_neighbors[i][0]) & set(nodes_neighbors[neighbor][0])))
            union = nodes_neighbors[i][1] + nodes_neighbors[neighbor][1] - intersect

            if nodes_neighbors[i][1] > nodes_neighbors[neighbor][1]:
                difResult = 1 + len(set(nodes_neighbors[neighbor][0]).difference(set(nodes_neighbors[i][0])))
            else:
                difResult = 1 + len(set(nodes_neighbors[i][0]).difference(set(nodes_neighbors[neighbor][0])))

            CN_sum += ((intersect / (intersect + union)) * (intersect / difResult))
            d[neighbor] = (neighbor, ((intersect / (intersect + union)) * (intersect / difResult)))

    elif nodes_neighbors[i][1] == 1:
        CN_sum = 0
        d[nodes_neighbors[i][0][0]] = (nodes_neighbors[i][0][0], 0)

    elif nodes_neighbors[i][1] == 0:
        CN_sum = 0
        d[-1] = (-1, -1)

    nodes_neighbors.setdefault(i, []).append(list(max(d.values(), key=itemgetter(1))))
    nodes_neighbors.setdefault(i, []).append([CN_sum, i, 0])

nodes_neighbors = {k: v for k, v in sorted(nodes_neighbors.items(), key=lambda item: item[1][3][0], reverse=True)}

# -------------------------- 选择最相似的邻居 -----------------------
for i in range(N):
    if nodes_neighbors[i][1] > 1:
        if nodes_neighbors[i][2][1] == 0:  # 如果相似度为0，则选择度数最高的邻居
            neighbors_degree = []
            for j in nodes_neighbors[i][0]:
                neighbors_degree.append((j, nodes_neighbors[j][1]))

            max_degree_neighbor = max(neighbors_degree, key=itemgetter(1))[0]
            nodes_neighbors.setdefault(i, []).append([max_degree_neighbor, -1])
            nodes_neighbors[i][3][1] = max_degree_neighbor

        elif nodes_neighbors[i][2][1] != 0:
            if nodes_neighbors[i][3][0] > nodes_neighbors[nodes_neighbors[i][2][0]][3][0]:
                nodes_neighbors.setdefault(i, []).append([i, nodes_neighbors[i][2][0]])
                nodes_neighbors[i][3][1] = i
            else:
                nodes_neighbors.setdefault(i, []).append([nodes_neighbors[i][2][0], nodes_neighbors[i][2][0]])
                nodes_neighbors[i][3][1] = nodes_neighbors[i][2][0]
    else:
        nodes_neighbors.setdefault(i, []).append([i, -1])
        nodes_neighbors[i][3][1] = i

for i in range(N):
    nodes_neighbors[i][3][1] = nodes_neighbors[nodes_neighbors[i][4][0]][3][1]

# ---------------------------- 选取前5%重要节点 -----------------------------
top_5percent = ceil(N * 5 / 100)
most_important = {}

dict_items = nodes_neighbors.items()
selected_items = list(dict_items)[:top_5percent]  # 取前5%项

for i in range(top_5percent):
    most_important[selected_items[i][0]] = (nodes_neighbors[selected_items[i][0]][4][1])

for i in most_important:
    temp_label = []
    if nodes_neighbors[i][3][0] >= nodes_neighbors[most_important[i]][3][0]:
        temp_label = nodes_neighbors[i][3][1]
        nodes_neighbors[most_important[i]][3][1] = temp_label
    else:
        temp_label = nodes_neighbors[most_important[i]][3][1]
        nodes_neighbors[i][3][1] = temp_label

    CN = list(set(nodes_neighbors[i][0]) & set(nodes_neighbors[most_important[i]][0]))

    for j in CN:
        nodes_neighbors[j][3][1] = temp_label
        nodes_neighbors[j][3][2] = 1

        nodes_neighbors[i][3][2] = 1
        nodes_neighbors[most_important[i]][3][2] = 1

del most_important
del CN

# --------------------------------- 平衡标签扩散 ------------------------------------------------
flag_lock = 1
counter = 1
high = 0
low = N - 1
nodes_key = list(nodes_neighbors.keys())

while counter < (N + 1):

    if flag_lock == 1:
        current_node = nodes_key[high]
        high += 1
        flag_lock = 0

        if nodes_neighbors[current_node][1] > 1:
            if nodes_neighbors[current_node][3][2] == 0:
                current_node_neighbor = []
                for j in nodes_neighbors[current_node][0]:
                    current_node_neighbor.append((j, nodes_neighbors[j][3][1]))

                sorted_input = sorted(current_node_neighbor, key=itemgetter(1))
                groups = groupby(sorted_input, key=itemgetter(1))

                neighbors_influence = []

                for i in groups:
                    sum_values = 0
                    for j in i[1]:
                        sum_values += nodes_neighbors[j[0]][3][0]
                    neighbors_influence.append((i[0], sum_values))
                nodes_neighbors[current_node][3][1] = max(neighbors_influence, key=itemgetter(1))[0]

    elif flag_lock == 0:
        current_node = nodes_key[low]
        low -= 1
        flag_lock = 1

        if nodes_neighbors[current_node][1] > 1:
            if nodes_neighbors[current_node][3][2] == 0:
                current_node_neighbor = []
                for j in nodes_neighbors[current_node][0]:
                    current_node_neighbor.append((j, nodes_neighbors[j][3][1]))

                sorted_input = sorted(current_node_neighbor, key=itemgetter(1))
                groups = groupby(sorted_input, key=itemgetter(1))

                neighbors_influence = []

                for i in groups:
                    sum_values = 0
                    for j in i[1]:
                        sum_values += (nodes_neighbors[current_node][1] * nodes_neighbors[j[0]][1])
                    neighbors_influence.append((i[0], sum_values))
                nodes_neighbors[current_node][3][1] = max(neighbors_influence, key=itemgetter(1))[0]

    counter += 1

del groups
del neighbors_influence

# ----------------------------- 给度数为1的节点分配标签 ---------------------------------

for i in range(N):
    if nodes_neighbors[i][1] == 1:
        nodes_neighbors[i][3][1] = nodes_neighbors[nodes_neighbors[i][0][0]][3][1]

# --------------------- 标签选择步骤（算法的迭代部分） ---------------------

for itter in range(iteration):
    for i in range(N):
        if nodes_neighbors[i][1] > 1:
            current_node_neighbor = []

            for j in nodes_neighbors[i][0]:
                current_node_neighbor.append((j, nodes_neighbors[j][3][1]))  # 邻居及其标签
            sorted_input = sorted(current_node_neighbor, key=itemgetter(1))
            groups = groupby(sorted_input, key=itemgetter(1))  # 按社区标签分组

            neighbors_frequency = []
            for j in groups:
                neighbors_frequency.append((j[0], len(list(j[1]))))  # 标签频率

            temp_max = max(neighbors_frequency, key=itemgetter(1))  # 频率最高的标签
            indices = [x for x, y in enumerate(neighbors_frequency) if y[1] == temp_max[1]]

            selected_label = []
            if len(indices) == 1:
                selected_label = temp_max[0]
            else:
                final_max = []
                max_influence = []
                for x in indices:
                    final_max.append(neighbors_frequency[x][0])  # 仅存储频率最高的标签

                for x in final_max:
                    temp_influence = 1
                    for y in current_node_neighbor:
                        if y[1] == x:
                            temp_influence *= nodes_neighbors[y[0]][3][0]
                    max_influence.append((x, temp_influence))

                selected_label = max(max_influence, key=itemgetter(1))[0]
            nodes_neighbors[i][3][1] = selected_label

# ---------------------------------- 合并小社区 -------------------------------------------------
if merge_flag == 1:

    nodes_labels = DataFrame.from_dict(nodes_neighbors, orient='index')
    nodes_labels.columns = ['Neighbor', 'Degree', 'max_Similar', 'NI_Label', 'node_NeighborLabel']
    unique_labels = nodes_labels['NI_Label'].apply(lambda x: x[1]).unique()

    communities_group = {}
    for i in unique_labels:
        communities_group[i] = []

    for i in range(N):
        communities_group[nodes_neighbors[i][3][1]].append(i)  # 按社区分组

    unique_labels_array = []
    for i in communities_group:
        temp_len = len(communities_group[i])
        if temp_len > 1:
            unique_labels_array.append((i, temp_len))

    max_community = max(unique_labels_array, key=itemgetter(1))[1]  # 最大社区的大小
    average_size = (N - max_community) / (len(unique_labels_array) - 1)  # 社区平均大小

    less_than_average_communities = list(filter(lambda x: x[1] < average_size, unique_labels_array))

    if less_than_average_communities:

        for i in less_than_average_communities:
            temp_small_communities = []

            for j in communities_group[i[0]]:
                temp_small_communities.append((j, nodes_neighbors[j][1] + nodes_neighbors[j][3][0]))

            candidate_node = max(temp_small_communities, key=itemgetter(1))[0]  # 社区候选节点

            temp_neighbors = []
            for j in nodes_neighbors[candidate_node][0]:
                temp_neighbors.append(
                    (j, nodes_neighbors[j][1] + nodes_neighbors[j][3][0]))  # 邻居及其得分

            max_neighbor_community = max(temp_neighbors, key=itemgetter(1))[0]  # 分数最高的邻居
            selected_label = []
            if nodes_neighbors[max_neighbor_community][3][1] != nodes_neighbors[candidate_node][3][1]:
                if nodes_neighbors[max_neighbor_community][1] >= nodes_neighbors[candidate_node][1]:
                    selected_label = nodes_neighbors[max_neighbor_community][3][1]
            if selected_label:
                for j in temp_small_communities:
                    nodes_neighbors[j[0]][3][1] = selected_label

# ------------------------------- 算法总执行时间 ---------------------------------------------------------------------
print("--- 总执行时间 %s 秒 ---" % (time.time() - start_time))

# ----------------------------------- 写入磁盘 ------------------------------------------------------
ordered_nodes_neighbors = collections.OrderedDict(sorted(nodes_neighbors.items()))
if write_flag == 1:
    with open('../results/' + dataset_name + '.txt', 'w') as filehandle:
        for i in ordered_nodes_neighbors:
            filehandle.write('%s\n' % ordered_nodes_neighbors[i][3][1])

# ---------------------------------- 社区数量 --------------------------------
nodes_labels = DataFrame.from_dict(nodes_neighbors, orient='index')
nodes_labels.columns = ['Neighbor', 'Degree', 'max_Similar', 'NI_Label', 'node_NeighborLabel']
number_of_communities = nodes_labels['NI_Label'].apply(lambda x: x[1]).unique()
print("社区数量: ", len(number_of_communities))

# ----------------------------------- 模块度 -------------------------------------------------------
if modularity_flag == 1:
    t = 0
    for i in nodes_neighbors:
        t += nodes_neighbors[i][1]
    edges = t / 2
    modu = 0
    are_neighbor = []

    for i in range(N):
        for j in range(N):
            if nodes_neighbors[i][3][1] == nodes_neighbors[j][3][1]:
                if nodes_neighbors[i][1] >= 1:
                    if j in nodes_neighbors[i][0]:
                        are_neighbor = 1
                    else:
                        are_neighbor = 0
                    modu += (are_neighbor - ((nodes_neighbors[i][1] * nodes_neighbors[j][1]) / (2 * edges)))

    modularity_final = modu / (2 * edges)
    print('模块度:  {}'.format(modularity_final))

# ------------------------------- 归一化互信息（NMI） ---------------------------------------
if NMI_flag == 1:
    real_labels = loadtxt("../data/groundtruth/" + dataset_name + "_real_labels.txt", comments="#", delimiter="\t",
                          unpack=False)
    detected_labels = []
    if dataset_name in ('karate', 'dolphins', 'polbooks', 'football'):
        for i in ordered_nodes_neighbors:
            detected_labels.append(ordered_nodes_neighbors[i][3][1])

        detected_labels = np.array(detected_labels)
        print('NMI:  {}'.format(normalized_mutual_info_score(real_labels, detected_labels)))

    else:
        nodes_map = loadtxt("../data/datasets/nodes_map/" + dataset_name + "_nodes_map.txt", comments="#",
                            delimiter="\t", unpack=False)

        for i in nodes_map:
            detected_labels.append(nodes_neighbors[i][3][1])

        print('NMI:  {}'.format(normalized_mutual_info_score(real_labels, detected_labels)))
print(ordered_nodes_neighbors)


# nodes_neighbors[i][0]：节点的邻居列表。
# nodes_neighbors[i][1]：节点的度数（邻居的数量）。
# nodes_neighbors[i][2]：最相似的邻居及其相似度。
# nodes_neighbors[i][3]：节点的标签信息，具体包含：
# nodes_neighbors[i][3][0]：节点的重要性分数。
# nodes_neighbors[i][3][1]：节点的标签。
# nodes_neighbors[i][3][2]：标签是否已更新的标志。
# nodes_neighbors[i][4]：节点的邻居标签信息。

# 节点总数
N = len(nodes_neighbors)

# 创建一个空的无向图
G = nx.Graph()

# 添加节点
for node in range(N):
    G.add_node(node)

# 添加边
for node, data in nodes_neighbors.items():
    neighbors = data[0]
    for neighbor in neighbors:
        if neighbor != -1:
            G.add_edge(node, neighbor)

# 定义节点的颜色
node_colors = []
for node in range(N):
    label = nodes_neighbors[node][3][1]  # 节点的标签

    if label == 0:
        node_colors.append('red')
    elif label == 33:
        node_colors.append('blue')
    else:
        node_colors.append('green')

# 定义节点的大小
node_sizes = [1000 * nodes_neighbors[node][3][0] for node in range(N)]  # 根据重要性分数设置节点大小

# 定义边的颜色
edge_colors = ['black' for _ in G.edges]

# 选择布局
pos = nx.spring_layout(G)

# 绘制节点，使用自定义颜色和大小
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)

# 绘制边，使用自定义颜色
nx.draw_networkx_edges(G, pos, edge_color=edge_colors)

# 添加节点标签，设置标签颜色
node_labels = {node: f"{node} " for node in range(N)}
nx.draw_networkx_labels(G, pos, labels=node_labels, font_color='white')

# 显示图形
plt.axis('off')
plt.show()