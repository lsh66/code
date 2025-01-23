import networkx as nx
import matplotlib.pyplot as plt



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
    if label == "0":
        node_colors.append('red')
    elif label == "33":
        node_colors.append('blue')


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
node_labels = {node: f"{node} ({nodes_neighbors[node][3][1]})" for node in range(N)}
nx.draw_networkx_labels(G, pos, labels=node_labels, font_color='white')

# 显示图形
plt.axis('off')
plt.show()