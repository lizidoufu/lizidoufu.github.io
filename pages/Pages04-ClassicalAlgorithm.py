

files = {
    "数组排序":[
        "2022-0211-1033-【排序】【插入排序，Insertion Sort】.ipynb",
        "2022-0211-1132-【排序】【选择排序，Select Sort】.ipynb",
        "2022-0209-1211-【排序】【冒泡排序，Bubble Sort】.ipynb",
        "2022-0211-1118-【排序】【快速排序，Quick Sort】【分治法】.ipynb",
        "2022-0211-1541-【排序】【归并排序，Merge Sort】【分治法】.ipynb",
        "2022-0211-1646-【排序】【堆排序，Heap Sort】.ipynb",
        "2022-0211-1710-【排序】【计数排序，Counting Sort】.ipynb",
        "2022-0212-1019-【排序】【希尔排序，Shell Sort】.ipynb",
        "2022-0212-1056-【排序】【拓扑排序，Topological Sort】.ipynb",
    ],
    "计算机算法":[
        "2022-0208-1638-【Fibonacci】【递归】.ipynb",
        "2022-0208-1640-【Fibonacci】【递推】.ipynb",
        "2022-0208-1652-【Fibonacci】【动态规划】.ipynb",
        "2022-0208-1710-【遍历问题】【广度优先遍历，BFS】.ipynb",
        "2022-0208-1727-【遍历问题】【深度优先遍历，DFS】.ipynb",
        "2022-0208-2340-【全排列问题】【回溯，Back Track】.ipynb",
        "2022-0208-1527-【TSP】【暴力搜索】.ipynb",
        "2022-0208-1556-【TSP】【贪心搜索】.ipynb",
        "2022-0208-2310-【最优化】【Genetic Algorithm，遗传算法】.ipynb",
    ],
    "最短路径":[
        "2022-0213-1201-【最短路径】【Floyd】【动态规划】.ipynb",
        "2022-0214-1050-【最短路径】【Dijstrak】【贪心，动态规划】.ipynb",
        "2022-0214-1428-【最短路径】【Bellman-Ford】【动态规划】.ipynb",
        "2022-0214-1536-【最短路径】【SPFA，Shortest Path Faster Algorithm】【动态规划，BFS】.ipynb",
        "2022-0208-1727-【遍历问题】【深度优先遍历，DFS】.ipynb",
        "2022-0208-2340-【全排列问题】【回溯，Back Track】.ipynb",
        "2022-0208-2225-【TSP】【Ane Colony，蚁群算法】.ipynb",
        "2022-0215-1001-【MTSP】【遗传算法，贪心】.ipynb",
    ],
    "机器学习":[
        "2022-0209-0812-【二分类】【Logistic回归】.ipynb",
        "2022-0209-0920-【二分类】【BP神经网络】.ipynb",
        "2022-0209-1101-【二分类】【CART决策树】.ipynb",
        "2022-0303-1136-【二分类】【SVM】（SimpleSMO）.ipynb",
        "2022-0303-1236-【二分类】【SVM】（PlattSMO）.ipynb",
        "2022-0303-1310-【二分类】【SVM】（KernelSMO）.ipynb",
        "2022-0210-1022-【最优化】【Gradient Descent，梯度下降法】.ipynb",
        "2022-0210-1045-【最优化】【Stochastic Gradient Descent，随机梯度下降】.ipynb",
        "2022-0209-1541-【聚类】【KMeans】.ipynb",
        "2022-0210-1120-【回归预测】【Linear Regression，LR，线性回归】.ipynb",
        "2022-0215-1201-【回归预测】【Weighted Linear Regression，WLR，加权线性回归】.ipynb",
        "2022-0215-1414-【回归预测】【Lowess，局部加权回归】.ipynb",
        "2022-0210-1147-【时序预测】【Auto Regressive model，AR，自回归模型】.ipynb",
        "2022-0210-1259-【判别分析】【K Nearest Neighbor，KNN，K近邻分析】.ipynb",
        "2022-0304-1640-【判别分析】【KNN】【KDTree-Simple】.ipynb",
        "2022-0210-1838-【异常值检测】【Isolation Forest，孤立森林】.ipynb",
        "2022-0210-1435-【降维】【Principal Component Analysis，PCA，主成分分析】.ipynb",
    ]
   
}



import os 
prefix = "https://gitee.com/Lizidoufu/algorithm-implementation/blob/master"
with open("./pages/Pages04-ClassicalAlgorithm.md", "w") as md:
    md.write("### 算法实现\n\n")
    md.write("手工实现部分算法，内容包含经典算法和机器学习等部分内容，部分代码参考于 LeetCode 和 Blog；\n\n")
    for head in files.keys():
        md.write(f"### {head}\n\n")
        for file in files[head]:
            href = os.path.join(prefix, file)
            md.write(f"<a href='{href}' style='text-decoration: none;'>{file}</a>\n")
    md.write("\n\n")
