import numpy as np


# ==============================================================================
#                             ЗАДАЧА 1 (BFS)
# ==============================================================================
def print_multiplication_step(front, matrix):
    N = len(front)
    def fmt(val):
        if np.isinf(val): return "None"
        return f"{int(val):>4}"

    front_str = "[" + ", ".join(fmt(v) for v in front) + "]"
    
    cols_header = " ".join(f"{i:>4}" for i in range(N))
    pad = len(front_str) + 4
    
    print(f"{'':{pad}}         {cols_header}")
    print() 

    for i in range(N):
        row_str = "[" + ", ".join(fmt(v) for v in matrix[i]) + "]"
        if i == N // 2:
            print(f"{front_str:<{pad}}   X   {i} {row_str}")
        else:
            print(f"{'':<{pad}}       {i} {row_str}")
            
    print("-" * (pad + len(row_str) + 12))


def generalized_vxm(front_vector, adj_matrix, add_op, mul_op):
    N = len(front_vector)
    next_front = np.full(N, np.inf)

    for j in range(N):
        vals_to_aggregate = []
        for i in range(N):
            if adj_matrix[i, j] != 0 and front_vector[i] != np.inf:
                val = mul_op(front_vector[i], adj_matrix[i, j])
                vals_to_aggregate.append(val)
        
        if vals_to_aggregate:
            next_front[j] = add_op(vals_to_aggregate)

    return next_front


def algebraic_bfs(adj_matrix, start_node):
    N = adj_matrix.shape[0]
    distances = np.full(N, np.inf)
    distances[start_node] = 0
    front = np.full(N, np.inf)
    front[start_node] = 0
    
    def add_min(values): return min(values)
    def mul_plus_one(f_val, m_val): return f_val + 1

    level = 1
    while True:
        print(f"\n======== ШАГ {level} ========")
        print_multiplication_step(front, adj_matrix)

        next_front = generalized_vxm(front, adj_matrix, add_min, mul_plus_one)
        
        visited_mask = ~np.isinf(distances)
        next_front[visited_mask] = np.inf
        
        def fmt_arr(arr): 
            return "[" + ", ".join("None" if np.isinf(x) else str(int(x)) for x in arr) + "]"
        
        print(f"f= {fmt_arr(next_front)}")

        if np.all(np.isinf(next_front)):
            print("\n-> Фронт пуст. Обход завершен!\n")
            break
            
        distances = np.minimum(distances, next_front)
        front = np.copy(next_front)
        
        print(f"n= {fmt_arr(distances)}")
        level += 1

    return "[" + ", ".join("None" if np.isinf(x) else str(int(x)) for x in distances) + "]"


# ==============================================================================
#                       ЗАДАЧА 2 (Замыкание графа / MST)
# ==============================================================================
def print_matrix(matrix):
    N = matrix.shape[0]
    def fmt(val):
        if np.isinf(val): return "None"
        return f"{int(val):>4}"
        
    cols_header = " ".join(f"{i:>4}" for i in range(N))
    print(f"       {cols_header}")
    for i in range(N):
        row_str = "[" + ", ".join(fmt(v) for v in matrix[i]) + "]"
        print(f"   {i} {row_str}")


def generalized_mxm(A, B, add_op, mul_op):
    N = A.shape[0]
    C = np.full((N, N), np.inf)

    for i in range(N):
        for j in range(N):
            vals = []
            for k in range(N):
                if A[i, k] != np.inf and B[k, j] != np.inf:
                    vals.append(mul_op(A[i, k], B[k, j]))
            
            if vals:
                C[i, j] = add_op(vals)

    return C


def graph_closure(adj_matrix, add_op, mul_op):
    N = adj_matrix.shape[0]
    result = np.copy(adj_matrix)

    level = 1
    while True:
        print(f"\n======== ВОЗВЕДЕНИЕ В СТЕПЕНЬ (Шаг {level}) ========")
        next_result = generalized_mxm(result, adj_matrix, add_op, mul_op)
        merged_result = np.minimum(result, next_result)
        
        print("Матрица после шага:")
        print_matrix(merged_result)
        
        if np.array_equal(result, merged_result):
            print("\n-> Матрица перестала меняться. Замыкание достигнуто!\n")
            break
            
        result = merged_result
        level += 1
        
    return result


# ==============================================================================
#                                ГЛАВНОЕ МЕНЮ
# ==============================================================================
def run_task1():
    print("\n[ЗАПУСК ЗАДАЧИ 1: Обобщенный BFS]")
    A = np.array([
        [0, 1, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
    ])
    start_vertex = 0
    print("Старт BFS с вершины:", start_vertex)
    result = algebraic_bfs(A, start_vertex)
    print("ИТОГОВЫЕ РАССТОЯНИЯ:", result)


def run_task2():
    print("\n[ЗАПУСК ЗАДАЧИ 2: Обобщенное замыкание графа (MST)]")
    A = np.array([
        [np.inf, 1,      3],
        [1,      np.inf, 2],
        [3,      2,      np.inf]
    ])

    def add_min_mst(values): return min(values)
    def mul_max_mst(a, b): return max(a, b)
    
    print("Исходная матрица:")
    print_matrix(A)

    closure_result = graph_closure(A, add_min_mst, mul_max_mst)

    N = A.shape[0]
    mst_result = np.full((N, N), np.inf)
    for i in range(N):
        for j in range(N):
            if A[i, j] != np.inf and A[i, j] == closure_result[i, j]:
                mst_result[i, j] = A[i, j]
                
    print("=======================================================================")
    print(" ИТОГОВЫЙ ОТВЕТ: МАТРИЦА СМЕЖНОСТИ MST (Минимального остовного дерева)")
    print("=======================================================================")
    print_matrix(mst_result)



print("=========================================")
print("         ВЫБОР ЗАДАЧИ АЛГЕБРЫ ГРАФОВ     ")
print("=========================================")
print("1 - Задача 1 (Обобщенный BFS: вектор на матрицу)")
print("2 - Задача 2 (Обобщенное замыкание и поиск MST)")
print("=========================================")
    
choice = input("Введите номер задачи (1 или 2): ").strip()
    
if choice == '1':
    run_task1()
elif choice == '2':
    run_task2()
else:
    print("Ошибка: неверный ввод. Запустите программу заново.")
