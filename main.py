#Дана разреженная матрица (CRS). Осуществить циклический сдвиг в матрице. Сдвинуть всю матрицу. 
#В первой строке первый элемент переносится на второе место, второй элемент на третье и т.д.
#Последний элемент в первой строке становиться первым элементом во второй строке.
#Последний элемент в последней строке переноситься на первую строку на первое место.

import pickle
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import time
import os
import psutil


def save_matrix(matrix, path='matrix.pkl'):
    with open(path, 'wb') as f:
        pickle.dump(matrix, f)


def load_matrix(path='matrix.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)


def print_sparse_matrix(matrix):
    rows, cols = matrix.shape
    max_dim = 20

    def print_submatrix(r_start, r_end, c_start, c_end, title):
        print(f"\n{title}")
        sub = matrix[r_start:r_end, c_start:c_end].toarray()
        for row in sub:
            print(" ".join(str(el) for el in row))

    if rows <= max_dim and cols <= max_dim:
        print(matrix.toarray())
    else:
        print_submatrix(0, 10, 0, 10, "Верхний левый угол")
        print_submatrix(0, 10, cols - 10, cols, "Верхний правый угол")
        print_submatrix(rows - 10, rows, 0, 10, "Нижний левый угол")
        print_submatrix(rows - 10, rows, cols - 10, cols, "Нижний правый угол")

        if rows >= 20:
            print_submatrix(0, 10, cols // 2 - 5, cols // 2 + 5, "Середина первых 10 строк")
            print_submatrix(rows - 10, rows, cols // 2 - 5, cols // 2 + 5, "Середина последних 10 строк")
        else:
            print_submatrix(0, rows, cols // 2 - 5, cols // 2 + 5, "Середина всех строк")


def visualize_matrix(matrix):
    rows, cols = matrix.shape
    plt.figure(figsize=(18, 12))  # Увеличено для большего количества подграфиков

    if rows <= 50 and cols <= 50:
        dense = matrix.toarray()
        plt.imshow(dense, cmap='viridis', interpolation='nearest')
        for i in range(rows):
            for j in range(cols):
                plt.text(j, i, str(dense[i, j]), ha='center', va='center', color='white', fontsize=8)
        plt.title("Matrix Visualization with Numbers")
        plt.colorbar()
        plt.show()
    else:
        print("Matrix too large. Showing key sections.")
        fig, axs = plt.subplots(3, 3, figsize=(18, 12))  # 3x3 grid

        middle_rows_start = rows // 2 - 5
        middle_rows_end = rows // 2 + 5
        middle_cols_start = cols // 2 - 5
        middle_cols_end = cols // 2 + 5

        submatrices = [
            (matrix[:10, :10], "Top-left"),
            (matrix[:10, middle_cols_start:middle_cols_end], "Middle of Top Rows"),
            (matrix[:10, -10:], "Top-right"),
            (matrix[middle_rows_start:middle_rows_end, :10], "Left of Middle Rows"),
            (matrix[middle_rows_start:middle_rows_end, middle_cols_start:middle_cols_end], "Center of Middle Rows"),
            (matrix[middle_rows_start:middle_rows_end, -10:], "Right of Middle Rows"),
            (matrix[-10:, :10], "Bottom-left"),
            (matrix[-10:, middle_cols_start:middle_cols_end], "Middle of Bottom Rows"),
            (matrix[-10:, -10:], "Bottom-right")
        ]

        for ax, (submatrix, title) in zip(axs.flatten(), submatrices):
            dense_submatrix = submatrix.toarray()
            ax.imshow(dense_submatrix, cmap='viridis')
            for i in range(dense_submatrix.shape[0]):
                for j in range(dense_submatrix.shape[1]):
                    ax.text(j, i, str(dense_submatrix[i, j]), ha='center', va='center', color='white', fontsize=8)
            ax.set_title(title)

        plt.tight_layout()
        plt.show()


def cyclic_shift_crs(matrix):
    rows, cols = matrix.shape
    coo = matrix.tocoo()
    row = coo.row.astype(np.int64)
    col = coo.col.astype(np.int64)

    flat_pos = row * np.int64(cols) + col
    total_elements = np.int64(rows) * np.int64(cols)

    new_pos = (flat_pos + 1) % total_elements
    new_row = new_pos // cols
    new_col = new_pos % cols

    shifted = csr_matrix((coo.data, (new_row, new_col)), shape=(rows, cols))
    return shifted


def performance_test():
    sizes = [100000]
    densities = [0.1]
    filename = "performance_results.txt"

    with open(filename, "w") as f:
        f.write("=== Performance Test Results ===\n")
        for density in densities:
            f.write(f"\n== Density: {density} ==\n")
            for size in sizes:
                try:
                    count = int(size * size * density)
                    if count == 0:
                        result = f"{size}x{size} - The density is too low.\n"
                        f.write(result)
                        continue

                    rows = np.random.randint(0, size, count)
                    cols = np.random.randint(0, size, count)
                    data = np.random.randint(1, 100, count)
                    matrix = csr_matrix((data, (rows, cols)), shape=(size, size))

                    start = time.time()
                    _ = cyclic_shift_crs(matrix)
                    end = time.time()

                    mem_used = psutil.Process(os.getpid()).memory_info().rss / 1024
                    elapsed_us = round((end - start) * 1_000_000, 2)
                    result = f"{size}x{size} - Time: {elapsed_us} us, Memory: {round(mem_used, 2)} KB\n"
                except MemoryError:
                    result = f"{size}x{size} - Error memory\n"
                except OverflowError:
                    result = f"{size}x{size} - Overflow\n"

                f.write(result)

    print(f"Результаты сохранены в файл: {filename}")


def manual_input():
    rows = int(input("Число строк: "))
    cols = int(input("Число столбцов: "))
    count = int(input("Количество ненулевых элементов: "))

    data = []
    r = []
    c = []

    for i in range(count):
        print(f"{i+1}/{count}")
        row = int(input("Введите строку (0-индексация): "))
        col = int(input("Введите столбец (0-индексация): "))
        val = int(input("Введите значение: "))
        r.append(row)
        c.append(col)
        data.append(val)

    matrix = csr_matrix((data, (r, c)), shape=(rows, cols))
    return matrix


def random_matrix():
    rows = int(input("Число строк: "))
    cols = int(input("Число столбцов: "))
    density = float(input("Плотность (0..1): "))
    count = int(rows * cols * density)

    if count == 0:
        print("Слишком малая плотность, нет ненулевых элементов.")
        return csr_matrix((rows, cols))

    r = np.random.randint(0, rows, count)
    c = np.random.randint(0, cols, count)
    data = np.random.randint(1, 100, count)
    matrix = csr_matrix((data, (r, c)), shape=(rows, cols))
    return matrix


def main():
    matrix = None
    while True:
        print("\n1. Создать случайную матрицу или вручную")
        print("2. Загрузить матрицу из файла")
        print("3. Сохранить матрицу в файл")
        print("4. Вывести матрицу")
        print("5. Визуализировать матрицу")
        print("6. Циклический сдвиг матрицы")
        print("7. Тест производительности")
        print("8. Выход")

        choice = input("Выбор: ")

        if choice == '1':
            mode = input("1 - случайная, 2 - вручную: ")
            if mode == '1':
                matrix = random_matrix()
            elif mode == '2':
                matrix = manual_input()
        elif choice == '2':
            file_path = input("Введите путь к файлу: ")
            matrix = load_matrix(file_path)
            print("Загружена матрица.")
        elif choice == '3':
            if matrix is not None:
                save_matrix(matrix)
                print("Сохранено.")
            else:
                print("Матрица не загружена.")
        elif choice == '4':
            if matrix is not None:
                print_sparse_matrix(matrix)
            else:
                print("Матрица не загружена.")
        elif choice == '5':
            if matrix is not None:
                visualize_matrix(matrix)
            else:
                print("Матрица не загружена.")
        elif choice == '6':
            if matrix is not None:
                try:
                    start = time.time()
                    process = psutil.Process(os.getpid())
                    mem_before = process.memory_info().rss

                    matrix = cyclic_shift_crs(matrix)

                    mem_after = process.memory_info().rss
                    end = time.time()

                    elapsed_us = round((end - start) * 1_000_000, 2)
                    mem_used_kb = round((mem_after - mem_before) / 1024, 2)

                    print(f"Сдвиг выполнен. Время: {elapsed_us} мкс. Память: {mem_used_kb} KB.")
                except MemoryError:
                    print("Недостаточно памяти для сдвига.")
                except OverflowError:
                    print("Слишком большая матрица. Переполнение целочисленного значения.")
            else:
                print("Матрица не загружена.")
        elif choice == '7':
            performance_test()
        elif choice == '8':
            break
        else:
            print("Неверный выбор.")


if __name__ == '__main__':
    main()
