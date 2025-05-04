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
        r_mid = rows // 2
        c_mid = cols // 2
        print_submatrix(r_mid - 5, r_mid + 5, c_mid - 5, c_mid + 5, "Центр")


def visualize_matrix(matrix):
    rows, cols = matrix.shape
    plt.figure(figsize=(8, 8))

    if rows <= 100 and cols <= 100:
        plt.spy(matrix, markersize=1)
        plt.title("Matrix Visualization")
        plt.show()
    else:
        print("Matrix too large. Showing corners and center.")
        dense = matrix.toarray()
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))

        axs[0, 0].imshow(dense[:10, :10], cmap='viridis')
        axs[0, 0].set_title('Top-left')

        axs[0, 1].imshow(dense[:10, -10:], cmap='viridis')
        axs[0, 1].set_title('Top-right')

        center_r, center_c = rows // 2, cols // 2
        axs[0, 2].imshow(dense[center_r-5:center_r+5, center_c-5:center_c+5], cmap='viridis')
        axs[0, 2].set_title('Center')

        axs[1, 0].imshow(dense[-10:, :10], cmap='viridis')
        axs[1, 0].set_title('Bottom-left')

        axs[1, 1].imshow(dense[-10:, -10:], cmap='viridis')
        axs[1, 1].set_title('Bottom-right')

        axs[1, 2].axis('off')

        plt.tight_layout()
        plt.show()


def cyclic_shift_crs(matrix):
    rows, cols = matrix.shape

    print("Матрица до сдвига:")
    print_sparse_matrix(matrix)

    coo = matrix.tocoo()
    row = coo.row.astype(np.int64)
    col = coo.col.astype(np.int64)

    # Вычисляем линейные индексы
    flat_pos = row * np.int64(cols) + col
    total_elements = np.int64(rows) * np.int64(cols)

    # Циклический сдвиг всех ненулевых элементов на один вперёд
    new_pos = (flat_pos + 1) % total_elements
    new_row = new_pos // cols
    new_col = new_pos % cols

    shifted = csr_matrix((coo.data, (new_row, new_col)), shape=(rows, cols))

    print("\nМатрица после сдвига:")
    print_sparse_matrix(shifted)

    return shifted


def performance_test():
    sizes = [10, 100, 500, 1000, 10000]
    for size in sizes:
        density = 0.01
        count = int(size * size * density)
        rows = np.random.randint(0, size, count)
        cols = np.random.randint(0, size, count)
        data = np.random.randint(1, 100, count)
        matrix = csr_matrix((data, (rows, cols)), shape=(size, size))

        start = time.time()
        shifted = cyclic_shift_crs(matrix)
        end = time.time()

        mem_used = psutil.Process(os.getpid()).memory_info().rss / (1024**2)

        print(f"{size}x{size} - Time: {round((end - start)*1000, 2)} ms, Memory: {round(mem_used, 2)} MB")


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
            matrix = load_matrix()
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
                    matrix = cyclic_shift_crs(matrix)
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
