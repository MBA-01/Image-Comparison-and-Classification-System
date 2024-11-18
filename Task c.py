from mpi4py import MPI
import numpy as np
import threading
import math

def pixel_difference(x, y):
    """Calculate the squared difference between two pixel values."""
    return (x - y) ** 2

def process_row(X_row, Y_row):
    """Process a single row to calculate the sum of squared differences."""
    return np.sum([pixel_difference(x, y) for x, y in zip(X_row, Y_row)], dtype=np.int64)

def sum_of_differences(X, Y, result, index_range, lock):
    """Calculate the sum of squared differences for a portion of the matrix using multiple threads."""
    local_sum = 0
    for i in range(index_range[0], index_range[1]):
        local_sum += process_row(X[i], Y[i])
    with lock:
        result[0] += local_sum

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        X = np.random.randint(0, 256, (1000, 1000), dtype=np.int64)
        Y = np.random.randint(0, 256, (1000, 1000), dtype=np.int64)
    else:
        X = None
        Y = None

    X = comm.bcast(X, root=0)
    Y = comm.bcast(Y, root=0)

    n_rows = X.shape[0]
    local_n_rows = n_rows // size
    start_row = rank * local_n_rows
    end_row = start_row + local_n_rows if rank != size - 1 else n_rows

    num_threads = 4
    threads = []
    local_result = np.array([0], dtype=np.int64)
    lock = threading.Lock()

    rows_per_thread = (end_row - start_row) // num_threads
    for i in range(num_threads):
        t_start_row = start_row + i * rows_per_thread
        t_end_row = t_start_row + rows_per_thread if i != num_threads - 1 else end_row
        thread = threading.Thread(target=sum_of_differences, args=(X, Y, local_result, (t_start_row, t_end_row), lock))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    global_sum = comm.reduce(local_result[0], op=MPI.SUM, root=0)

    if rank == 0:
        N, M = X.shape
        if global_sum >= 0 and N * M > 0:
            Ed2 = math.sqrt(global_sum / (N * M))
            print(f"Global Sum of Squared Differences: {global_sum}")
            print(f"Ed2: {Ed2}")
        else:
            print(f"Error: Invalid values for square root calculation. Global Sum: {global_sum}, Elements Count: {N * M}")

if __name__ == "__main__":
    main()
