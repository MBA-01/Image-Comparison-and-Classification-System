from mpi4py import MPI
import numpy as np
import threading
import math

# Pixel difference function that uses numpy to avoid overflow
def pixel_difference(x, y):
    return (x - y) ** 2

# Function to compute the sum of squared differences using multiple threads
def sum_of_differences(X, Y, result, index_range, lock):
    # Using numpy to sum up to avoid overflow issues
    local_sum = np.sum([pixel_difference(X[i], Y[i]) for i in range(index_range[0], index_range[1])])
    with lock:
        result[0] += local_sum

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        X = np.random.randint(0, 256, 1000000, dtype=np.int64)
        Y = np.random.randint(0, 256, 1000000, dtype=np.int64)
    else:
        X = None
        Y = None

    X = comm.bcast(X, root=0)
    Y = comm.bcast(Y, root=0)

    n = len(X)
    local_n = n // size
    start = rank * local_n
    end = start + local_n if rank != size - 1 else n

    num_threads = 4
    threads = []
    local_result = np.array([0], dtype=np.int64)
    lock = threading.Lock()

    thread_load = (end - start) // num_threads
    for i in range(num_threads):
        t_start = start + i * thread_load
        t_end = t_start + thread_load if i != num_threads - 1 else end
        thread = threading.Thread(target=sum_of_differences, args=(X, Y, local_result, (t_start, t_end), lock))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    global_sum = comm.reduce(local_result[0], op=MPI.SUM, root=0)

    if rank == 0:
        if global_sum < 0:
            print("Unexpected negative sum")
        else:
            if n > 0:
                Ed1 = math.sqrt(global_sum / n)
                print(f"Global Sum of Differences: {global_sum}")
                print(f"Ed1: {Ed1}")
            else:
                print("Error: No data to process.")

if __name__ == "__main__":
    main()
