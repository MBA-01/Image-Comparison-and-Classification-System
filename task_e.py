from mpi4py import MPI
import numpy as np
import threading
import math
import cv2

# Pixel difference function that uses numpy to avoid overflow
def pixel_difference(x, y):
    return (x - y) ** 2

# Function to compute the sum of squared differences using multiple threads
def sum_of_differences(X, Y, result, index_range, lock):
    local_sum = np.sum([pixel_difference(X[i], Y[i]) for i in range(index_range[0], index_range[1])], dtype=np.int64)
    with lock:
        result[0] += local_sum

# Function to process rows in parallel using threads
def process_rows_in_parallel(X, Y, start_row, end_row, result, lock, num_threads=4):
    threads = []
    rows_per_thread = (end_row - start_row) // num_threads
    for i in range(num_threads):
        t_start_row = start_row + i * rows_per_thread
        t_end_row = t_start_row + rows_per_thread if i != num_threads - 1 else end_row
        thread = threading.Thread(target=sum_of_differences, args=(X, Y, result, (t_start_row, t_end_row), lock))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

# Function to load images using OpenCV and preprocess them
def load_and_preprocess_image(image_url):
    image = cv2.imread(image_url, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image from {image_url}")
    return image

# Main function that uses MPI and Algorithm 3 for distributed computation
def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        # Load both images and ensure they are the same size
        image1 = load_and_preprocess_image("61d35yWHQmL._AC_UY1100_.jpg")
        image2 = load_and_preprocess_image("61LYRZ-uH6L._AC_UY1100_.jpg")
        
        # Resize both images to the smallest common size if they are different
        if image1.shape != image2.shape:
            common_shape = (min(image1.shape[0], image2.shape[0]), min(image1.shape[1], image2.shape[1]))
            image1 = cv2.resize(image1, common_shape)
            image2 = cv2.resize(image2, common_shape)
    else:
        image1 = None
        image2 = None

    # Broadcast images to all processes
    image1 = comm.bcast(image1, root=0)
    image2 = comm.bcast(image2, root=0)

    n_rows = image1.shape[0]
    local_n_rows = n_rows // size
    start_row = rank * local_n_rows
    end_row = start_row + local_n_rows if rank != size - 1 else n_rows

    local_result = np.array([0], dtype=np.int64)
    lock = threading.Lock()

    # Process rows in parallel using threads
    process_rows_in_parallel(image1, image2, start_row, end_row, local_result, lock)

    global_sum = comm.reduce(local_result[0], op=MPI.SUM, root=0)

    if rank == 0:
        N, M = image1.shape
        if global_sum >= 0 and N * M > 0:
            Ed2 = math.sqrt(global_sum / (N * M))
            percentage_distance = (Ed2 / (N * M)) * 100
            print(f"Global Sum of Squared Differences: {global_sum}")
            print(f"Ed2: {Ed2}")
            print(f"Percentage Distance Value: {percentage_distance}%")
        else:
            print(f"Error: Invalid values for square root calculation. Global Sum: {global_sum}, Elements Count: {N * M}")

if __name__ == "__main__":
    main()
