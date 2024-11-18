from mpi4py import MPI
import numpy as np
import pandas as pd
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

# Load Fashion-MNIST dataset from a CSV file
def load_fashion_mnist_csv(csv_path):
    data = pd.read_csv(csv_path)
    labels = data.iloc[:, 0].values
    images = data.iloc[:, 1:].values.reshape(-1, 28, 28).astype(np.uint8)
    return images, labels

# Load and preprocess an image using OpenCV
def load_and_preprocess_image(image_path, size=(28, 28)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    image = cv2.resize(image, size)
    return image

# Main function to compare test image with dataset images
def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Load the training dataset
    if rank == 0:
        train_csv_path = "fashion-mnist_train.csv"
        test_image_path = "61d35yWHQmL._AC_UY1100_.jpg"
        train_images, train_labels = load_fashion_mnist_csv(train_csv_path)
        test_image = load_and_preprocess_image(test_image_path)
    else:
        train_images = None
        train_labels = None
        test_image = None

    train_images = comm.bcast(train_images, root=0)
    train_labels = comm.bcast(train_labels, root=0)
    test_image = comm.bcast(test_image, root=0)

    local_train_images = np.array_split(train_images, size)[rank]
    local_train_labels = np.array_split(train_labels, size)[rank]
    local_results = []

    for train_image in local_train_images:
        local_result = np.array([0], dtype=np.int64)
        lock = threading.Lock()

        # Process rows in parallel using threads
        process_rows_in_parallel(test_image, train_image, 0, test_image.shape[0], local_result, lock, num_threads=4)

        global_sum = comm.reduce(local_result[0], op=MPI.SUM, root=0)

        if rank == 0:
            N, M = test_image.shape
            if global_sum >= 0 and N * M > 0:
                Ed2 = math.sqrt(global_sum / (N * M))
                percentage_distance = (Ed2 / (N * M)) * 100
                local_results.append(percentage_distance)
            else:
                print(f"Error: Invalid values for square root calculation. Global Sum: {global_sum}, Elements Count: {N * M}")

    global_results = comm.gather(local_results, root=0)
    global_train_labels = comm.gather(local_train_labels, root=0)

    if rank == 0:
        all_results = [item for sublist in global_results for item in sublist]
        all_labels = [item for sublist in global_train_labels for item in sublist]
        min_value = min(all_results)
        min_index = all_results.index(min_value)

        # Confidence Calculation
        confidence = 1 - (min_value / sum(all_results) * len(all_results))

        print(f"Percentage Distance Value: {all_results[min_index]}")
        print(f"Minimum Percentage Distance: {min_value}")
        print(f"Sum of Percentage Distances: {sum(all_results)}")
        print(f"Number of Training Elements: {len(all_results)}")
        print(f"Confidence of Result: {confidence}")

if __name__ == "__main__":
    main()
