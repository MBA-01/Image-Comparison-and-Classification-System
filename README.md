# Image Comparison and Classification System

## Overview
This project implements a distributed image comparison and classification system using MPI (Message Passing Interface) and multi-threading. It's designed to compare images against a dataset (specifically the Fashion-MNIST dataset) and determine similarities using pixel-wise comparisons.

Check the Project Report for detailed documentation.[Report](https://github.com/MBA-01/Image-Comparison-and-Classification-System/blob/18aa717eee018cec63bf39bf5aed8d63184ac7ec/Image%20Classification%20and%20Parallel%20Computation.pdf)

## Key Features
- Distributed processing using MPI
- Multi-threaded computation for improved performance
- Support for Fashion-MNIST dataset processing
- Custom image comparison capabilities
- Confidence score calculation for classifications

## Technical Implementation
The system uses multiple levels of parallelization:
1. **MPI Distribution**: Splits workload across multiple machines/processes
2. **Multi-threading**: Each MPI process utilizes multiple threads for computation
3. **Numpy Optimization**: Efficient array operations for pixel comparisons

## Components

### 1. Image Processing
- Loads and preprocesses images using OpenCV
- Supports grayscale image processing
- Automatic image resizing to match dataset dimensions (28x28)

### 2. Distance Calculation
- Implements pixel-wise squared difference calculation
- Uses thread-safe operations for parallel processing
- Handles numerical overflow through appropriate data types

### 3. Dataset Management
- Supports Fashion-MNIST CSV format
- Efficient data distribution across MPI processes
- Label management for classification tasks

## Benefits

1. **Performance**
   - Distributed processing reduces computation time
   - Multi-threading utilizes available CPU cores efficiently
   - Optimized for large dataset processing

2. **Scalability**
   - Easily scales across multiple machines
   - Handles varying dataset sizes
   - Adaptable thread count based on system capabilities

3. **Accuracy**
   - Precise pixel-level comparison
   - Confidence score calculation
   - Overflow protection for large calculations

## Usage

### Prerequisites

bash
```pip install mpi4py numpy pandas opencv-python```


### Running the Program

bash
```mpiexec -n <number_of_processes> python algo_6.py```



### Input Requirements
- Fashion-MNIST training dataset in CSV format
- Test image in common format (jpg, png, etc.)
- Both inputs should be in the same directory as the script

## Output
The program provides:
- Percentage distance values
- Minimum distance to closest match
- Confidence scores
- Processing statistics

## Use Cases
1. Fashion item classification
2. Image similarity detection
3. Dataset comparison
4. Quality control in image processing
5. Pattern recognition systems

## Technical Notes
- Optimal performance with 4 threads per MPI process
- Supports grayscale images
- Automatic image resizing to 28x28 pixels
- Uses int64 data type to prevent overflow

## Security
- Input validation for all file operations
- Thread-safe operations with proper locking
- Memory overflow protection
- Error handling for invalid inputs
- Secure file path handling

## Limitations
- Currently optimized for Fashion-MNIST format
- Requires grayscale images
- Memory usage scales with dataset size
- Requires MPI environment setup

## Future Improvements
1. Support for color images
2. Dynamic thread allocation
3. Additional distance metrics
4. GPU acceleration support
5. Real-time processing capabilities

## Contributing
1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request


## License
This project is licensed under the MIT License - see the LICENSE file for details.
