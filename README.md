# Cross-Domain_Distill

This repository provides the official codebase for our paper:

**Title:**  
_Real-Time Distributed Optical Fiber Vibration Recognition via Extreme Lightweight Model and Cross-Domain Distillation_

## Overview

This project presents an ultra-lightweight, depthwise separable convolutional neural network (DSCNN-3) combined with a novel cross-domain knowledge distillation framework designed for large-scale, real-time distributed optical fiber vibration sensing (DVS) systems.

Key contributions:
- A compact 3-layer DSCNN model with only **4,141 parameters** optimized for resource-constrained platforms.
- A **physically-guided distillation framework** integrating frequency-domain priors into time-domain learning.
- An FPGA implementation using **shift-add quantization**, achieving **13,494 samples/sec** processing rate over **168.68 km** sensing fiber.
- Experimental results demonstrating **95.72% accuracy** under unseen environmental conditions.

## Repository content

This repository provides the necessary codes for the evaluation and simulation of the proposed scheme. It includes:

- **Testing Codes**: You will find codes for testing the efficiency and accuracy of the distributed optical fiber vibration sensing algorithm.

- **Example Samples**: We have included example samples for you to experiment with and evaluate the performance of the algorithm.

- **Trained Models**: Pre-trained models are provided for easy evaluation and testing purposes.

The complete dataset for training and evaluation is available on Google Drive. Please access the dataset [here](https://drive.google.com/drive/folders/1LK-k0a7M_M6h3VveUCc4wbd_T7ONR1Gb?usp=sharing).

Please note that the training codes, and hardware design codes will be made available once the associated article is published.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions, suggestions, or collaboration, please contact:

**Dr. Hao Wu**
Huazhong University of Science and Technology
Email: \[[wuhaoboom@hust.edu.cn](mailto:wuhaoboom@hust.edu.cn)]
