# Deeplearning4j Benchmarks

Benchmarks popular models and configurations on Deeplearning4j, and output performance and versioning statistics.

#### Core Benchmarks

* BenchmarkCifar: uses the CIFAR-10 dataset to benchmark CNN models
    * MLP: using simple, single layer feed forward with MNIST data 
    * Lenet: using common LeNet CNN model with MNIST data
* BenchmarkCustom: user-provided datasets for benchmarking CNN models

## Top Benchmarks

The following benchmarks have been run using the SNAPSHOT version of DL4J 0.8.1. This version utilizes workspace concepts and is significantly faster than 0.8.0.

### AlexNet 16x3x224x224

The AlexNet batch 16 benchmark below was developed as a comparison to: https://github.com/jcjohnson/cnn-benchmarks.

DL4J summary:

| Forward | Backward | Total  |  Training Iteration |
|---|---|---|---|
|  0.44 | 2.1  | 2.54  | 54.01  |

Full versioning and statistics:

```
                        Name                                       ALEXNET
                 Description                         CIFAR-10 16x3x224x224
            Operating System                  GNU/Linux Ubuntu 16.04.2 LTS
                     Devices              TITAN X (Pascal) 6 1 12779978752
                   CPU Cores                                            12
                     Backend                                          CUDA
                 BLAS Vendor                                        CUBLAS
                CUDA Version                                          8000
               CUDNN Version                                          6020
                Total Params                                      20344650
                Total Layers                                            11
        Avg Feedforward (ms)                                          0.44
           Avg Backprop (ms)                                           2.1
          Avg Iteration (ms)                                         54.01
             Avg Samples/sec                                        290.71
             Avg Batches/sec                                         18.17
```

### AlexNet 128x3x224x224

The AlexNet batch 128 benchmark is a comparison to benchmarks on popular CNNs: https://github.com/soumith/convnet-benchmarks.

DL4J summary:

| Forward | Backward | Total  |  Training Iteration |
|---|---|---|---|
|  0.54 | 6.78  | 7.32  | 69.08  |

Full versioning and statistics:

```
                        Name                                       ALEXNET
                 Description                        CIFAR-10 128x3x224x224
            Operating System                  GNU/Linux Ubuntu 16.04.2 LTS
                     Devices              TITAN X (Pascal) 6 1 12779978752
                   CPU Cores                                            12
                     Backend                                          CUDA
                 BLAS Vendor                                        CUBLAS
                CUDA Version                                          8000
               CUDNN Version                                          6020
                Total Params                                      20344650
                Total Layers                                            11
        Avg Feedforward (ms)                                          0.54
           Avg Backprop (ms)                                          6.78
          Avg Iteration (ms)                                         69.08
             Avg Samples/sec                                       1572.51
             Avg Batches/sec                                         12.29
```

## Running Benchmarks

Each core benchmark class uses specific parameters. You must build this repository before running benchmarks.

* First build using `mvn package -DskipTests`.
* Then run specific benchmark class such as `java -cp /path/to/the.jar BenchmarkCifar -batch 128 -model ALEXNET`.

## Contributing

Contributions are welcome. Please see https://deeplearning4j.org/devguide.
