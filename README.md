# Deeplearning4j Benchmarks

Benchmarks popular models and configurations on Deeplearning4j, and output performance and versioning statistics.

#### Core Benchmarks

* BenchmarkCifar: uses the CIFAR-10 dataset to benchmark CNN models
    * MLP: using simple, single layer feed forward with MNIST data 
    * Lenet: using common LeNet CNN model with MNIST data
* BenchmarkCustom: user-provided datasets for benchmarking CNN models

## Top Benchmarks

The following benchmarks have been run using the SNAPSHOT version of DL4J 0.8.1.
This version utilizes workspace concepts and is significantly faster for inference
than 0.8.0. The number of labels used for benchmarks was 1000. Note that for full
training iteration timings, the number of labels and batch size impacts updater timing.

### AlexNet 16x3x224x224

The AlexNet batch 16 benchmark below was developed as a comparison 
to: https://github.com/jcjohnson/cnn-benchmarks. Note that the linked benchmarks do not provide 
values for training iterations.

DL4J summary (milliseconds):

| Forward | Backward | Total  |  Training Iteration |
|---|---|---|---|
|  2 | 8  | 10  | 17.98  |

Full versioning and statistics:

```
                        Name                                       ALEXNET
                 Description                     SIMULATEDCNN 16x3x224x224
            Operating System                  GNU/Linux Ubuntu 16.04.2 LTS
                     Devices              TITAN X (Pascal) 6 1 12779978752
                   CPU Cores                                            12
                     Backend                                          CUDA
                 BLAS Vendor                                        CUBLAS
                CUDA Version                                          8000
               CUDNN Version                                          6020
                Total Params                                      24400680
                Total Layers                                            11
        Avg Feedforward (ms)                                             2
           Avg Backprop (ms)                                             8
          Avg Iteration (ms)                                         17.98
             Avg Samples/sec                                        866.57
             Avg Batches/sec                                         54.16
```

### AlexNet 128x3x224x224

The AlexNet batch 128 benchmark is a comparison to benchmarks on popular
CNNs: https://github.com/soumith/convnet-benchmarks. Note that the linked benchmarks do
not provide values for training iterations.

DL4J summary (milliseconds):

| Forward | Backward | Total  |  Training Iteration |
|---|---|---|---|
|  11 | 39  | 50 | 65.73  |

Full versioning and statistics:

```
                        Name                                       ALEXNET
                 Description                    SIMULATEDCNN 128x3x224x224
            Operating System                  GNU/Linux Ubuntu 16.04.2 LTS
                     Devices              TITAN X (Pascal) 6 1 12779978752
                   CPU Cores                                            12
                     Backend                                          CUDA
                 BLAS Vendor                                        CUBLAS
                CUDA Version                                          8000
               CUDNN Version                                          6020
                Total Params                                      24400680
                Total Layers                                            11
        Avg Feedforward (ms)                                            11
           Avg Backprop (ms)                                            39
          Avg Iteration (ms)                                         65.73
             Avg Samples/sec                                       1895.18
             Avg Batches/sec                                         14.81
```

## Running Benchmarks

Each core benchmark class uses specific parameters. You must build this repository before running benchmarks.

* First build using `mvn package -DskipTests`.
* Then run specific benchmark class such as `java -cp /path/to/the.jar BenchmarkCifar -batch 128 -model ALEXNET`.

## Contributing

Contributions are welcome. Please see https://deeplearning4j.org/devguide.
