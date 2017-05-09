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
CUDA_VISIBLE_DEVICES has been set to 1.

### AlexNet 16x3x224x224

The AlexNet batch 16 benchmark below was developed as a comparison 
to: https://github.com/jcjohnson/cnn-benchmarks. Note that the linked benchmarks do not provide 
values for training iterations.

DL4J summary (milliseconds):

| Forward | Backward | Total  |  Training Iteration |
|---|---|---|---|
|  2 | 8  | 10  | 16.68  |

Full versioning and statistics:

```
                        Name                                       ALEXNET
                 Description                     SIMULATEDCNN 16x3x224x224
            Operating System                  GNU/Linux Ubuntu 16.04.2 LTS
                     Devices              TITAN X (Pascal) 6 1 12782075904
                   CPU Cores                                            12
                     Backend                                          CUDA
                 BLAS Vendor                                        CUBLAS
                CUDA Version                                          8000
               CUDNN Version                                          6020
                Total Params                                      24400680
                Total Layers                                            11
        Avg Feedforward (ms)                                             2
           Avg Backprop (ms)                                             8
          Avg Iteration (ms)                                         16.68
             Avg Samples/sec                                        935.17
             Avg Batches/sec                                         58.45
```

### AlexNet 128x3x224x224

The AlexNet batch 128 benchmark is a comparison to benchmarks on popular
CNNs: https://github.com/soumith/convnet-benchmarks. Note that the linked benchmarks do
not provide values for training iterations.

DL4J summary (milliseconds):

| Forward | Backward | Total  |  Training Iteration |
|---|---|---|---|
|  11 | 37.96  | 48.96 | 63.11  |

Full versioning and statistics:

                        Name                                       ALEXNET
                 Description                    SIMULATEDCNN 128x3x224x224
            Operating System                  GNU/Linux Ubuntu 16.04.2 LTS
                     Devices              TITAN X (Pascal) 6 1 12782075904
                   CPU Cores                                            12
                     Backend                                          CUDA
                 BLAS Vendor                                        CUBLAS
                CUDA Version                                          8000
               CUDNN Version                                          6020
                Total Params                                      24400680
                Total Layers                                            11
        Avg Feedforward (ms)                                            11
           Avg Backprop (ms)                                         37.96
          Avg Iteration (ms)                                         63.11
             Avg Samples/sec                                       1973.71
             Avg Batches/sec                                         15.42
```

## LeNet 16x3x224x224

DL4J summary (milliseconds):

| Forward | Backward | Total  |  Training Iteration |
|---|---|---|---|
|  12 | 25  | 37 | 44.75  |

Full versioning and statistics:

```
                        Name                                         LENET
                 Description                     SIMULATEDCNN 16x3x224x224
            Operating System                  GNU/Linux Ubuntu 16.04.2 LTS
                     Devices              TITAN X (Pascal) 6 1 12782075904
                   CPU Cores                                            12
                     Backend                                          CUDA
                 BLAS Vendor                                        CUBLAS
                CUDA Version                                          8000
               CUDNN Version                                          6020
                Total Params                                      70753070
                Total Layers                                             6
        Avg Feedforward (ms)                                            12
           Avg Backprop (ms)                                            25
          Avg Iteration (ms)                                         44.75
             Avg Samples/sec                                        350.83
             Avg Batches/sec                                         21.93
```

## LeNet 128x3x224x224

DL4J summary (milliseconds):

| Forward | Backward | Total  |  Training Iteration |
|---|---|---|---|
|  28.13 | 130.4  | 158.17 | 164.24  |

Full versioning and statistics:

```
                        Name                                         LENET
                 Description                    SIMULATEDCNN 128x3x224x224
            Operating System                  GNU/Linux Ubuntu 16.04.2 LTS
                     Devices              TITAN X (Pascal) 6 1 12782075904
                   CPU Cores                                            12
                     Backend                                          CUDA
                 BLAS Vendor                                        CUBLAS
                CUDA Version                                          8000
               CUDNN Version                                          6020
                Total Params                                      70753070
                Total Layers                                             6
        Avg Feedforward (ms)                                         28.13
           Avg Backprop (ms)                                         130.4
          Avg Iteration (ms)                                        164.24
             Avg Samples/sec                                        758.82
             Avg Batches/sec                                          5.93
```

## VGG-16

DL4J summary (milliseconds):

This benchmark is analogous to VGG-16 Torch which is [available here](https://github.com/jcjohnson/cnn-benchmarks#vgg-16). The
model uses 1,000 classes/outputs. All available optimizations have been applied.

| Forward | Backward | Total  |  Training Iteration |
|---|---|---|---|
|  44.68 | 166.52  | 211.2 | 205.03  |

```
                        Name                                         VGG16
                 Description                     SIMULATEDCNN 16x3x224x224
            Operating System                  GNU/Linux Ubuntu 16.04.2 LTS
                     Devices              TITAN X (Pascal) 6 1 12782075904
                   CPU Cores                                            12
                     Backend                                          CUDA
                 BLAS Vendor                                        CUBLAS
                CUDA Version                                          8000
               CUDNN Version                                          6020
                Total Params                                      39803688
                Total Layers                                            19
        Avg Feedforward (ms)                                         44.68
           Avg Backprop (ms)                                        166.52
          Avg Iteration (ms)                                        205.03
             Avg Samples/sec                                         75.95
             Avg Batches/sec                                          4.75
```


## Running Benchmarks

Each core benchmark class uses specific parameters. You must build this repository before running benchmarks.

* First build using `mvn package -DskipTests`.
* Then run specific benchmark class such as `java -cp /path/to/the.jar BenchmarkCifar -batch 128 -model ALEXNET`.

## Contributing

Contributions are welcome. Please see https://deeplearning4j.org/devguide.
