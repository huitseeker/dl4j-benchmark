# Deeplearning4j Benchmarks

Benchmarks popular models and configurations on Deeplearning4j, and output performance and versioning statistics.

#### Core Benchmarks

* BenchmarkCnn: simulates input like most benchmarking tools
    * Comes with variety of convolutional models including VGG-16, LeNet, and AlexNet
* BenchmarkCifar: uses the CIFAR-10 dataset to benchmark CNN models
    * MLP: using simple, single layer feed forward with MNIST data 
    * Lenet: using common LeNet CNN model with MNIST data
* BenchmarkCustom: user-provided datasets for benchmarking CNN models

## Top Benchmarks

The following benchmarks have been run using the SNAPSHOT version of DL4J 0.9.1.
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
|  2 | 5.01  | 7.01  | 14.33  |

Full versioning and statistics:

```
                        Name                                       ALEXNET
                 Description                            VGG16 16x3x224x224
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
           Avg Backprop (ms)                                          5.01
          Avg Iteration (ms)                                         14.33
             Avg Samples/sec                                       1075.93
             Avg Batches/sec                                         67.25
```

### AlexNet 128x3x224x224

The AlexNet batch 128 benchmark is a comparison to benchmarks on popular
CNNs: https://github.com/soumith/convnet-benchmarks. Note that the linked benchmarks do
not provide values for training iterations.

DL4J summary (milliseconds):

| Forward | Backward | Total  |  Training Iteration |
|---|---|---|---|
|  10 | 33.32  | 43.32 | 58.58  |

Full versioning and statistics:

```
                        Name                                       ALEXNET
                 Description                           ALEXNET 128x3x224x224
            Operating System                  GNU/Linux Ubuntu 16.04.2 LTS
                     Devices              TITAN X (Pascal) 6 1 12779978752
                   CPU Cores                                            12
                     Backend                                          CUDA
                 BLAS Vendor                                        CUBLAS
                CUDA Version                                          8000
               CUDNN Version                                          6020
                Total Params                                      24400680
                Total Layers                                            11
        Avg Feedforward (ms)                                            10
           Avg Backprop (ms)                                         33.32
          Avg Iteration (ms)                                         58.58
             Avg Samples/sec                                        2098.4
             Avg Batches/sec                                         16.39
```

## LeNet 16x3x224x224

DL4J summary (milliseconds):

| Forward | Backward | Total  |  Training Iteration |
|---|---|---|---|
|  5 | 18.02  | 23.02 | 35.28  |

Full versioning and statistics:

```
                        Name                                         LENET
                 Description                            LENET 16x3x224x224
            Operating System                  GNU/Linux Ubuntu 16.04.2 LTS
                     Devices              TITAN X (Pascal) 6 1 12779978752
                   CPU Cores                                            12
                     Backend                                          CUDA
                 BLAS Vendor                                        CUBLAS
                CUDA Version                                          8000
               CUDNN Version                                          6020
                Total Params                                      70753070
                Total Layers                                             6
        Avg Feedforward (ms)                                             5
           Avg Backprop (ms)                                         18.02
          Avg Iteration (ms)                                         35.28
             Avg Samples/sec                                        435.66
             Avg Batches/sec                                         27.23
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
|  44.24 | 129.04  | 173.28 | 178.39  |

Full versioning and statistics:

```
                        Name                                         VGG16
                 Description                            VGG16 16x3x224x224
            Operating System                  GNU/Linux Ubuntu 16.04.2 LTS
                     Devices              TITAN X (Pascal) 6 1 12779978752
                   CPU Cores                                            12
                     Backend                                          CUDA
                 BLAS Vendor                                        CUBLAS
                CUDA Version                                          8000
               CUDNN Version                                          6020
                Total Params                                      39803688
                Total Layers                                            19
        Avg Feedforward (ms)                                         44.24
           Avg Backprop (ms)                                        129.04
          Avg Iteration (ms)                                        178.39
             Avg Samples/sec                                         86.15
             Avg Batches/sec                                          5.38
```


## Running Benchmarks

Each core benchmark class uses specific parameters. You must build this repository before running benchmarks.

* First build using `mvn package -DskipTests`.
* Then run specific benchmark class such as `java -cp /path/to/the.jar BenchmarkCnn -batch 128 -model ALEXNET`.

## Contributing

Contributions are welcome. Please see https://deeplearning4j.org/devguide.
