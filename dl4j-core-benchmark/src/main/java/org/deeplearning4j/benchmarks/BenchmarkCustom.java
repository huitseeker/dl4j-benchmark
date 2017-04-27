package org.deeplearning4j.benchmarks;

import lombok.extern.slf4j.Slf4j;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.filters.PathFilter;
import org.datavec.api.io.filters.RandomPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.ResizeImageTransform;
import org.deeplearning4j.datasets.datavec.ParallelRecordReaderDataSetIterator;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.ParallelExistingMiniBatchDataSetIterator;
import org.deeplearning4j.models.ModelType;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.dataset.ExistingMiniBatchDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.util.Random;

/**
 * Benchmarks popular CNN models using the CIFAR-10 dataset.
 */
@Slf4j
public class BenchmarkCustom extends BaseBenchmark {

    // values to pass in from command line when compiled, esp running remotely
    @Option(name = "--modelType", usage = "Model type (e.g. ALEXNET, VGG16, or CNN).", aliases = "-model")
    public static ModelType modelType = ModelType.ALEXNET;
    //    @Option(name="--numGPUs",usage="How many workers to use for multiple GPUs.",aliases = "-ng")
//    public int numGPUs = 0;
    @Option(name="--datasetPath",usage="Path to the parent directly of multiple directories of classes of images.",aliases = "-dataset")
    public static String datasetPath = null;
    @Option(name="--numLabels",usage="Train batch size.",aliases = "-labels")
    public static int numLabels = -1;
    @Option(name="--trainBatchSize",usage="Train batch size.",aliases = "-batch")
    public static int trainBatchSize = 128;
    @Option(name="--deviceCache",usage="Set CUDA device cache.",aliases = "-dcache")
    public static long deviceCache = 6L;
    @Option(name="--hostCache",usage="Set CUDA host cache.",aliases = "-hcache")
    public static long hostCache = 12L;
    @Option(name="--gcThreads",usage="Set Garbage Collection threads.",aliases = "-gcthreads")
    public static int gcThreads = 5;
    @Option(name="--gcWindow",usage="Set Garbage Collection window in milliseconds.",aliases = "-gcwindow")
    public static int gcWindow = 5000;
    @Option(name="--inputDimension",usage="The height and width of the dataset",aliases = "-dim")
    public static int inputDimension = 224;
    @Option(name="--resizeDimension",usage="Set Garbage Collection window in milliseconds.",aliases = "-resize")
    public static int resizeDimension = inputDimension;
    @Option(name="--profile",usage="Run profiler and print results",aliases = "-profile")
    public static boolean profile = false;

    private int height = 224;
    private int width = 224;
    private int channels = 3;
    private String datasetName  = "CUSTOM";
    private int seed = 42;

    public void run(String[] args) throws Exception {
        // Parse command line arguments if they exist
        CmdLineParser parser = new CmdLineParser(this);
        try {
            parser.parseArgument(args);
        } catch (CmdLineException e) {
            // handling of wrong arguments
            System.err.println(e.getMessage());
            parser.printUsage(System.err);
        }

        this.height = resizeDimension;
        this.width = resizeDimension;

        // memory management optimizations
//        CudaEnvironment.getInstance().getConfiguration()
//                // key option enabled
//                .allowMultiGPU(true)
//                .allowCrossDeviceAccess(false)
//                // we're allowing larger memory caches
//                .setMaximumDeviceCache(0L * 1024L * 1024L * 1024L)
//                .setMaximumHostCache(0L * 1024L * 1024L * 1024L)
//                .setNumberOfGcThreads(5)
//                .setNoGcWindowMs(gcWindow);

        Nd4j.create(1);
        Nd4j.getMemoryManager().togglePeriodicGc(false);
        Nd4j.getMemoryManager().setAutoGcWindow(gcWindow);
        Nd4j.getMemoryManager().setOccasionalGcFrequency(0);

        if(modelType == ModelType.ALL || modelType == ModelType.RNN)
            throw new UnsupportedOperationException("Image benchmarks are applicable to CNN models only.");

        if(datasetPath==null)
            throw new IllegalArgumentException("You must specify a valid path to a labelled dataset of images.");

        log.info("Loading data...");

        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        File mainPath = new File(datasetPath);
        FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, new Random(seed));

        PathFilter pathFilter;
        if(numLabels>-1)
            pathFilter = new BalancedPathFilter(new Random(seed), labelMaker, 1000000, numLabels, 6000);
        else
            pathFilter = new RandomPathFilter(new Random(seed), NativeImageLoader.ALLOWED_FORMATS);

        InputSplit[] split = fileSplit.sample(pathFilter, 1.0);
        ImageTransform resize = new ResizeImageTransform(resizeDimension, resizeDimension);
        RecordReader trainRR = new ImageRecordReader(inputDimension, inputDimension, channels, labelMaker, resize);
        trainRR.initialize(split[0]);
        DataSetIterator iter = new ParallelRecordReaderDataSetIterator.Builder(trainRR)
                .setBatchSize(trainBatchSize)
                .setNumberOfPossibleLabels(trainRR.getLabels().size())
                .numberOfWorkers(2)
                .prefetchBufferSize(4)
                .build();

        log.info("Preparing benchmarks for "+split[0].locations().length+" images, "+iter.getLabels().size()+" labels");

        benchmark(height, width, channels, trainRR.getLabels().size(), trainBatchSize, seed, datasetName, iter, modelType, profile);

//        DataSetIterator iter = new ParallelExistingMiniBatchDataSetIterator(new File("/home/justin/Datasets/umdfaces_aligned_224_presave_train/"),"presave-train-%d.bin", 2, 8, true);
//        benchmark(height, width, channels, 8644, trainBatchSize, seed, datasetName, iter, modelType, profile);



        System.exit(0);
    }

    public static void main(String[] args) throws Exception {
        new BenchmarkCustom().run(args);
    }
}
