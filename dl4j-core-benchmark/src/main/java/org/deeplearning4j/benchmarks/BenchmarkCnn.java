package org.deeplearning4j.benchmarks;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.datasets.iterator.impl.BenchmarkDataSetIterator;
import org.deeplearning4j.models.ModelType;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Benchmarks popular CNN models using the CIFAR-10 dataset.
 */
@Slf4j
public class BenchmarkCnn extends BaseBenchmark {

    // values to pass in from command line when compiled, esp running remotely
    @Option(name = "--modelType", usage = "Model type (e.g. ALEXNET, VGG16, or CNN).", aliases = "-model")
    public static ModelType modelType = ModelType.ALEXNET;
    @Option(name="--numLabels",usage="Train batch size.",aliases = "-labels")
    public static int numLabels = 1000;
    @Option(name="--totalIterations",usage="Train batch size.",aliases = "-iterations")
    public static int totalIterations = 300;
    @Option(name="--trainBatchSize",usage="Train batch size.",aliases = "-batch")
    public static int trainBatchSize = 128;
    @Option(name="--gcWindow",usage="Set Garbage Collection window in milliseconds.",aliases = "-gcwindow")
    public static int gcWindow = 5000;
    @Option(name="--inputDimension",usage="The height and width of the dataset",aliases = "-dim")
    public static int inputDimension = 224;
    @Option(name="--profile",usage="Run profiler and print results",aliases = "-profile")
    public static boolean profile = false;

    private int height = 224;
    private int width = 224;
    private int channels = 3;
    private String datasetName  = "SIMULATEDCNN";
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

        this.height = inputDimension;
        this.width = inputDimension;

        Nd4j.create(1);
        Nd4j.getMemoryManager().togglePeriodicGc(false);
        Nd4j.getMemoryManager().setAutoGcWindow(gcWindow);
        Nd4j.getMemoryManager().setOccasionalGcFrequency(0);

        log.info("Loading data...");
        int[] shape = new int[]{trainBatchSize, 3, 224, 224};
        DataSetIterator iter = new BenchmarkDataSetIterator(shape, numLabels, totalIterations);

        log.info("Preparing benchmarks for "+totalIterations+" iterations, "+numLabels+" labels");

        benchmark(height, width, channels, numLabels, trainBatchSize, seed, datasetName, iter, modelType, profile);

        System.exit(0);
    }

    public static void main(String[] args) throws Exception {
        new BenchmarkCnn().run(args);
    }
}
