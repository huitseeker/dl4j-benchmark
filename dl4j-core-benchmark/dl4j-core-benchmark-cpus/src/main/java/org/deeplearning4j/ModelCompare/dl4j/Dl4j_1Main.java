package org.deeplearning4j.ModelCompare.dl4j;

import org.deeplearning4j.Utils.DL4J_Utils;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.text.DecimalFormat;

/**
 */
public class Dl4j_1Main {
    private static Logger log = LoggerFactory.getLogger(Dl4j_1Main.class);

    // values to pass in from command line when compiled, esp running remotely
    @Option(name="--modelType",usage="Model type.",aliases = "-mT")
    public String modelType = "mlp";
    @Option(name="--numGPUs",usage="How many workers to use for multiple GPUs.",aliases = "-ng")
    // 12 is best on AWS
    public int numGPUs = 0;
    @Option(name="--halfPrecision",usage="Apply half precision for GPUs.",aliases = "-h")
    public boolean half = false;

    protected final int seed = 42;
    protected final int nCores = 32;

    protected int height;
    protected int width;
    protected int channels;
    protected int numLabels;
    protected int batchSize;
    protected int epochs;
    protected double learningRate;
    protected double momentum;
    protected double l2;

    public void setVaribales() {
        if(modelType.equals("mlp")) {
            height = 28;
            width = 28;
            channels = 1;
            numLabels = 10;
            batchSize = 100;
            epochs = 15;
            learningRate = 6e-3;
            momentum = 0.9;
            l2 = 1e-4;
        } else {
            height = 28;
            width = 28;
            channels = 1;
            numLabels = 10;
            batchSize = 100;
            epochs = 15;
            learningRate = 1e-2;
            momentum = 0.9;
            l2 = 5e-4;
        }
    }

    public void run(String[] args) throws Exception {
        long totalTime = System.currentTimeMillis();
        MultiLayerNetwork network;

        // Parse command line arguments if they exist
        CmdLineParser parser = new CmdLineParser(this);
        try {
            parser.parseArgument(args);
        } catch (CmdLineException e) {
            // handling of wrong arguments
            System.err.println(e.getMessage());
            parser.printUsage(System.err);
        }

        setVaribales();

        if(numGPUs > 0 && half)
            DataTypeUtil.setDTypeForContext(DataBuffer.Type.HALF);

        log.debug("Load data");
        long dataLoadTime = System.currentTimeMillis();
        DataSetIterator trainData = new MultipleEpochsIterator(epochs, new MnistDataSetIterator(batchSize,true,12345));
        DataSetIterator testData = new MnistDataSetIterator(batchSize,false,12345);
        dataLoadTime = System.currentTimeMillis() - dataLoadTime;

        log.debug("Build model");
        if(modelType.equals("mlp")) {
            log.debug("MLP");
            network = new Dl4j_MLP(height, width, channels, numLabels, learningRate, momentum, l2, seed).build_model();
        }
        else {
            log.debug("Lenet");
            network = new Dl4j_Lenet(height, width, channels, numLabels, learningRate, momentum, l2, seed).build_model();
        }
        log.debug("Train model");
        long trainTime = System.currentTimeMillis();
        DL4J_Utils.train(network, trainData, numGPUs);
        trainTime = System.currentTimeMillis() - trainTime;

        log.debug("Evaluate model");
        long testTime = System.currentTimeMillis();
        Evaluation eval = network.evaluate(testData);
        log.debug(eval.stats());
        DecimalFormat df = new DecimalFormat("#.####");
        log.info(df.format(eval.accuracy()));
        testTime = System.currentTimeMillis() - testTime;

        totalTime = System.currentTimeMillis() - totalTime ;
        log.info("****************Example finished********************");
        DL4J_Utils.printTime("Data", dataLoadTime);
        DL4J_Utils.printTime("Train", trainTime);
        DL4J_Utils.printTime("Test", testTime);
        DL4J_Utils.printTime("Total", totalTime);

    }

    public static void main(String[] args) throws Exception {
        new Dl4j_1Main().run(args);
    }

}
