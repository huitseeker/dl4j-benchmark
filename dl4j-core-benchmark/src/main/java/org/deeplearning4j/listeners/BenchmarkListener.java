package org.deeplearning4j.listeners;

/**
 * Created by justin on 3/25/17.
 */

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.atomic.AtomicLong;

/**
 * Variation of PerformanceListener that allows collection of statistics.
 *
 * @author Justin Long (@crockpotveggies)
 */
public class BenchmarkListener implements IterationListener {
    private final int frequency;
    private static final Logger logger = LoggerFactory.getLogger(org.deeplearning4j.optimize.listeners.PerformanceListener.class);
    private ThreadLocal<Double> samplesPerSec = new ThreadLocal<>();
    private ThreadLocal<Double> batchesPerSec = new ThreadLocal<>();
    private ThreadLocal<Long> lastTime = new ThreadLocal<>();
    private ThreadLocal<AtomicLong> iterationCount = new ThreadLocal<>();

    private BenchmarkReport benchmarkReport;

    private String device;

    public BenchmarkListener(BenchmarkReport benchmarkReport) {
        this.benchmarkReport = benchmarkReport;
        this.frequency = 1;
    }

    @Override
    public boolean invoked() {
        return false;
    }

    @Override
    public void invoke() {

    }

    @Override
    public void iterationDone(Model model, int iteration) {
        // we update lastTime on every iteration
        // just to simplify things
        if (lastTime.get() == null)
            lastTime.set(System.currentTimeMillis());

        if (samplesPerSec.get() == null)
            samplesPerSec.set(0.0);

        if (batchesPerSec.get() == null)
            batchesPerSec.set(0.0);

        if (iterationCount.get() == null)
            iterationCount.set(new AtomicLong(0));

        if(iterationCount.get().get() <= 3*frequency)
            lastTime.set(System.currentTimeMillis());

        if (iterationCount.get().getAndIncrement() % frequency == 0 && iterationCount.get().get() > 3*frequency) {
            long currentTime = System.currentTimeMillis();

            long timeSpent = currentTime - lastTime.get();
            float timeSec = timeSpent / 1000f;

            INDArray input;
            if (model instanceof ComputationGraph) {
                // for comp graph (with multidataset
                ComputationGraph cg = (ComputationGraph) model;
                INDArray[] inputs = cg.getInputs();

                if (inputs != null && inputs.length > 0)
                    input = inputs[0];
                else
                    input = model.input();
            } else {
                input = model.input();
            }

            long tadLength = Shape.getTADLength(input.shape(), ArrayUtil.range(1, input.rank()));

            long numSamples = input.lengthLong() / tadLength;

            samplesPerSec.set((double) (numSamples / timeSec));
            batchesPerSec.set((double) (1 / timeSec));

            benchmarkReport.setIterations(iterationCount.get().get());
            benchmarkReport.addIterationTime(timeSpent);
            if(!Double.isInfinite(samplesPerSec.get())) benchmarkReport.addSamplesSec(samplesPerSec.get());
            if(!Double.isInfinite(batchesPerSec.get())) benchmarkReport.addBatchesSec(batchesPerSec.get());
        }

        lastTime.set(System.currentTimeMillis());
    }
}
