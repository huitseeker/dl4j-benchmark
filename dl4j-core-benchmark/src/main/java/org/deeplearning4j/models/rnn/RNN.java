package org.deeplearning4j.models.rnn;

import lombok.AllArgsConstructor;
import org.deeplearning4j.models.TestableModel;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Created by Alex on 13/04/2017.
 */
@AllArgsConstructor
public class RNN implements TestableModel {

    private int inputSize;
    private int[] layerSizes;
    private int outputSize;
    private long seed;
    private Updater updater;

    public MultiLayerConfiguration conf(){
        NeuralNetConfiguration.ListBuilder builder = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.XAVIER)
                .updater(updater)
                .seed(seed)
                .activation(Activation.TANH)
                .list();

        for( int i=0; i<layerSizes.length; i++ ){
            builder.layer(i, new GravesLSTM.Builder()
                    .nIn(i == 0 ? inputSize : layerSizes[i-1])
                    .nOut(layerSizes[i])
                    .build());
        }
        builder.layer(layerSizes.length, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                .nIn(layerSizes[layerSizes.length-1])
                .nOut(outputSize)
                .activation(Activation.SOFTMAX)
                .build());

        return builder.build();
    }

    @Override
    public Model init() {
        MultiLayerNetwork network = new MultiLayerNetwork(conf());
        network.init();
        return network;
    }
}
