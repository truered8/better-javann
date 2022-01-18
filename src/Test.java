import java.io.IOException;

import ml.MLP;
import ml.Util.*;

public class Test {
    
    public static void main(String[] args) throws IOException {

        int[] structure = { 2, 1 };
        MLP m = new MLP(structure, Activation.SIGMOID, Loss.CE);
        m.initialize();

        float[] data[] = {
                { 1, 60000, 1 }, { 7, 200, 0 }, { (float) 0.6, 700000, 1 },
                { (float) 0.8, 120000, 1 }, { 5, 500, 0 }, { 8, 10, 0 },
                { (float) 0.5, 5000000, 1 }, { 6, 1, 0 } };

        for (int i = 0; i < data.length; i++)
            data[i][1] = (float) Math.log10(data[i][1]);

        m.getTrainingData(data);
        m.customTrain(
            false,
            1000,
            100,
            100,
            (float) 0.1,
            4,
            (float) 1.0,
            "model.png"
        );
        
        float[] inputs = { 1, (float) Math.log10(1000000) };
        m.printOutputs(inputs);

    }

}
