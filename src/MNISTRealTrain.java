import java.io.IOException;

import ml.NewMLP;

public class MNISTRealTrain {

	public static void main(String[] args) throws ClassNotFoundException, IOException {
		
		int trainTime = 50000,printError = 1,saveTime = 10000,batchSize = 50;
		float learningRate = (float).1,decay = (float).99;
		String fileName = "C:/Users/Babtu/Documents/Models/MNISTRealV1.ser";
		NewMLP mnist = NewMLP.load("C:/Users/Babtu/Documents/Models/MNISTRealV1.ser");
		mnist.customTrain(true,trainTime,printError,saveTime,batchSize,learningRate,decay,fileName);


	}

}
