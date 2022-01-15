import ml.MLP;
import ml.NewMLP;

public class MNISTRealInitializer {

	public static void main(String[] args) throws Exception {

		int neurons[] = {784,30,10};
		float[] fakeTrain[] = new float[59999][793];
		float[] fakeTest[] = new float[9999][793];
		NewMLP mnist = new NewMLP(neurons,"softmax","CE");
		mnist.getData(fakeTrain);
		mnist.getData(mnist.normalize(mnist.toOneHot("mnist_train.csv",59999),255));
		mnist.getTestData(fakeTest);
		mnist.getTestData(mnist.normalize(mnist.toOneHot("mnist_test.csv",9999),255));
		mnist.createNetwork();
		mnist.save("C:/Users/Babtu/Documents/Models/MNISTRealV1.ser");
		System.out.println("Done!");

	}

}
