package ml;

public class Util {

    /**
     * The different activation functions that can be used.
     */
    public enum Activation {
        /** Absolute Value: abs(x) */
        ABS,
        /** Rectified Linear Unit: max(0, x) */
        RELU,
        /** Sigmoid: 1 / (1 + e ^ (-x)) */
        SIGMOID,
        /** Softmax: e ^ x / sum(e ^ (x_i)) */
        SOFTMAX
    }

    /**
     * The different loss functions that can be used.
     */
    public enum Loss {
        /** Squared Error: (x - y) ^ 2 */
        SE,
        /** Cross Entropy: log(x) * y */
        CE
    }

    /**
	 * Returns the input after being activated.
	 * @param x         The input to be activated
	 * @param function  The activation function
	 * @return          The activated input
	 */
    public static float activate(float x, Activation activation) {
        switch (activation) {
            case ABS:
                return Math.abs(x);
            case SIGMOID:
                return (float) (1 / (1 + Math.exp(x * -1)));
            case RELU:
            default:
                return Math.max(0, x);
        }
	}
	
	/**
	 * Returns the input vector after being activated with softmax.
	 * @param x  The input vector
	 * @return   The activated input
	 */
	public static float[] softmax(float[] x) {
		float sum = 0;
		float exponentials[] = new float[x.length];
		for (int i = 0; i < x.length; i++) {
			exponentials[i] = Math.min((float) Math.exp(x[i]), Float.MAX_VALUE);
			sum += exponentials[i];
		}
		if (Float.isInfinite((sum))) {
			sum = Float.MAX_VALUE;
		}
		float[] activations = new float[x.length];
		for (int i = 0; i < activations.length; i++) {
			activations[i] = (float) (exponentials[i] / sum);
			if (activations[i] == 0) {
				activations[i] = (float) 1e-35;
			}
			if (Float.isNaN(activations[i])) {
				System.out.println("NaN Activation");
			}
		}
		return activations;
	}
	
	/**
	 * Returns the weighted sum of the inputs and the bias.
	 * @param n  The input vector
	 * @param w  The weight vector
	 * @param b  The bias
	 * @return   The weighted sum
	 */
    public static float calculate(float[] x, float[] w, float b) {
        float sum = 0;
		for (int i = 0; i < x.length; i++) {
			sum += x[i] * w[i];
		}
		return sum + b;
	}
	
	/**
	 * Calculates the current loss of the model.
	 * @param x         The predicted value
	 * @param y         The actual value
	 * @param function  The loss function
	 * @return          The current loss
	 */
    public static float loss(float x, float y, Loss loss) {
        switch (loss) {
            case SE:
                return (float) Math.pow(x - y, 2);
            case CE:
            default:
                return (float) Math.log(x) * y;
        }
    }
    
}
