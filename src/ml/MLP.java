package ml;
import java.io.*;
import java.util.Arrays;
import java.util.Random;
import java.util.Scanner;
import java.util.concurrent.ThreadLocalRandom;

import ml.Util.*;

public class MLP implements Serializable {

	@Serial
	private static final long serialVersionUID = 1L;
	
	/** Activation function */
	Activation activation;

	/** Loss function */
	Loss loss; 
	
	/** Activations of each neuron */
	private final float[][] activations;

	/** Number of neurons in each layer */
	private final int[] neurons;

	/** Training data */
	private float[][] trainInputs;

	/** Training labels */
	private float[][] trainLabels;

	/** Test data */
	private float[][] testInputs;

	/** Test labels */
	private float[][] testLabels;

	/** Weights between neurons */
	private float[][][] weights;

	/** Biases for each layer */
	private float[][] biases;

	/** Average training loss */
	private float averageLoss;

	/**
	 * Initializes a multilayer perceptron (MLP).
	 * @param neurons  The number of neurons in each layer
	 * @param activation     The activation function
	 * @param loss           The loss function
	 */
	public MLP(int[] neurons, Activation activation, Loss loss) {
		this.neurons = neurons;
		activations = new float[neurons.length][];
		for(int i = 0;i < activations.length;i++) {
			activations[i] = new float[neurons[i]];
		}
		this.activation = activation;
		this.loss = loss;
	}

	/**
	 * Retrieves training data from array.
	 * @param data   A 2D array of the training data and labels
	 */
	public void getTrainingData(float[][] data) {

		float[][] trainY1 = new float[data.length][data[0].length - neurons[neurons.length - 1]];
		for (int i = 0; i < data.length; i++) {
			if (data[i].length - neurons[neurons.length - 1] >= 0)
				System.arraycopy(data[i], 0, trainY1[i], 0, data[i].length - neurons[neurons.length - 1]);
		}
		trainInputs = data;
		trainLabels = trainY1;
	}

	/**
	 * Retrieves test data from an array.
	 * @param data   A 2D array of the test data and labels
	 */
	public void getTestData(float[][] data) {
		float[][] testY1 = new float[data.length][data[0].length - neurons[neurons.length - 1]];
		for (int i = 0; i < data.length; i++) {
			if (data[i].length - neurons[neurons.length - 1] >= 0)
				System.arraycopy(data[i], 0, testY1[i], 0, data[i].length - neurons[neurons.length - 1]);
		}
		testInputs = data;
		testLabels = testY1;
	}

	/**
	 * Returns the vector in the given file in a One-hot format.
	 * @param fileName    the path to the file
	 * @param dataLength  the length of the vector
	 * @return            the one-hot vector
	 */
	public float[][] toOneHot(String fileName, int dataLength) throws Exception {
		float[][] trainingData = new float[dataLength][neurons[0] + neurons[neurons.length - 1]];
		Scanner train = new Scanner(new BufferedReader(new FileReader(fileName)));
		train.nextLine();
		for (float[] row : trainingData) {
			String[] stringValues = train.nextLine().split(", ");
			float[] floatValues = new float[stringValues.length];
			for (int j = 0; j < floatValues.length; j++) {
				floatValues[j] = Float.parseFloat(stringValues[j]);
			}
			float[] oneHot = new float[neurons[neurons.length - 1]];
			int index = (int) floatValues[0];
			oneHot[index] = 1;
			if (neurons[0] >= 0) System.arraycopy(floatValues, 1, row, 0, neurons[0]);
			if (neurons[neurons.length - 1] >= 0)
				System.arraycopy(oneHot, 0, row, neurons[0], neurons[neurons.length - 1]);
		}
		train.close();
		return trainingData;
	}

	/**
	 * Returns a scaled version of the given data.
	 * @param x      A 2D array of data
	 * @param range  The range the data should be scaled to
	 * @return       A 2D array of scaled data
	 */
	public float[][] normalize(float[][] x, float range) {
		float[][] output = new float[x.length][x[0].length];
		for(int i = 0;i < output.length;i++) {
			for(int j = 0; j < output[i].length - neurons[neurons.length - 1]; j++) {
				output[i][j] = x[i][j] / range;
			}
			if (output[i].length - (output[i].length - neurons[neurons.length - 1]) >= 0)
				System.arraycopy(x[i], output[i].length - neurons[neurons.length - 1], output[i], output[i].length - neurons[neurons.length - 1], output[i].length - (output[i].length - neurons[neurons.length - 1]));
		}
		return output;
	}

	/** Returns the cost. */
	public float getCost() {
		return averageLoss;
	}

	/**
	 * Initializes weights and biases randomly.
	 */
	public void initialize() {
		float max = (float).1;
		float min = (float)-.1;

		float[][][] weights1 = new float[neurons.length - 1][][];
		float[][] biases1 = new float[neurons.length - 1][];

		for(int i = 0;i < weights1.length;i++) {
			float[][] currentWeights = new float[neurons[i + 1]][neurons[i]];
			for(int j = 0;j < currentWeights.length;j++) {
				for(int k = 0;k < currentWeights[j].length;k++) {
					currentWeights[j][k] = min + (float) Math.random() * (max - min);
				}
			}
			weights1[i] = currentWeights;
		}

		for(int i = 0;i < biases1.length;i++) {
			float[] tempArray = new float[neurons[i + 1]];
			for(int j = 0;j < tempArray.length;j++) {
				tempArray[j] = min + (float)Math.random() * (max - min);
			}
			biases1[i] = tempArray;
		}

		weights = weights1;
		biases = biases1;
	}

	/**
	 * Completes a single iteration of stochastic gradient descent.
	 * @param learningRate  The rate at which to update the parameters
	 * @param batchSize     The number of training examples to go train on
	 */
	public void trainSGD(float learningRate, int batchSize) {
		for(int i = 0;i < weights.length;i++) {
			for(int j = 0;j < weights[i].length;j++) {
				for(int k = 0;k < weights[i][j].length;k++) {
					if(Float.isNaN(weights[i][j][k])) {
						System.out.println("NaN Weight" + "i: " + i + "j: " + j + "k: " + k);
					}
				}
			}
		}
		// Randomize training examples
		Random rnd = ThreadLocalRandom.current();
	    for(int i = trainInputs.length - 1; i > 0; i--) {
	      int index = rnd.nextInt(i + 1);
	      float[] a = trainInputs[index];
	      trainInputs[index] = trainInputs[i];
	      trainInputs[i] = a;
	    }
	    float[][] trainY1 = new float[trainInputs.length][trainInputs[0].length - neurons[neurons.length - 1]];
		for(int i = 0; i < trainInputs.length; i++) {
			if (trainInputs[i].length - neurons[neurons.length - 1] >= 0)
				System.arraycopy(trainInputs[i], 0, trainY1[i], 0, trainInputs[i].length - neurons[neurons.length - 1]);
		}
	    trainLabels = trainY1;

		// Organize data into batches for Stochastic gradient descent
		float[][][] trainXBatches;
		if(trainInputs.length % batchSize == 0) {
			trainXBatches = new float[(int)(trainInputs.length / batchSize)][][];
		} else {
			trainXBatches = new float[(trainInputs.length / batchSize) + 1][][];
		}
		int trainXExamples = 0;
		for(int i = 0;i < trainXBatches.length;i++) {
			int examples;
			if(i != trainXBatches.length - 1) {
				examples = batchSize;
			} else if(trainInputs.length % batchSize == 0) {
				examples = batchSize;
			} else {
				examples = trainInputs.length % batchSize;
			}
			float[][] currentExamples = new float[examples][];
			for(int j = 0;j < examples;j++) {
				currentExamples[j] = trainInputs[trainXExamples];
				trainXExamples++;
			}
			trainXBatches[i] = currentExamples;
		}
		float[][][] trainYBatches = new float[(trainInputs.length / batchSize) + 1][][];
		int trainYExamples = 0;
		for(int i = 0;i < trainYBatches.length;i++) {
			int examples;
			if(i != trainYBatches.length - 1) {
				examples = batchSize;
			} else {
				examples = trainLabels.length % batchSize;
			}
			float[][] currentExamples = new float[examples][];
			for(int j = 0;j < examples;j++) {
				currentExamples[j] = trainLabels[trainYExamples];
				trainYExamples++;
			}
			trainYBatches[i] = currentExamples;
		}

		// Initialize arrays of total derivatives; e.g. totalDCostDWeights means the total derivative of the cost with respect to the weights
		float totalCost = 0;
		float[][][] totalDCostDWeights = new float[weights.length][][];
		for(int k = 0;k < weights.length;k++) {
			float[][] tempArray = new float[weights[k].length][activations[k].length];
			totalDCostDWeights[k] = tempArray;
		}

		float[][] totalDCostDBiases = new float[biases.length][];
		for(int k = 0;k < biases.length;k++) {
			float[] tempArray = new float[biases[k].length];
			totalDCostDBiases[k] = tempArray;
		}

		for(int j = 0;j < trainXBatches.length;j++) {
			// Iterate through all training examples in the batch
			for(int k = 0;k < trainXBatches[j].length;k++) {
				// Feed data forward
				for(int l = 0;l < activations.length - 1;l++) {
					for(int m = 0;m < activations[l].length;m++) {
						if(l == 0) {
							activations[l][m] = trainYBatches[j][k][m];
						} else if(activation != Activation.SOFTMAX){
							activations[l][m] = Util.activate(Util.calculate(activations[l - 1], weights[l - 1][m], biases[l - 1][m]), activation);
						} else {
							activations[l][m] = Util.calculate(activations[l - 1], weights[l - 1][m], biases[l - 1][m]);
						}
					}
				}

				// Compute cost
				float[] targets = new float[neurons[neurons.length - 1]];
				for(int l = 0;l < targets.length;l++) {
					targets[targets.length - (l + 1)] = trainXBatches[j][k][trainXBatches[j][k].length - (l + 1)];
				}
				for(int l = 0;l < activations[activations.length - 1].length;l++) {
					activations[activations.length - 1][l] = Util.calculate(activations[activations.length - 2], weights[weights.length - 1][l], biases[biases.length - 1][l]);
					if(Float.isNaN(activations[activations.length - 1][l])) {
						System.out.print(weights[0][0][0]);
					}
				}
				float[] activatedOutputs = new float[neurons[neurons.length - 1]];
				if(activation != Activation.SOFTMAX) {
					for(int l = 0;l < activatedOutputs.length;l++) {
						activatedOutputs[l] = Util.activate(activations[activations.length - 1][l], activation);
					}
				} else {
					activatedOutputs = Util.softmax(activations[activations.length - 1]);
				}
				float cost = 0;
				for(int l = 0; l < neurons[neurons.length - 1]; l++) {
					cost += Util.loss(activatedOutputs[l], targets[l], loss);
				}
				if(loss == Loss.CE) {
					cost *= -1;
				}

				// Initialize and compute arrays of derivatives
				// The derivative of the cost function
				float[] dCostDActivatedOutputs = new float[neurons[neurons.length - 1]];
				for(int l = 0;l < dCostDActivatedOutputs.length;l++) {
					if(loss == Loss.SE) {
						dCostDActivatedOutputs[l] = 2 * (activatedOutputs[l] - targets[l]);
					} else if(loss == Loss.CE) {
						dCostDActivatedOutputs[l] = activatedOutputs[l] - targets[l];
					}
				}

				// The derivative of the activation function
				float[][] dActivatedOutputsDOutputs = new float[activations.length - 1][];
				for(int l = 0;l < dActivatedOutputsDOutputs.length;l++) {
					float[] tempArray = new float[neurons[neurons.length - (l + 1)]];
					dActivatedOutputsDOutputs[l] = tempArray;
				}
				for(int l = 0;l < dActivatedOutputsDOutputs.length;l++) {
					for(int m = 0;m < dActivatedOutputsDOutputs[l].length;m++) {
						if(activation == Activation.SIGMOID) {
							dActivatedOutputsDOutputs[l][m] = Util.activate(activations[activations.length - (l + 1)][m], activation) * (1 - Util.activate(activations[activations.length - (l + 1)][m], activation));
						} else if(activation == Activation.SOFTMAX) {
							if(l == 0) {
								dActivatedOutputsDOutputs[0][0] = Util.activate(activations[activations.length - 1][m], Activation.SIGMOID);
							} else {
								dActivatedOutputsDOutputs[l][m] = Util.activate(activations[activations.length - (l + 1)][m], Activation.SIGMOID);
							}
						} else if(activation == Activation.RELU) {
							if(activations[activations.length - (l + 1)][m] > 0) {
								dActivatedOutputsDOutputs[l][m] = 1;
								System.out.println("0");
							} else {
								dActivatedOutputsDOutputs[l][m] = 0;
							}
						} else {
							dActivatedOutputsDOutputs[l][m] = 1;
						}
					}
				}

				// The derivative of the calculate function
				float[][][] dOutputsDWeights = new float[weights.length][][];
				for(int l = 0;l < weights.length;l++) {
					float[][] tempArray = new float[weights[l].length][activations[l].length];
					dOutputsDWeights[l] = tempArray;
				}
				for(int l = 0;l < dOutputsDWeights.length;l++) {
					for(int m = 0;m < dOutputsDWeights[l].length;m++) {
						System.arraycopy(activations[l], 0, dOutputsDWeights[l][m], 0, dOutputsDWeights[l][m].length);
					}
				}

				// The derivative of the cost function with respect to the weights and biases uses the chain rule by multiplying the derivatives of each of the functions being applied to the weights and biases
				// For example,  the derivative of cost(sigmoid(calculate(weight))) is dCost * dSigmoid * dCalculate
				float[][][] dCostDWeights = new float[weights.length][][];
				for(int l = 0;l < weights.length;l++) {
					float[][] tempArray = new float[weights[l].length][activations[l].length];
					dCostDWeights[l] = tempArray;
				}
				for(int l = 0;l < dCostDWeights.length;l++) {
					for(int m = 0;m < dCostDWeights[dCostDWeights.length - (l + 1)].length;m++) {
						for(int n = 0;n < dCostDWeights[dCostDWeights.length - (l + 1)][m].length;n++) {
							if(l == 0) {
								dCostDWeights[dCostDWeights.length - (l + 1)][m][n] = dCostDActivatedOutputs[m] * dActivatedOutputsDOutputs[l][m] * dOutputsDWeights[dOutputsDWeights.length - (l + 1)][m][n];
							} else {
								// Add up the costs for the different possible ways each weight can affect the output
								float derivativeSum = 0;
								for(int o = 0;o < dCostDWeights[dCostDWeights.length - l].length;o++) {
									derivativeSum+= dCostDWeights[dCostDWeights.length - l][o][m];
								}
								// The derivative of a weight in a hidden layer also uses the chain rule; it gets multiplied by the derivative of the weight connected to it in the next layer
								dCostDWeights[dCostDWeights.length - (l + 1)][m][n] = derivativeSum * dActivatedOutputsDOutputs[l][m] * dOutputsDWeights[dOutputsDWeights.length - (l + 1)][m][n];
							}
						}
					}
				}

				float[][] dCostDBiases = new float[biases.length][];
				for(int l = 0;l < biases.length;l++) {
					float[] tempArray = new float[biases[l].length];
					dCostDBiases[l] = tempArray;
				}
				for(int l = 0;l < dCostDBiases.length;l++) {
					if(l == 0) {
						for(int m = 0;m < dCostDBiases[dCostDBiases.length - (l + 1)].length;m++) {
							dCostDBiases[dCostDBiases.length - (l + 1)][m] = dCostDActivatedOutputs[m] * dActivatedOutputsDOutputs[l][m];
						}
					} else {
						for(int m = 0;m < dCostDBiases[dCostDBiases.length - (l + 1)].length;m++) {
							float derivativeSum = 0;
							// Add up the costs for the different possible ways each bias can affect the output
							for(int n = 0;n < dCostDBiases[dCostDBiases.length - l].length;n++) {
								derivativeSum += dCostDBiases[dCostDBiases.length - l][n];
							}
							dCostDBiases[dCostDBiases.length - (l + 1)][m] = derivativeSum * dActivatedOutputsDOutputs[l][m];
						}
					}
				}

				// Add cost and derivatives to totals
				totalCost += cost;
				for(int l = 0;l < totalDCostDWeights.length;l++) {
					for(int m = 0;m < totalDCostDWeights[l].length;m++) {
						for(int n = 0;n < totalDCostDWeights[l][m].length;n++) {
							totalDCostDWeights[l][m][n] += dCostDWeights[l][m][n];
						}
					}
				}
				for(int l = 0;l < totalDCostDBiases.length;l++) {
					for(int m = 0;m < totalDCostDBiases[l].length;m++) {
						totalDCostDBiases[l][m] += dCostDBiases[l][m];
					}
				}
			}
			averageLoss = totalCost / trainXBatches[j].length;

			// Initialize and compute arrays of average derivatives
			float[][][] averageDCostDWeights = new float[weights.length][][];
			for(int k = 0;k < weights.length;k++) {
				float[][] tempArray = new float[weights[k].length][activations[k].length];
				for(int l = 0;l < tempArray.length;l++) {
					for(int m = 0;m < tempArray[l].length;m++) {
						tempArray[l][m] = totalDCostDWeights[k][l][m] / trainXBatches[j].length;
					}
				}
				averageDCostDWeights[k] = tempArray;
			}

			float[][] averageDCostDBiases = new float[biases.length][];
			for(int k = 0;k < biases.length;k++) {
				float[] tempArray = new float[biases[k].length];
				for(int l = 0;l < tempArray.length;l++) {
					tempArray[l] = totalDCostDBiases[k][l] / trainXBatches[j].length;
				}
				averageDCostDBiases[k] = tempArray;
			}

			// Subtract a fraction of the gradient from the weights and biases
			for(int k = 0;k < weights.length;k++) {
				for(int l = 0;l < weights[k].length;l++) {
					for(int m = 0;m < weights[k][l].length;m++) {
						weights[k][l][m] -= learningRate * averageDCostDWeights[k][l][m];
					}
				}
			}

			for(int k = 0;k < biases.length;k++) {
				for(int l = 0;l < biases[k].length;l++) {
					biases[k][l] -= learningRate * averageDCostDBiases[k][l];
				}
			}
		}
	}

	/**
	 * Completes several iterations of stochastic gradient descent.
	 * @param test           Whether or not to compute the test accuracy
	 * @param trainTime      The number of iterations to train for
	 * @param printLoss      The prequency at which to print the loss
	 * @param saveTime       The frequency at which to save the model
	 * @param batchSize      The number of training examples per iteration
	 * @param learningRate   The rate at which to update the parameters
	 * @param fileName       The path to where the model should be saved
	 */
	public void customTrain(boolean test, int trainTime, int printLoss, int saveTime, float learningRate, int batchSize, String fileName) throws IOException {
		for(int i = 0;i <= trainTime;i++) {
			trainSGD(learningRate, batchSize);
			if(i % printLoss == 0) {
				System.out.println("Iteration: " + i + " Loss: " + getCost());
				if(test) {
					System.out.println("Accuracy: " + test() + " / " + testInputs.length);
				}
			}
			if(saveTime != 0 && i % saveTime == 0) {
				save(fileName);
			}
		}
	}

	/**
	 * Evaluates the model on the training set.
	 * @return the current test accuracy
	 */
	public float test() {
		for (float[] floats : activations) {
			Arrays.fill(floats, 0);
		}
		float accuracy = 0;
		for(int i = 0; i < testInputs.length; i++) {
			// Feed data forward
			for(int j = 0;j < activations.length - 1;j++) {
				for(int k = 0;k < activations[j].length;k++) {
					if(j == 0) {
						activations[j][k] = testLabels[i][k];
					} else {
						activations[j][k] = Util.activate(Util.calculate(activations[j - 1], weights[j - 1][k], biases[j - 1][k]), activation);
					}
				}
			}

			float[] targets = new float[neurons[neurons.length - 1]];
			for(int j = 0;j < targets.length;j++) {
				targets[targets.length - (j + 1)] = testInputs[i][testInputs[i].length - (j + 1)];
			}
			for(int j = 0;j < activations[activations.length - 1].length;j++) {
				activations[activations.length - 1][j] = Util.calculate(activations[activations.length - 2], weights[weights.length - 1][j], biases[biases.length - 1][j]);
			}
			float[] activatedOutputs = new float[neurons[neurons.length - 1]];
			if(activation != Activation.SOFTMAX) {
				for(int j = 0;j < activatedOutputs.length;j++) {
					activatedOutputs[j] = Util.activate(activations[activations.length - 1][j], activation);
				}
			} else {
				activatedOutputs = Util.softmax(activations[activations.length - 1]);
			}
			boolean correct = true;
			for(int j = 0; j < neurons[neurons.length - 1]; j++) {
				if ((float) Math.round(activatedOutputs[j]) != targets[j]) {
					correct = false;
					break;
				}
			}
			if(correct) {
				accuracy += 1;
			}
		}
		return accuracy;

	}

	/**
	 * Returns the model's prediction on the given input.
	 * @param inputs  A 1D array of data
	 * @return        The model's prediction on `inputs`
	 */
	public float[] feedForward(float[] inputs) {
		float[] newOutputs = new float[neurons[neurons.length - 1]];
		if(activations.length == 2) {
			if(activation != Activation.SOFTMAX) {
				for(int i = 0;i < newOutputs.length;i++) {
					newOutputs[i] = Util.activate(Util.calculate(inputs, weights[0][i], biases[0][i]), activation);
				}
			} else {
				for(int i = 0;i < newOutputs.length;i++) {
					newOutputs[i] = Util.calculate(inputs, weights[0][i], biases[0][i]);
				}
				newOutputs = Util.softmax(newOutputs);
			}
		} else {
			for(int i = 0;i < activations.length - 2;i++) {
				if(i == 0) {
					for(int j = 0; j < neurons[1]; j++) {
						activations[1][j] = Util.activate(Util.calculate(inputs, weights[i][j], biases[i][j]), activation);
					}
				} else {
					for(int j = 0; j < neurons[i + 1]; j++) {
						activations[i + 1][j] = Util.activate(Util.calculate(activations[i - 1], weights[i][j], biases[i][j]), activation);
					}
				}
			}
			if(activation != Activation.SOFTMAX) {
				for(int i = 0;i < newOutputs.length;i++) {
					newOutputs[i] = Util.activate(Util.calculate(activations[activations.length - 2], weights[weights.length - 1][i], biases[biases.length - 1][i]), activation);
				}
			} else {
				for(int i = 0;i < newOutputs.length;i++) {
					newOutputs[i] = Util.calculate(activations[activations.length - 2], weights[weights.length - 1][i], biases[biases.length - 1][i]);
				}
				newOutputs = Util.softmax(newOutputs);
			}
		}
		return newOutputs;
	}

	/**
	 * Prints the model's prediction on an input vector.
	 * @param inputs      The input vector
	 */
	public void printOutputs(float[] inputs) {
		System.out.println("\n" + "Predicted Outputs:");
		for (int i = 0; i < neurons[neurons.length - 1]; i++) {
			System.out.println(feedForward(inputs)[i]);
		}
	}
	
	/**
	 * Saves the model to a file.
	 * @param fileName      The path to save the model to
	 */
	public void save(String fileName) throws IOException {
		OutputStream outStream = new FileOutputStream(fileName);
		ObjectOutputStream fileObjectOut = new ObjectOutputStream(outStream);
		fileObjectOut.writeObject(this);
		fileObjectOut.close();
		outStream.close();
	}
	
	/**
	 * Loads a model from a file
	 * @param fileName                 The path to load from
	 * @return                         The loaded MLP
	 */
	public static MLP load(String fileName) throws IOException,  ClassNotFoundException {
        InputStream inStream = new FileInputStream(fileName);
		ObjectInputStream fileObjectIn = new ObjectInputStream(inStream);
        MLP loaded = (MLP) fileObjectIn.readObject();
        fileObjectIn.close();
        return loaded;
	}
}

