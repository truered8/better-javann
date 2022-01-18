package ml;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.Serializable;
import java.util.Random;
import java.util.Scanner;
import java.util.concurrent.ThreadLocalRandom;

import ml.Util.*;

public class MLP implements Serializable{

	private static final long serialVersionUID = 1L;
	
	/** Activation function */
	Activation activation;

	/** Loss function */
	Loss loss; 
	
	/** Activations of each neuron */
	private float[][] activations;

	/** Number of neurons in each layer */
	private int[] neuronNumbers;
	
	/** Training data */
	private float[] trainX[];

	/** Training labels */
	private float[] trainY[];

	/** Test data */
	private float[] testX[];

	/** Test labels */
	private float[] testY[];
	
	/** Weights between neurons */
	private float[][][] weights;

	/** Biases for each label */
	private float[][] biases;
	
	/** Average training loss */
	private float averageLoss;
	
	/**
	 * Initializes a multilayer perceptron (MLP).
	 * @param neuronNumbers  The number of neurons in each layer
	 * @param activation     The activation function
	 * @param loss           The loss function
	 */
	public MLP(int[] neuronNumbers, Activation activation, Loss loss) {
		this.neuronNumbers = neuronNumbers;
		activations = new float[neuronNumbers.length][];
		for(int i = 0;i < activations.length;i++) {
			activations[i] = new float[neuronNumbers[i]];
		}
		this.activation = activation;
		this.loss = loss;
	}
	
	/**
	 * Retrieves training data from a Comma Separated Values (CSV) file.
	 * @param   fileName  The path to the data
	 * @return  void
	 * @throws  Exception
	 */
	public void getCSVData(String fileName) throws Exception {
		// Copy values from CSV file into trainX array
		int columns = 0;
		int row = 0;
		String inLine = "";

		Scanner reader = new Scanner(new BufferedReader(new FileReader(fileName)));
		BufferedReader read = new BufferedReader(new FileReader(fileName));
		while (read.readLine() != null) {
			columns += 1;
		}

		float[] trainX1[] = new float[columns - 1][reader.nextLine().split(", ").length];
		while (reader.hasNextLine()) {
			inLine = reader.nextLine();
			String[] inArray = inLine.split(", ");
			for (int x = 0; x < inArray.length; x++) {
				if (inArray[x].equals("?")) {
					trainX1[row][x] = -99999;
				} else {
					trainX1[row][x] = Float.parseFloat(inArray[x]);
				}
			}
			row++;
		}

		// Copy attributes from trainX to trainY
		float[] trainY1[] = new float[trainX1.length][trainX1[0].length - neuronNumbers[neuronNumbers.length - 1]];
		for (int i = 0; i < trainX1.length; i++) {
			for (int j = 0; j < trainX1[i].length - neuronNumbers[neuronNumbers.length - 1]; j++) {
				trainY1[i][j] = trainX1[i][j];
			}
		}
		trainX = trainX1;
		trainY = trainY1;
		reader.close();
		read.close();
	}
	
	/**
	 * Retrieves training data from array.
	 * @param data   A 2D array of the training data and labels
	 * @return void
	 */
	public void getTrainingData(float[][] data) {

		float[] trainY1[] = new float[data.length][data[0].length - neuronNumbers[neuronNumbers.length - 1]];
		for (int i = 0; i < data.length; i++) {
			for (int j = 0; j < data[i].length - neuronNumbers[neuronNumbers.length - 1]; j++) {
				trainY1[i][j] = data[i][j];
			}
		}
		trainX = data;
		trainY = trainY1;
	}
	
	/**
	 * Retrieves test data from an array.
	 * @param data   A 2D array of the test data and labels
	 * @return void
	 */
	public void getTestData(float[][] data) {
		float[] testY1[] = new float[data.length][data[0].length - neuronNumbers[neuronNumbers.length - 1]];
		for (int i = 0; i < data.length; i++) {
			for (int j = 0; j < data[i].length - neuronNumbers[neuronNumbers.length - 1]; j++) {
				testY1[i][j] = data[i][j];
			}
		}
		testX = data;
		testY = testY1;
	}
	
	/**
	 * Returns the vector in the given file in a One-hot format.
	 * @param fileName    the path to the file
	 * @param dataLength  the length of the vector
	 * @return            the one-hot vector
	 * @throws            Exception
	 */
	public float[][] toOneHot(String fileName, int dataLength) throws Exception {
		float[] trainingData[] = new float[dataLength][neuronNumbers[0] + neuronNumbers[neuronNumbers.length - 1]];
		Scanner train = new Scanner(new BufferedReader(new FileReader("C:/Users/Babtu/Documents/Datasets/" + fileName)));
		train.nextLine();
		for(int i = 0;i < trainingData.length;i++) {
			String stringValues[] = train.nextLine().split(", ");
			float floatValues[] = new float[stringValues.length];
			for(int j = 0;j < floatValues.length;j++) {
				floatValues[j] = Float.parseFloat(stringValues[j]);
			}
			float oneHot[] = new float[neuronNumbers[neuronNumbers.length - 1]];
			int index = (int)floatValues[0];
			oneHot[index] = 1;
			for(int j = 0;j < neuronNumbers[0];j++) {
				trainingData[i][j] = floatValues[j + 1];
			}
			for(int j = 0;j < neuronNumbers[neuronNumbers.length - 1];j++) {
				trainingData[i][j + neuronNumbers[0]] = oneHot[j];
			}
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
		float[] output[] = new float[x.length][x[0].length];
		for(int i = 0;i < output.length;i++) {
			for(int j = 0;j < output[i].length - neuronNumbers[neuronNumbers.length - 1];j++) {
				output[i][j] = x[i][j] / range;
			}
			for(int j = output[i].length - neuronNumbers[neuronNumbers.length - 1];j < output[i].length;j++) {
				output[i][j] = x[i][j];
			}
		}
		return output;
	}
	
	/** Returns the training data. */
	public float[][] getTrainX() {
		return trainX;
	}

	/** Returns the training labels. */
	public float[][] getTrainY() {
		return trainY;
	}

	/** Returns the test data. */
	public float[][] getTestX() {
		return testX;
	}

	/** Returns the weights. */
	public float[][][] getWeights() {
		return weights;
	}

	/** Returns the biases */
	public float[][] getBiases() {
		return biases;
	}

	/** Returns the cost. */
	public float getCost() {
		return averageLoss;
	}

	/** Sets the training data. */
	public void setTrainX(int i, int j, float newData) {
		this.trainX[i][j] = newData;
	}
	
	/** Sets a specific weight. */
	public void setWeight(int i, int j, int k, float newWeight) {
		weights[i][j][k] = newWeight;
	}

	/** Sets a specific bias. */
	public void setBias(int i, int j, float newBias) {
		biases[i][j] = newBias;
	}

	/**
	 * Initializes weights and biases randomly.
	 * @return void
	 */
	public void initialize() {
		
		float max = (float).1;
		float min = (float)-.1;

		float[][] weights1[] = new float[neuronNumbers.length - 1][][];
		float[] biases1[] = new float[neuronNumbers.length - 1][];
		
		for(int i = 0;i < weights1.length;i++) {
			float[] currentWeights[] = new float[neuronNumbers[i + 1]][neuronNumbers[i]];
			for(int j = 0;j < currentWeights.length;j++) {
				for(int k = 0;k < currentWeights[j].length;k++) {
					currentWeights[j][k] = min + (float) Math.random() * (max - min);
				}
			}
			weights1[i] = currentWeights;
		}

		for(int i = 0;i < biases1.length;i++) {
			float tempArray[] = new float[neuronNumbers[i + 1]];
			for(int j = 0;j < tempArray.length;j++) {
				tempArray[j] = min + (float)Math.random() * (max - min);
			}
			biases1[i] = tempArray;
		}
		
		weights = weights1;
		biases = biases1;
	}
		
	/**
	 * Retrieves weights from a file.
	 * @param fileName      The path to the file
	 * @return void
	 * @throws IOException
	 */
	public void getFileWeights(String fileName) throws IOException {
		Scanner reader = new Scanner(new BufferedReader(new FileReader(fileName)));
		int layer = 0;
		int neuron = 0;
		int lastNeuron = 0;
		int line = 1;
		String currentLine = "";
		String lastLine = null;
		while (reader.hasNextLine()) {
			currentLine = reader.nextLine();
			if (line != 1 && currentLine.equals("")) {
				lastNeuron = 0;
				neuron += 1;
				if (lastLine.equals("")) {
					layer += 1;
					neuron = 0;
				}
			} else {
				weights[layer][neuron][lastNeuron] = Float.parseFloat(currentLine);
				lastNeuron++;
			}
			line++;
			lastLine = currentLine;
		}
		reader.close();
	}
	
	/**
	 * Retrieves biases from a file.
	 * @param fileName      The path to the file
	 * @throws IOException
	 */
	public void getFileBiases(String fileName) throws IOException {
		Scanner reader = new Scanner(new BufferedReader(new FileReader("C:/Users/Babtu/Documents/" + fileName)));
		int layer = 0;
		int neuron = 0;
		String currentLine = "";
		while(reader.hasNextLine()) {
			currentLine = reader.nextLine();
			if(currentLine.equals("")) {
				layer += 1;
				neuron = 0;
			} else {
				biases[layer][neuron] = Float.parseFloat(currentLine);
				neuron += 1;
			}
		}
		reader.close();
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
	    for(int i = trainX.length - 1; i > 0; i--) {
	      int index = rnd.nextInt(i + 1);
	      float[] a = trainX[index];
	      trainX[index] = trainX[i];
	      trainX[i] = a;
	    }
	    float[] trainY1[] = new float[trainX.length][trainX[0].length - neuronNumbers[neuronNumbers.length - 1]];
		for(int i = 0;i < trainX.length;i++) {
			for(int j = 0;j < trainX[i].length - neuronNumbers[neuronNumbers.length - 1];j++) {
				trainY1[i][j] = trainX[i][j];
			}
		}
	    trainY = trainY1;

		// Organize data into batches for Stochastic gradient descent
		float[][] trainXBatches[];
		if(trainX.length % batchSize == 0) {
			trainXBatches = new float[(int)(trainX.length / batchSize)][][];
		} else {
			trainXBatches = new float[(int)(trainX.length / batchSize) + 1][][];
		}
		int trainXExamples = 0;
		for(int i = 0;i < trainXBatches.length;i++) {
			int examples = 0;
			if(i != trainXBatches.length - 1) {
				examples = batchSize;
			} else if(trainX.length % batchSize == 0) {
				examples = batchSize;
			} else {
				examples = trainX.length % batchSize;
			}
			float[][] currentExamples = new float[examples][];
			for(int j = 0;j < examples;j++) {
				currentExamples[j] = trainX[trainXExamples];
				trainXExamples++;
			}
			trainXBatches[i] = currentExamples;
		}
		float[][] trainYBatches[] = new float[(int)(trainX.length / batchSize) + 1][][];
		int trainYExamples = 0;
		for(int i = 0;i < trainYBatches.length;i++) {
			int examples = 0;
			if(i != trainYBatches.length - 1) {
				examples = batchSize;
			} else {
				examples = trainY.length % batchSize;
			}
			float[][] currentExamples = new float[examples][];
			for(int j = 0;j < examples;j++) {
				currentExamples[j] = trainY[trainYExamples];
				trainYExamples++;
			}
			trainYBatches[i] = currentExamples;
		}
			
		// Initialize arrays of total derivatives; e.g. totalDCostDWeights means the total derivative of the cost with respect to the weights
		float totalCost = 0;
		float[][] totalDCostDWeights[] = new float[weights.length][][];
		for(int k = 0;k < weights.length;k++) {
			float[] tempArray[] = new float[weights[k].length][activations[k].length];
			totalDCostDWeights[k] = tempArray;
		}

		float[] totalDCostDBiases[] = new float[biases.length][];
		for(int k = 0;k < biases.length;k++) {
			float tempArray[] = new float[biases[k].length];
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
				float targets[] = new float[neuronNumbers[neuronNumbers.length - 1]];
				for(int l = 0;l < targets.length;l++) {
					targets[targets.length - (l + 1)] = trainXBatches[j][k][trainXBatches[j][k].length - (l + 1)];
				}
				for(int l = 0;l < activations[activations.length - 1].length;l++) {
					activations[activations.length - 1][l] = Util.calculate(activations[activations.length - 2], weights[weights.length - 1][l], biases[biases.length - 1][l]);
					if(Float.isNaN(activations[activations.length - 1][l])) {
						System.out.print(weights[0][0][0]);
					}
				}
				float activatedOutputs[] = new float[neuronNumbers[neuronNumbers.length - 1]];
				if(activation != Activation.SOFTMAX) {
					for(int l = 0;l < activatedOutputs.length;l++) {
						activatedOutputs[l] = Util.activate(activations[activations.length - 1][l], activation);
					}
				} else {
					activatedOutputs = Util.softmax(activations[activations.length - 1]);
				}
				float cost = 0;
				for(int l = 0;l < neuronNumbers[neuronNumbers.length - 1];l++) {
					cost += Util.loss(activatedOutputs[l], targets[l], loss);
				}
				if(loss == Loss.CE) {
					cost *= -1;
				}
				
				// Initialize and compute arrays of derivatives
				// The derivative of the cost function
				float dCostDActivatedOutputs[] = new float[neuronNumbers[neuronNumbers.length - 1]];
				for(int l = 0;l < dCostDActivatedOutputs.length;l++) {
					if(loss == Loss.SE) {
						dCostDActivatedOutputs[l] = 2 * (activatedOutputs[l] - targets[l]);
					} else if(loss == Loss.CE) {
						dCostDActivatedOutputs[l] = activatedOutputs[l] - targets[l];
					}
				}

				// The derivative of the activation function
				float[] dActivatedOutputsDOutputs[] = new float[activations.length - 1][];
				for(int l = 0;l < dActivatedOutputsDOutputs.length;l++) {
					float tempArray[] = new float[neuronNumbers[neuronNumbers.length - (l + 1)]];
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
						} else if(activation == Activation.SOFTMAX) {
							if(l == 0) {
								dActivatedOutputsDOutputs[l][m] = 1;// activatedOutputs[m] - trainXBatch[k][m + neuronNumbers[0]];
							} else {
								dActivatedOutputsDOutputs[l][m] = 1;// activate(neurons[neurons.length - (l + 1)][m], activation) * (1 - activate(neurons[neurons.length - (l + 1)][m], activation));
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
				float[][] dOutputsDWeights[] = new float[weights.length][][];
				for(int l = 0;l < weights.length;l++) {
					float[] tempArray[] = new float[weights[l].length][activations[l].length];
					dOutputsDWeights[l] = tempArray;
				}
				for(int l = 0;l < dOutputsDWeights.length;l++) {
					for(int m = 0;m < dOutputsDWeights[l].length;m++) {
						for(int n = 0;n < dOutputsDWeights[l][m].length;n++) {
							dOutputsDWeights[l][m][n] = activations[l][n];
						}
					}
				}

				// The derivative of the cost function with respect to the weights and biases uses the chain rule by multiplying the derivatives of each of the functions being applied to the weights and biases
				// For example,  the derivative of cost(sigmoid(calculate(weight))) is dCost * dSigmoid * dCalculate 
				float[][] dCostDWeights[] = new float[weights.length][][];
				for(int l = 0;l < weights.length;l++) {
					float[] tempArray[] = new float[weights[l].length][activations[l].length];
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

				float[] dCostDBiases[] = new float[biases.length][];
				for(int l = 0;l < biases.length;l++) {
					float tempArray[] = new float[biases[l].length];
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
			float[][] averageDCostDWeights[] = new float[weights.length][][];
			for(int k = 0;k < weights.length;k++) {
				float[] tempArray[] = new float[weights[k].length][activations[k].length];
				for(int l = 0;l < tempArray.length;l++) {
					for(int m = 0;m < tempArray[l].length;m++) {
						tempArray[l][m] = totalDCostDWeights[k][l][m] / trainXBatches[j].length;
					}
				}
				averageDCostDWeights[k] = tempArray;
			}

			float[] averageDCostDBiases[] = new float[biases.length][];
			for(int k = 0;k < biases.length;k++) {
				float tempArray[] = new float[biases[k].length];
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
	 * @param decay          The decay to use in regularization
	 * @param fileName       The path to where the model should be saved
	 * @return void
	 * @throws IOException
	 */
	public void customTrain(boolean test, int trainTime, int printLoss, int saveTime, float learningRate, int batchSize, float decay, String fileName) throws IOException {
		for(int i = 0;i <= trainTime;i++) {
			trainSGD(learningRate, batchSize);
			if(i % printLoss == 0) {
				System.out.println("Iteration: " + i + " Loss: " + getCost());
				if(test) {
					System.out.println("Accuracy: " + test() + " / " + testX.length);
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
		for(int i = 0;i < activations.length;i++) {
			for(int j = 0;j < activations[i].length;j++) {
				activations[i][j] = 0;
			}
		}
		float accuracy = 0;
		for(int i = 0;i < testX.length;i++) {
			// Feed data forward
			for(int j = 0;j < activations.length - 1;j++) {
				for(int k = 0;k < activations[j].length;k++) {
					if(j == 0) {
						activations[j][k] = testY[i][k];
					} else {
						activations[j][k] = Util.activate(Util.calculate(activations[j - 1], weights[j - 1][k], biases[j - 1][k]), activation);
					}
				}
			}

			float targets[] = new float[neuronNumbers[neuronNumbers.length - 1]];
			for(int j = 0;j < targets.length;j++) {
				targets[targets.length - (j + 1)] = testX[i][testX[i].length - (j + 1)];
			}
			for(int j = 0;j < activations[activations.length - 1].length;j++) {
				activations[activations.length - 1][j] = Util.calculate(activations[activations.length - 2], weights[weights.length - 1][j], biases[biases.length - 1][j]);
			}
			float activatedOutputs[] = new float[neuronNumbers[neuronNumbers.length - 1]];
			if(activation != Activation.SOFTMAX) {
				for(int j = 0;j < activatedOutputs.length;j++) {
					activatedOutputs[j] = Util.activate(activations[activations.length - 1][j], activation);
				}
			} else {
				activatedOutputs = Util.softmax(activations[activations.length - 1]);
			}
			boolean correct = true;
			for(int j = 0;j < neuronNumbers[neuronNumbers.length - 1];j++) {
				if((float) Math.round(activatedOutputs[j]) != targets[j]) {
					correct = false;
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
		float newOutputs[] = new float[neuronNumbers[neuronNumbers.length - 1]];
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
					for(int j = 0;j < neuronNumbers[1];j++) {
						activations[1][j] = Util.activate(Util.calculate(inputs, weights[i][j], biases[i][j]), activation);
					}
				} else {
					for(int j = 0;j < neuronNumbers[i + 1];j++) {
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
	 * Prints the current weights of the model.
	 * @return void
	 */
	public void printWeights() {
		for (int i = 0; i < weights.length; i++) {
			for (int j = 0; j < weights[i].length; j++) {
				for (int k = 0; k < weights[i][j].length; k++) {
					System.out.println(weights[i][j][k]);
				}
				System.out.println("");
			}
			System.out.println("");
		}
	}
	
	/**
	 * Prints the current biases of the model.
	 * @return void
	 */
	public void printBiases() {
		for(int i = 0;i < biases.length;i++) {
			for(int j = 0;j < biases[i].length;j++) {
				System.out.println(biases[i][j]);
			}
			System.out.println("");
		}
	}
	
	/**
	 * Prints the current parameters of the model.
	 * @return void
	 */
	public void printParameters() {
		System.out.println("Weights:");
		printWeights();
		System.out.println("Biases:");
		printBiases();
	}
	
	/**
	 * Prints the model's prediction on an input vector.
	 * @param x      The input vector
	 * @return void
	 */
	public void printOutputs(float inputs[]) {
		System.out.println("\n" + "Predicted Outputs:");
		for (int i = 0; i < neuronNumbers[neuronNumbers.length - 1]; i++) {
			System.out.println(feedForward(inputs)[i]);
		}
	}
	
	/**
	 * Saves the model to a file.
	 * @param fileName      The path to save the model to
	 * @return void
	 * @throws IOException
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
	 * @throws IOException
	 * @throws ClassNotFoundException
	 */
	public static MLP load(String fileName) throws IOException,  ClassNotFoundException {
        InputStream inStream = new FileInputStream(fileName);
		ObjectInputStream fileObjectIn = new ObjectInputStream(inStream);
        MLP loaded = (MLP) fileObjectIn.readObject();
        fileObjectIn.close();
        return loaded;
	}
}

