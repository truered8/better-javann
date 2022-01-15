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
public class NewMLP implements Serializable{

	private static final long serialVersionUID = 1L;
	// Hyperparameters
	String activation;
	String loss;
	
	// Neurons
	private float[][] neurons;
	private int[] neuronNumbers;
	
	// Training and testing data
	private float[] totalData[];
	private float[] totalAttributes[];
	private float[] testData[];
	private float[] testAttributes[];
	
	// Parameters
	private float[][][] weights;
	private float[][] biases;
	
	// Training cost
	private float averageCost;
	
	// Initializes variables that define the structure of the network
	public NewMLP(int[] neuronNumbers,String activation,String loss) {
		this.neuronNumbers = neuronNumbers;
		neurons = new float[neuronNumbers.length][];
		for(int i = 0;i < neurons.length;i++) {
			neurons[i] = new float[neuronNumbers[i]];
		}
		this.activation = activation;
		this.loss = loss;
	}
	
	/**
	 * Retrieves data from file.
	 * @param fileName - path to the data
	 * @throws Exception
	 */
	public void getCSVData(String fileName) throws Exception {
		// Copy values from CSV file into totalData array
		int columns = 0;
		int row = 0;
		String inLine = "";

		Scanner reader = new Scanner(new BufferedReader(new FileReader("C:/Users/Babtu/Documents/Datasets/" + fileName)));
		BufferedReader read = new BufferedReader(new FileReader("C:/Users/Babtu/Documents/Datasets/" + fileName));
		while(read.readLine() != null) {
			columns += 1;
		}

		float[] totalData1[] = 
			new float[columns - 1][reader.nextLine().split(",").length];
		while(reader.hasNextLine()) {
			inLine = reader.nextLine();
			String[] inArray = inLine.split(",");
			for(int x = 0;x < inArray.length;x++) {
				if(inArray[x].equals("?")) {
					totalData1[row][x] = -99999;
				} else {
					totalData1[row][x] = Float.parseFloat(inArray[x]);
				}
			}
			row++;
		}

		// Copy attributes from totalData to totalAttributes
		float[] totalAttributes1[] = new float[totalData1.length][totalData1[0].length - neuronNumbers[neuronNumbers.length - 1]];
		for(int i = 0;i < totalData1.length;i++) {
			for(int j = 0;j < totalData1[i].length - neuronNumbers[neuronNumbers.length - 1];j++) {
				totalAttributes1[i][j] = totalData1[i][j];
			}
		}
		totalData = totalData1;
		totalAttributes = totalAttributes1;
		reader.close();
		read.close();
	}
	public void getData(float[][] data) {

		float[] totalAttributes1[] = new float[data.length][data[0].length - neuronNumbers[neuronNumbers.length - 1]];
		for(int i = 0;i < data.length;i++) {
			for(int j = 0;j < data[i].length - neuronNumbers[neuronNumbers.length - 1];j++) {
				totalAttributes1[i][j] = data[i][j];
			}
		}
		totalData = data;
		totalAttributes = totalAttributes1;
	}
	public void getCSVTestData(String fileName) throws Exception {
		// Copy values from CSV file into totalData array
		int columns = 0;
		int row = 0;
		String inLine = "";

		Scanner reader = new Scanner(new BufferedReader(new FileReader("C:/Users/Babtu/Documents/Datasets/" + fileName)));
		BufferedReader read = new BufferedReader(new FileReader("C:/Users/Babtu/Documents/Datasets/" + fileName));
		while(read.readLine() != null) {
			columns += 1;
		}

		float[] totalData1[] = new float[columns - 1][reader.nextLine().split(",").length];
		while(reader.hasNextLine()) {
			inLine = reader.nextLine();
			String[] inArray = inLine.split(",");
			for(int x = 0;x < inArray.length;x++) {
				if(inArray[x].equals("?")) {
					totalData1[row][x] = -99999;
				} else {
					totalData1[row][x] = Float.parseFloat(inArray[x]);
				}
			}
			row++;
		}
		testData = totalData1;
		reader.close();
		read.close();
	}
	public void getTestData(float[][] data) {
		float[] testAttributes1[] = new float[data.length][data[0].length - neuronNumbers[neuronNumbers.length - 1]];
		for(int i = 0;i < data.length;i++) {
			for(int j = 0;j < data[i].length - neuronNumbers[neuronNumbers.length - 1];j++) {
				testAttributes1[i][j] = data[i][j];
			}
		}
		testData = data;
		testAttributes = testAttributes1;
	}
	public float[][] toOneHot(String fileName,int dataLength) throws Exception {
		float[] trainingData[] = new float[dataLength][neuronNumbers[0] + neuronNumbers[neuronNumbers.length - 1]];
		Scanner train = new Scanner(new BufferedReader(new FileReader("C:/Users/Babtu/Documents/Datasets/" + fileName)));
		train.nextLine();
		for(int i = 0;i < trainingData.length;i++) {
			String stringValues[] = train.nextLine().split(",");
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
	
	// Normalizes data
	public float[][] normalize(float[][] x,float range) {
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
	
	// Getters and setters
	public float[][] getTotalData() {
		return totalData;
	}
	public float[][] getTotalAttributes() {
		return totalAttributes;
	}
	public float[][] getTestingData() {
		return testData;
	}
	public float[][][] getWeights() {
		return weights;
	}
	public float[][] getBiases() {
		return biases;
	}

	public void setData(int i, int j, float newData) {
		this.totalData[i][j] = newData;
	}
	public void setWeight(int i,int j,int k,float newWeight) {
		weights[i][j][k] = newWeight;
	}
	public void setBias(int i,int j,float newBias) {
		biases[i][j] = newBias;
	}

	// Initializes weights and biases randomly for training
	public void createNetwork() {
		
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
		
	// Retrieves weights and biases from text files
	public void setFileWeights(String fileName) throws IOException {
		Scanner reader = new Scanner(new BufferedReader(new FileReader("C:/Users/Babtu/Documents/" + fileName)));
		int layer = 0;
		int neuron = 0;
		int lastNeuron = 0;
		int line = 1;
		String currentLine = "";
		String lastLine = null;
		while(reader.hasNextLine()) {
			currentLine = reader.nextLine();
			if(line != 1 && currentLine.equals("")) {
				lastNeuron = 0;
				neuron += 1;
				if(lastLine.equals("")) {
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
	public void setFileBiases(String fileName) throws IOException {
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
	
	// Completes one iteration of Stochastic gradient descent
	public void trainSGD(float learningRate,int batchSize) {
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
	    for(int i = totalData.length - 1; i > 0; i--) {
	      int index = rnd.nextInt(i + 1);
	      float[] a = totalData[index];
	      totalData[index] = totalData[i];
	      totalData[i] = a;
	    }
	    float[] totalAttributes1[] = new float[totalData.length][totalData[0].length - neuronNumbers[neuronNumbers.length - 1]];
		for(int i = 0;i < totalData.length;i++) {
			for(int j = 0;j < totalData[i].length - neuronNumbers[neuronNumbers.length - 1];j++) {
				totalAttributes1[i][j] = totalData[i][j];
			}
		}
	    totalAttributes = totalAttributes1;

		// Organize data into batches for Stochastic gradient descent
		float[][] totalDataBatches[];
		if(totalData.length % batchSize == 0) {
			totalDataBatches = new float[(int)(totalData.length / batchSize)][][];
		} else {
			totalDataBatches = new float[(int)(totalData.length / batchSize) + 1][][];
		}
		int totalDataExamples = 0;
		for(int i = 0;i < totalDataBatches.length;i++) {
			int examples = 0;
			if(i != totalDataBatches.length - 1) {
				examples = batchSize;
			} else if(totalData.length % batchSize == 0) {
				examples = batchSize;
			} else {
				examples = totalData.length % batchSize;
			}
			float[][] currentExamples = new float[examples][];
			for(int j = 0;j < examples;j++) {
				currentExamples[j] = totalData[totalDataExamples];
				totalDataExamples++;
			}
			totalDataBatches[i] = currentExamples;
		}
		float[][] totalAttributesBatches[] = new float[(int)(totalData.length / batchSize) + 1][][];
		int totalAttributesExamples = 0;
		for(int i = 0;i < totalAttributesBatches.length;i++) {
			int examples = 0;
			if(i != totalAttributesBatches.length - 1) {
				examples = batchSize;
			} else {
				examples = totalAttributes.length % batchSize;
			}
			float[][] currentExamples = new float[examples][];
			for(int j = 0;j < examples;j++) {
				currentExamples[j] = totalAttributes[totalAttributesExamples];
				totalAttributesExamples++;
			}
			totalAttributesBatches[i] = currentExamples;
		}
			
		// Initialize arrays of total derivatives; e.g. totalDCostDWeights means the total derivative of the cost with respect to the weights
		float totalCost = 0;
		float[][] totalDCostDWeights[] = new float[weights.length][][];
		for(int k = 0;k < weights.length;k++) {
			float[] tempArray[] = new float[weights[k].length][neurons[k].length];
			totalDCostDWeights[k] = tempArray;
		}

		float[] totalDCostDBiases[] = new float[biases.length][];
		for(int k = 0;k < biases.length;k++) {
			float tempArray[] = new float[biases[k].length];
			totalDCostDBiases[k] = tempArray;
		}

		for(int j = 0;j < totalDataBatches.length;j++) {
			// Iterate through all training examples in the batch
			for(int k = 0;k < totalDataBatches[j].length;k++) {
				// Feed data forward
				for(int l = 0;l < neurons.length - 1;l++) {
					for(int m = 0;m < neurons[l].length;m++) {
						if(l == 0) {
							neurons[l][m] = totalAttributesBatches[j][k][m];
						} else if(!activation.equals("softmax")){
							neurons[l][m] = activate(calculate(neurons[l - 1],weights[l - 1][m],biases[l - 1][m]),activation);
						} else {
							neurons[l][m] = calculate(neurons[l - 1],weights[l - 1][m],biases[l - 1][m]);
						}
					}
				}

				// Compute cost
				float targets[] = new float[neuronNumbers[neuronNumbers.length - 1]];
				for(int l = 0;l < targets.length;l++) {
					targets[targets.length - (l + 1)] = totalDataBatches[j][k][totalDataBatches[j][k].length - (l + 1)];
				}
				for(int l = 0;l < neurons[neurons.length - 1].length;l++) {
					neurons[neurons.length - 1][l] = calculate(neurons[neurons.length - 2],weights[weights.length - 1][l],biases[biases.length - 1][l]);
					if(Float.isNaN(neurons[neurons.length - 1][l])) {
						System.out.print(weights[0][0][0]);
					}
				}
				float activatedOutputs[] = new float[neuronNumbers[neuronNumbers.length - 1]];
				if(!activation.equals("softmax")) {
					for(int l = 0;l < activatedOutputs.length;l++) {
						activatedOutputs[l] = activate(neurons[neurons.length - 1][l],activation);
					}
				} else {
					activatedOutputs = softmax(neurons[neurons.length - 1]);
				}
				float cost = 0;
				for(int l = 0;l < neuronNumbers[neuronNumbers.length - 1];l++) {
					cost += loss(activatedOutputs[l],targets[l],loss);
				}
				if(loss.equals("CE")) {
					cost *= -1;
				}
				
				// Initialize and compute arrays of derivatives
				// The derivative of the cost function
				float dCostDActivatedOutputs[] = new float[neuronNumbers[neuronNumbers.length - 1]];
				for(int l = 0;l < dCostDActivatedOutputs.length;l++) {
					if(loss.equals("MSE")) {
						dCostDActivatedOutputs[l] = 2 * (activatedOutputs[l] - targets[l]);
					} else if(loss.equals("CE")) {
						dCostDActivatedOutputs[l] = activatedOutputs[l] - targets[l];
					}
				}

				// The derivative of the activation function
				float[] dActivatedOutputsDOutputs[] = new float[neurons.length - 1][];
				for(int l = 0;l < dActivatedOutputsDOutputs.length;l++) {
					float tempArray[] = new float[neuronNumbers[neuronNumbers.length - (l + 1)]];
					dActivatedOutputsDOutputs[l] = tempArray;
				}
				for(int l = 0;l < dActivatedOutputsDOutputs.length;l++) {
					for(int m = 0;m < dActivatedOutputsDOutputs[l].length;m++) {
						if(activation.equals("sigmoid")) {
							dActivatedOutputsDOutputs[l][m] = activate(neurons[neurons.length - (l + 1)][m],activation) * (1 - activate(neurons[neurons.length - (l + 1)][m],activation));
						} else if(activation.equals("softplus")) {
							if(l == 0) {
								dActivatedOutputsDOutputs[0][0] = activate(neurons[neurons.length - 1][m],"sigmoid");
							} else {
								dActivatedOutputsDOutputs[l][m] = activate(neurons[neurons.length - (l + 1)][m],"sigmoid");
							}
						} else if(activation.equals("softmax")) {
							if(l == 0) {
								dActivatedOutputsDOutputs[l][m] = 1;// activatedOutputs[m] - totalDataBatch[k][m + neuronNumbers[0]];
							} else {
								dActivatedOutputsDOutputs[l][m] = 1;// activate(neurons[neurons.length - (l + 1)][m],activation) * (1 - activate(neurons[neurons.length - (l + 1)][m],activation));
							}
						} else if(activation.equals("ReLU")) {
							if(neurons[neurons.length - (l + 1)][m] > 0) {
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
					float[] tempArray[] = new float[weights[l].length][neurons[l].length];
					dOutputsDWeights[l] = tempArray;
				}
				for(int l = 0;l < dOutputsDWeights.length;l++) {
					for(int m = 0;m < dOutputsDWeights[l].length;m++) {
						for(int n = 0;n < dOutputsDWeights[l][m].length;n++) {
							dOutputsDWeights[l][m][n] = neurons[l][n];
						}
					}
				}

				// The derivative of the cost function with respect to the weights and biases uses the chain rule by multiplying the derivatives of each of the functions being applied to the weights and biases
				// For example, the derivative of cost(sigmoid(calculate(weight))) is dCost * dSigmoid * dCalculate 
				float[][] dCostDWeights[] = new float[weights.length][][];
				for(int l = 0;l < weights.length;l++) {
					float[] tempArray[] = new float[weights[l].length][neurons[l].length];
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
			averageCost = totalCost / totalDataBatches[j].length;

			// Initialize and compute arrays of average derivatives
			float[][] averageDCostDWeights[] = new float[weights.length][][];
			for(int k = 0;k < weights.length;k++) {
				float[] tempArray[] = new float[weights[k].length][neurons[k].length];
				for(int l = 0;l < tempArray.length;l++) {
					for(int m = 0;m < tempArray[l].length;m++) {
						tempArray[l][m] = totalDCostDWeights[k][l][m] / totalDataBatches[j].length;
					}
				}
				averageDCostDWeights[k] = tempArray;
			}

			float[] averageDCostDBiases[] = new float[biases.length][];
			for(int k = 0;k < biases.length;k++) {
				float tempArray[] = new float[biases[k].length];
				for(int l = 0;l < tempArray.length;l++) {
					tempArray[l] = totalDCostDBiases[k][l] / totalDataBatches[j].length;
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

	// Customizable implementation of SGD
	public void customTrain(boolean test,int trainTime,int printError,int saveTime,int batchSize,float learningRate,float decay,String fileName) throws IOException {
		for(int i = 0;i <= trainTime;i++) {
			trainSGD(learningRate,batchSize);
			if(i % printError == 0) {
				System.out.println("Iteration: " + i);
				System.out.println("Loss: " + getCost());
				if(test) {
					System.out.println("Accuracy: " + test() + " / " + testData.length);
				}
			}
			if(saveTime != 0 && i % saveTime == 0) {
				save(fileName);
			}
		}
	}
	
	// Returns accuracy for test data
	public float test() {
		for(int i = 0;i < neurons.length;i++) {
			for(int j = 0;j < neurons[i].length;j++) {
				neurons[i][j] = 0;
			}
		}
		float accuracy = 0;
		for(int i = 0;i < testData.length;i++) {
			// Feed data forward
			for(int j = 0;j < neurons.length - 1;j++) {
				for(int k = 0;k < neurons[j].length;k++) {
					if(j == 0) {
						neurons[j][k] = testAttributes[i][k];
					} else {
						neurons[j][k] = activate(calculate(neurons[j - 1],weights[j - 1][k],biases[j - 1][k]),activation);
					}
				}
			}

			float targets[] = new float[neuronNumbers[neuronNumbers.length - 1]];
			for(int j = 0;j < targets.length;j++) {
				targets[targets.length - (j + 1)] = testData[i][testData[i].length - (j + 1)];
			}
			for(int j = 0;j < neurons[neurons.length - 1].length;j++) {
				neurons[neurons.length - 1][j] = calculate(neurons[neurons.length - 2],weights[weights.length - 1][j],biases[biases.length - 1][j]);
			}
			float activatedOutputs[] = new float[neuronNumbers[neuronNumbers.length - 1]];
			if(!activation.equals("softmax")) {
				for(int j = 0;j < activatedOutputs.length;j++) {
					activatedOutputs[j] = activate(neurons[neurons.length - 1][j],activation);
				}
			} else {
				activatedOutputs = softmax(neurons[neurons.length - 1]);
			}
			boolean correct = true;
			for(int j = 0;j < neuronNumbers[neuronNumbers.length - 1];j++) {
				if(step(activatedOutputs[j]) != targets[j]) {
					correct = false;
				}
			}
			if(correct) {
				accuracy += 1;
			}
		}
		return accuracy;
	
	}
	
	public float getCost() {
		return averageCost;
	}
	
	// Returns an array of outputs calculated by the network
	public float[] feedForward(float[] inputs) {
		float newOutputs[] = new float[neuronNumbers[neuronNumbers.length - 1]];
		if(neurons.length == 2) {
			if(!activation.equals("softmax")) {
				for(int i = 0;i < newOutputs.length;i++) {
					newOutputs[i] = activate(calculate(inputs,weights[0][i],biases[0][i]),activation);
				}
			} else {
				for(int i = 0;i < newOutputs.length;i++) {
					newOutputs[i] = calculate(inputs,weights[0][i],biases[0][i]);
				}
				newOutputs = softmax(newOutputs);
			}
		} else {
			for(int i = 0;i < neurons.length - 2;i++) {
				if(i == 0) {
					for(int j = 0;j < neuronNumbers[1];j++) {
						neurons[1][j] = activate(calculate(inputs,weights[i][j],biases[i][j]),activation);
					}
				} else {
					for(int j = 0;j < neuronNumbers[i + 1];j++) {
						neurons[i + 1][j] = activate(calculate(neurons[i - 1],weights[i][j],biases[i][j]),activation);
					}
				}
			}
			if(!activation.equals("softmax")) {
				for(int i = 0;i < newOutputs.length;i++) {
					newOutputs[i] = activate(calculate(neurons[neurons.length - 2],weights[weights.length - 1][i],biases[biases.length - 1][i]),activation);
				}
			} else {
				for(int i = 0;i < newOutputs.length;i++) {
					newOutputs[i] = calculate(neurons[neurons.length - 2],weights[weights.length - 1][i],biases[biases.length - 1][i]);
				}
				newOutputs = softmax(newOutputs);
			}
		}
		return newOutputs;
	}

	// Mathematical functions
	public static float activate(float x,String function) {
		float result = 0;
		if(function.equals("sigmoid") || function.equals("softmax")) {
			result = (float) (1 / (1 + Math.exp(x * -1)));
		} else if(function.equals("abs")) {
			result = Math.abs(x);
		} else if(function.equals("softplus")) {
			result = (float) Math.log(1 + Math.exp(x));
		} else if(function.equals("ReLU")) {
			result = Math.max(0,x);
		} else {
			result = x;
		}
		return result;
	}
	public static float[] softmax(float[] x) {
		float sum = 0;
		float exponentials[] = new float[x.length];
		for(int i = 0;i < x.length;i++) {
			exponentials[i] = Math.min((float)Math.exp(x[i]),Float.MAX_VALUE);
			sum += exponentials[i];
		}
		if(Float.isInfinite((sum))) {
			sum = Float.MAX_VALUE;
		}
		float[] activations = new float[x.length];
		for(int i = 0;i < activations.length;i++) {
			activations[i] = (float)(exponentials[i] / sum);
			if(activations[i] == 0) {
				activations[i] = (float)1e-35;
			}
			if(Float.isNaN(activations[i])) {
				System.out.println("NaN Activation");
			}
		}
		return activations;
	}
	public static float calculate(float[] n,float[] w,float b) {
		return dotProduct(n,w) + b;
	}
	public static float step(float x) {
		return (float)Math.round(x); 
	}
	public static float dotProduct(float[] x,float[] y) {
		float sum = 0;
		for(int i = 0;i < x.length;i++) {
			sum += x[i] * y[i];
		}
		return sum;
	}
	public static float loss(float x,float y,String function) {
		float loss = 23;
		if(function.equals("MSE")) {
			loss = (float)Math.pow(x - y, 2);
		} else if(function.equals("CE")) {
			loss = (float)Math.log(x) * y;
		}
		return loss;
	}
	
	// Prints weights and biases
	public void printWeights() {
		for(int i = 0;i < weights.length;i++) {
			for(int j = 0;j < weights[i].length;j++) {
				for(int k = 0;k < weights[i][j].length;k++) {
					System.out.println(weights[i][j][k]);
				}
				System.out.println("");
			}
			System.out.println("");
		}
	}
	public void printBiases() {
		for(int i = 0;i < biases.length;i++) {
			for(int j = 0;j < biases[i].length;j++) {
				System.out.println(biases[i][j]);
			}
			System.out.println("");
		}
	}
	public void printParameters() {
		System.out.println("Weights:");
		printWeights();
		System.out.println("Biases:");
		printBiases();
	}
	public void printOutputs(float inputs[]) {
		System.out.println("\n" + "Predicted Outputs:");
		for(int i = 0;i < neuronNumbers[neuronNumbers.length - 1];i++) {
			System.out.println(feedForward(inputs)[i]);
		}
	}
	public void save(String fileName) throws IOException {
        OutputStream outStream = new FileOutputStream(fileName);
        ObjectOutputStream fileObjectOut = new ObjectOutputStream(outStream);
        fileObjectOut.writeObject(this);
        fileObjectOut.close();
        outStream.close();
	}
	public static NewMLP load(String fileName) throws IOException, ClassNotFoundException {
        InputStream inStream = new FileInputStream(fileName);
		ObjectInputStream fileObjectIn = new ObjectInputStream(inStream);
        NewMLP loaded = (NewMLP)fileObjectIn.readObject();
        fileObjectIn.close();
        return loaded;
	}
}

