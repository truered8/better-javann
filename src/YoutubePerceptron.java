
import java.util.Scanner;

public class YoutubePerceptron {

    public static void main(String[] args) {

        // Initialize dataset of video length, views, and if it was watched.
        double[] totalData[] = {
                { 1, 60000, 1 }, { 7, 200, 0 }, { 0.6, 700000, 1 },
                { 0.8, 120000, 1 }, { 5, 500, 0 }, { 8, 10, 0 },
                { 0.5, 5000000, 1 }, { 6, 1, 0 } };

        for (int i = 0; i < totalData.length; i++)
            totalData[i][1] = Math.log10(totalData[i][1]);

        // Maximum and minimum values for weights and bias
        double max = .1;
        double min = -.1;

        // Rate of gradient descent
        double learning_rate = .2;

        // Initialize weights and bias randomly
        double w0 = min + Math.random() * (max - min);
        double w1 = min + Math.random() * (max - min);
        double b = min + Math.random() * (max - min);

        // Train for 50000 iterations
        for (int j = 0; j < 50000; j++) {

            // Initialize total cost and derivatives
            double total_SE = 0;
            double total_dSE_dw0 = 0;
            double total_dSE_dw1 = 0;
            double total_dSE_db = 0;

            // Iterate through all training examples
            for (int i = 0; i < totalData.length; i++) {

                // Calculate cost
                double x0 = totalData[i][0];
                double x1 = totalData[i][1];
                double actual = totalData[i][2];
                double g = weightedSum(x0, x1, w0, w1, b);
                double f = sigmoid(g);
                double SE = Math.pow((f - actual), 2);

                // Add cost to total cost
                total_SE += SE;

                // Calculate derivatives
                double dSE_df = 2 * (f - actual);
                double df_dg = f * (1 - f);
                double dg_dw0 = x0;
                double dg_dw1 = x1;
                double dg_db = 1;
                double dSE_dw0 = dSE_df * df_dg * dg_dw0;
                double dSE_dw1 = dSE_df * df_dg * dg_dw1;
                double dSE_db = dSE_df * df_dg * dg_db;

                // Add derivatives to total derivatives
                total_dSE_dw0 += dSE_dw0;
                total_dSE_dw1 += dSE_dw1;
                total_dSE_db += dSE_db;

            }

            // Calculate average cost and derivatives
            double MSE = total_SE / totalData.length;
            double dMSE_dw0 = total_dSE_dw0 / totalData.length;
            double dMSE_dw1 = total_dSE_dw1 / totalData.length;
            double dMSE_db = total_dSE_db / totalData.length;

            // Subtract a fraction of the gradient from the weights and bias
            w0 -= learning_rate * dMSE_dw0;
            w1 -= learning_rate * dMSE_dw1;
            b -= learning_rate * dMSE_db;

            // Print error every 10000 iterations
            if (j % 10000 == 0)
                System.out.println("The mean squared error is " + MSE + ".");

        }

        // Print weights and bias
        Scanner keyboard = new Scanner(System.in);
        System.out.println("w0 is equal to " + w0 + ".");
        System.out.println("w1 is equal to " + w1 + ".");
        System.out.println("b is equal to " + b + ".");

        // Gather new input values
        System.out.println("What is the length of the video in minutes?");
        double length = keyboard.nextDouble();
        System.out.println("How many views does the video have?");
        double views = keyboard.nextDouble();

        // Predict whether you will watch the given video
        if (Math.round(sigmoid(weightedSum(length, Math.log10(views), w0, w1, b))) == 1)
            System.out.println("You will watch the video.");
        else
            System.out.println("You won't watch the video.");

        keyboard.close();

    }

    // Mathematical functions

    /**
     * "Squishes" the input to be between 0 and 1.
     * 
     * @param x
     * @return {@code 1 / (1 + e ^ -x)}
     */
    public static double sigmoid(double x) {
        return 1 / (1 + Math.exp(x * -1));
    }

    /**
     * Returns the weighted sum of the inputs.
     * 
     * @param x0 First input value.
     * @param x1 Second input value.
     * @param w0 First weight.
     * @param w0 Second weight.
     * @param b  Bias.
     * @return {@code x0 * w0 + x1 * w1 + b}
     */
    public static double weightedSum(double x0, double x1, double w0, double w1, double b) {
        return x0 * w0 + x1 * w1 + b;
    }

}
