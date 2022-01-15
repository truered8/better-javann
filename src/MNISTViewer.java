import java.awt.AlphaComposite;
import java.awt.Dimension;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.Scanner;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

public class MNISTViewer {

	public static void main(String[] args) throws FileNotFoundException {
		
		int example = 7;
		Scanner train = new Scanner(new BufferedReader(new FileReader("C:/Users/Babtu/Documents/Datasets/mnist_train.csv")));
		for(int i = 0;i < example;i++) {
			train.nextLine();
		}
		String stringValues[] = train.nextLine().split(",");
		int intValues[] = new int[stringValues.length - 1];
		for(int i = 1;i < stringValues.length;i++) {
			intValues[i - 1] = Integer.parseInt(stringValues[i]);
		}
		int count = 0;
		BufferedImage newImage = new BufferedImage(28,28,BufferedImage.TYPE_INT_ARGB);
		for(int i = 0;i < 28;i++) {
			for(int j = 0;j < 28;j++) {
				int argb = 0;
				argb = argb | (255 << 24) | (intValues[count] << 16) | (intValues[count] << 8) | intValues[count];
				newImage.setRGB(j,i,argb);
				count++;
			}
		}
		BufferedImage bigImage = createResizedCopy(newImage,280,280,false);
		int height = 280,width = 280;
		JFrame frame = new JFrame();
        JLabel label = new JLabel(new ImageIcon(bigImage), JLabel.CENTER);
		frame = new JFrame("Display");
		frame.setPreferredSize(new Dimension(height,width));
		frame.setMaximumSize(new Dimension(height,width));
		frame.setMinimumSize(new Dimension(height,width));
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.setFocusable(true);
		frame.setResizable(false);
		frame.setLocationRelativeTo(null);
		frame.add(label);
		frame.setVisible(true);
		train.close();
		
	}
	public static BufferedImage createResizedCopy(Image originalImage,int scaledWidth,int scaledHeight,boolean preserveAlpha) {
        int imageType = preserveAlpha ? BufferedImage.TYPE_INT_RGB : BufferedImage.TYPE_INT_ARGB;
        BufferedImage scaledBI = new BufferedImage(scaledWidth, scaledHeight, imageType);
        Graphics2D g = scaledBI.createGraphics();
        if (preserveAlpha) {
            g.setComposite(AlphaComposite.Src);
        }
        g.drawImage(originalImage, 0, 0, scaledWidth, scaledHeight, null); 
        g.dispose();
        return scaledBI;
    }

}
