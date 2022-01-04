import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class FeedForward {
	static int m, l, n, k; // m = input nodes, l = hidden nodes, n = output nodes, k = training examples
	static List<List<Double>> x = new ArrayList<List<Double>>();
	static List<List<Double>> y = new ArrayList<List<Double>>();
	static List<List<Double>> outputW = new ArrayList<List<Double>>();
	static List<List<Double>> hiddenW = new ArrayList<List<Double>>();
	static List<Double> hiddenLayerNodes = new ArrayList<Double>();
	static List<Double> outputLayerNodes = new ArrayList<Double>();
	static double MSE = 0.0;
	
	FeedForward(String fileName) {
		initializeData(fileName);
		normalizeData();
	}
	
	private static void initializeData(String fileName) {
		try  {  
			File file=new File(fileName); 
			FileReader fr=new FileReader(file);
			BufferedReader br=new BufferedReader(fr); 
			String line;  
			int lineNum = 1;
			while((line=br.readLine())!=null)  {
				if (lineNum == 1) {
					m = Integer.parseInt(line.split(" ")[0]);
					l = Integer.parseInt(line.split(" ")[1]);
					n = Integer.parseInt(line.split(" ")[2]);
				} else if (lineNum == 2) {
					k = Integer.parseInt(line);
				} else {
					String [] data = line.trim().split("\\s+");
					List<Double> inputSet = new ArrayList<Double>();
					inputSet.add(1.0);
					List<Double> outputSet = new ArrayList<Double>();
					for (int i = 0; i < data.length; i++) {
						double num = Double.parseDouble(data[i]);
						if (i >= m) {
							outputSet.add(num);
						} else {
							inputSet.add(num);
						}
					}
					x.add(inputSet);
					y.add(outputSet);
				}
				lineNum++;
			}  
			fr.close();
		}  catch(IOException e)  {  
			e.printStackTrace();  
		}
	}
	
	public void setWeights(List<List<Double>> outputW, List<List<Double>> hiddenW) {
		this.outputW = outputW;
		this.hiddenW = hiddenW;
	}
	
	private static void normalizeData() {
		Double[] zeros = new Double[m];
		Arrays.fill(zeros, 0.0);
		// Calculate Mean
		List<Double> mean = Arrays.asList(zeros);
		Double[] fillY = new Double[n];
		Arrays.fill(fillY, 10000.0);
		List<Double> minY = Arrays.asList(fillY);
		fillY = new Double[n];
		Arrays.fill(fillY, 0.0);
		List<Double> maxY = Arrays.asList(fillY);
		for (int i = 0; i < k; i++) {
			for (int j = 1; j < m; j++) {
				double newSum = mean.get(j-1) + x.get(i).get(j);
				mean.set(j-1, newSum);
				
				if (i == k-1) {
					mean.set(j-1, newSum/k);
				}
			}
			// Get the min and max of output y
			for (int j = 0; j < n; j++) {
				if (y.get(i).get(j) < minY.get(j)) {
					minY.set(j, y.get(i).get(j));
				}
				if (y.get(i).get(j) > maxY.get(j)) {
					maxY.set(j, y.get(i).get(j));
				}
			}
		}
		// Calculate Standard Deviation
		zeros = new Double[m];
		Arrays.fill(zeros, 0.0);
		List<Double> std = Arrays.asList(zeros);
		for (int i = 0; i < k; i++) {
			for (int j = 1; j < m; j++) {
				double temp = std.get(j-1) + Math.pow((x.get(i).get(j) - mean.get(j-1)), 2.0);
				std.set(j-1, temp);
				
				if (i == k-1) {
					std.set(j-1, Math.sqrt(temp));
				}
			}
		}
		// Calculate normalized values
		for (int i = 0; i < k; i++) {
			// Normalize X values
			for (int j = 1; j < m; j++) {
				Double newVal = (x.get(i).get(j) - mean.get(j-1)) / std.get(j-1);
				x.get(i).set(j, newVal);
			}
			// Normalize Y values
			for (int j = 0; j < n; j++) {
				double newVal = (y.get(i).get(j) - minY.get(j)) / (maxY.get(j) - minY.get(j));
				y.get(i).set(j, newVal);
			}
		}
	}
	
	// Activation Function
	private static double sigmoid(double num) {
		return 1.0 / (1.0 + Math.exp((-1.0 * num)));
	}
	
	public static List<Double> calculateHidden(int trainingIndex) {
		hiddenLayerNodes = new ArrayList<Double>();
		for (int i = 0; i < l; i++) { // loop over hidden layer neurons
			double param = 0.0;
			for (int j = 0; j < m; j++) { // loop over features
				param += (hiddenW.get(i).get(j) * x.get(trainingIndex).get(j));
			}
			hiddenLayerNodes.add(sigmoid(param));
		}
		return hiddenLayerNodes;
	}
	
	public static List<Double> calculateOutput() {
		outputLayerNodes = new ArrayList<Double>();
		for (int i = 0; i < n; i++) {
			double param = 0.0;
			for (int j = 0; j < l; j++) {
				param += (outputW.get(i).get(j) * hiddenLayerNodes.get(j));
			}
			outputLayerNodes.add(sigmoid(param));
		}
		return outputLayerNodes;
	}
	
	public static double costFunction(int trainingIndex) {
		double sum = 0.0;
		for (int i = 0; i < n; i++) {
			//System.out.println(outputLayerNodes.get(i) + " vs " + y.get(trainingIndex).get(i));
			sum += Math.pow(outputLayerNodes.get(i) - y.get(trainingIndex).get(i), 2.0);
		}
		return 0.5 * sum;
	}
	
	public static double calculateMSE() {
		double cost = 0.0;
		for (int index = 0; index < k; index++) {
			calculateHidden(index);
			calculateOutput();
			cost += costFunction(index);
		}
		MSE = cost / (double) k;
		System.out.println("MSE = " + MSE);
		return MSE;
	}
	
	public static void main(String[] args) {
		
		FeedForward feedforward = new FeedForward("test.txt");
		feedforward.setWeights(outputW, hiddenW);
		feedforward.calculateMSE();
		
	}
}
