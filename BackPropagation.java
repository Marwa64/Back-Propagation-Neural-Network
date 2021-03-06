import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class BackPropagation {
	static int m, l, n, k; // m = input nodes, l = hidden nodes, n = output nodes, k = training examples
	static List<List<Double>> x = new ArrayList<List<Double>>();
	static List<List<Double>> y = new ArrayList<List<Double>>();
	static List<List<Double>> outputW = new ArrayList<List<Double>>();
	static List<List<Double>> hiddenW = new ArrayList<List<Double>>();
	static List<Double> hiddenLayerNodes = new ArrayList<Double>();
	static List<Double> outputLayerNodes = new ArrayList<Double>();
	static double MSE = 0.0, learningRate = 0.5;
	
	BackPropagation(int m, int l, int n, int k, List<List<Double>> x, List<List<Double>> y) {
		this.m = m;
		this.l = l;
		this.n = n;
		this.k = k;
		this.x = x;
		this.y = y;
		
		// Initialize Weight
		for (int i = 0; i < l; i++) {
			List<Double> temp = new ArrayList<Double>();
			for (int j = 0; j < m+1; j++) {
				double random = Math.random() * (0.8-(-0.3)) + (-0.3);
				temp.add(random);
			}
			hiddenW.add(temp);
		}
		for (int i = 0; i < n; i++) {
			List<Double> temp = new ArrayList<Double>();
			for (int j = 0; j < l; j++) {
				double random = Math.random() * (0.8-(-0.3)) + (-0.3);
				temp.add(random);
			}
			outputW.add(temp);
		}
	}

	public void gradientDescent(int iterations, FeedForward feedforward) {
		for (int itr = 0; itr < iterations; itr++) {
			double cost = 0.0;
			// Loop over the training examples
			for (int i = 0; i < k; i++) {
				
				feedforward.setWeights(outputW, hiddenW);
				this.hiddenLayerNodes = feedforward.calculateHidden(i);
				this.outputLayerNodes = feedforward.calculateOutput();
				
				List<Double> outputDelta = new ArrayList<Double>();
				List<Double> hiddenDelta = new ArrayList<Double>();
				
				// Calculate delta for output neurons
				for (int j = 0; j < n; j++) {
					double delta = (outputLayerNodes.get(j) - y.get(i).get(j)) * outputLayerNodes.get(j) * (1 - outputLayerNodes.get(j));
					outputDelta.add(delta);
				}
				
				// Calculate delta for hidden neurons
				for (int j = 0; j < l; j++) {
					double sum = 0.0, delta = 0.0;
					for (int j2 = 0; j2 < n; j2++) {
						sum += outputDelta.get(j2) * outputW.get(j2).get(j);
					}
					delta = sum * hiddenLayerNodes.get(j) * (1 - hiddenLayerNodes.get(j));
					hiddenDelta.add(delta);
				}
				
				// Update output weight
				for (int j = 0; j < n; j++) {
					for (int j2 = 0; j2 < l; j2++) {
						double newWeight = outputW.get(j).get(j2) - learningRate * outputDelta.get(j) * hiddenLayerNodes.get(j2);
						outputW.get(j).set(j2, newWeight);
					}
				}
				
				// Update hidden weight
				for (int j = 0; j < l; j++) {
					for (int j2 = 0; j2 < m+1; j2++) {
						double newWeight = hiddenW.get(j).get(j2) - learningRate * hiddenDelta.get(j) * x.get(i).get(j2);
						hiddenW.get(j).set(j2, newWeight);
					}
				}
				cost += costFunction(i);
			}
			MSE = cost / (double) k;
			System.out.println("Iteration " + itr + " : Cost = " + cost);
		}
		System.out.println("\n---------------------\n");
		System.out.println("MSE = " + MSE);
	}
	
	public static double costFunction(int trainingIndex) {
		double sum = 0.0;
		for (int i = 0; i < n; i++) {
			sum += Math.pow(outputLayerNodes.get(i) - y.get(trainingIndex).get(i), 2.0);
		}
		return 0.5 * sum;
	}
	
	public void saveWeights() {
	    try {
	        FileWriter myWriter = new FileWriter("weights.txt");
	        myWriter.write("Hidden Weight:\n");
	        for (int i = 0; i < hiddenW.size(); i++) {
	        	myWriter.write(hiddenW.get(i).toString());
	        	myWriter.write("\n");
	        }

	        myWriter.write("Output Weight:\n");
	        for (int i = 0; i < outputW.size(); i++) {
	        	myWriter.write(outputW.get(i).toString());
	        	if (i != outputW.size()-1)
	        		myWriter.write("\n");
	        }
	        myWriter.close();
	        System.out.println("Weights Saved.");
	      } catch (IOException e) {
	        System.out.println("An error occurred.");
	        e.printStackTrace();
	      }
	}

	public static void main(String[] args) {
		
		FeedForward ff = new FeedForward("train.txt");
		BackPropagation bp = new BackPropagation(ff.m, ff.l, ff.n, ff.k, ff.x, ff.y);
		bp.gradientDescent(500, ff);
		bp.saveWeights();
	}
}
