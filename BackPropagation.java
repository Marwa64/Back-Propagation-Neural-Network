import java.util.ArrayList;
import java.util.List;

public class BackPropagation {
	static int m, l, n, k; // m = input nodes, l = hidden nodes, n = output nodes, k = training examples
	static List<List<Double>> x = new ArrayList<List<Double>>();
	static List<List<Double>> y = new ArrayList<List<Double>>();
	static List<List<Double>> outputW = new ArrayList<List<Double>>();
	static List<List<Double>> hiddenW = new ArrayList<List<Double>>();
	static List<Double> hiddenLayerNodes = new ArrayList<Double>();
	static List<List<Double>> outputLayerNodes = new ArrayList<List<Double>>();
	static double MSE = 0.0, learningRate = 0.002;
	
	BackPropagation(int m, int l, int n, int k, List<List<Double>> x, List<List<Double>> y, List<List<Double>> outputW, List<List<Double>> hiddenW, List<Double> hiddenLayerNodes, List<List<Double>> outputLayerNodes) {
		this.m = m;
		this.l = l;
		this.n = n;
		this.k = k;
		this.x = x;
		this.y = y;
		this.outputW = outputW;
		this.hiddenW = hiddenW;
		this.hiddenLayerNodes = hiddenLayerNodes;
		this.outputLayerNodes = outputLayerNodes;
	}
	
	public void gradientDescent(int iterations, FeedForward ff) {
		
		for (int itr = 0; itr < iterations; itr++) {
			double cost = 0.0;
			// Loop over the training examples
			for (int i = 0; i < k; i++) {
				
				ff.setWeights(outputW, hiddenW);
				this.hiddenLayerNodes = ff.calculateHidden(i);
				this.outputLayerNodes = ff.calculateOutput();
				
				//System.out.println(outputLayerNodes);
				
				List<Double> outputDelta = new ArrayList<Double>();
				List<Double> hiddenDelta = new ArrayList<Double>();
				// Calculate delta for output neurons
				for (int j = 0; j < n; j++) {
					double delta = (outputLayerNodes.get(i).get(j) - y.get(i).get(j)) * outputLayerNodes.get(i).get(j) * (1 - outputLayerNodes.get(i).get(j));
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
						double newWeight = outputW.get(j).get(j2) - (learningRate * outputDelta.get(j) * hiddenLayerNodes.get(j2));
						outputW.get(j).set(j2, newWeight);
					}
				}
				//System.out.println(outputW);
				
				// Update hidden weight
				for (int j = 0; j < l; j++) {
					for (int j2 = 0; j2 < m+1; j2++) {
						double newWeight = hiddenW.get(j).get(j2) - (learningRate * hiddenDelta.get(j) * x.get(i).get(j2));
						hiddenW.get(j).set(j2, newWeight);
					}
				}
				//System.out.println(hiddenW);
				cost += costFunction(i);
				//System.out.println("Iteration " + itr + " : example " + i + " cost = " + cost);
			}
			MSE = cost / (double) k;
			//System.out.println("Iteration " + itr + " : Cost = " + cost);
		}
	}
	
	public static double costFunction(int trainingIndex) {
		double sum = 0.0;
		for (int i = 0; i < n; i++) {
			sum += Math.pow(outputLayerNodes.get(trainingIndex).get(i) - y.get(trainingIndex).get(i), 2.0);
		}
		return 0.5 * sum;
	}

}
