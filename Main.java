import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Main {

	public static void main(String[] args) {
		
		FeedForward ff = new FeedForward("train.txt");
		
		List<List<Double>> outputW = new ArrayList<List<Double>>();
		List<List<Double>> hiddenW = new ArrayList<List<Double>>();
		
		// Initialize Weight
		Double[] zeros;
		for (int i = 0; i < ff.l; i++) {
			zeros = new Double[ff.m+1];
			Arrays.fill(zeros, 0.0);
			hiddenW.add(Arrays.asList(zeros));
		}
		for (int i = 0; i < ff.n; i++) {
			zeros = new Double[ff.l];
			Arrays.fill(zeros, 0.0);
			outputW.add(Arrays.asList(zeros));
		}
		
		ff.setWeights(outputW, hiddenW);
		BackPropagation bp = new BackPropagation(ff.m, ff.l, ff.n, ff.k, ff.x, ff.y, outputW, hiddenW);
		bp.gradientDescent(500, ff);
		
		/*FeedForward feedforward = new FeedForward("test.txt");
		feedforward.setWeights(outputW, hiddenW);
		feedforward.calculateMSE();*/
	}
}
