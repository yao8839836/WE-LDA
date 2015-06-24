package topic;

import java.util.Map;

public class WeZlabelLDA {

	int[][] documents;

	int V;

	int K;

	double alpha;

	double beta;

	int[][] z;

	double[][] nw;

	int[][] nd;

	// 每个主题下的must-link及其相关度

	Map<Integer, Map<Integer, Double>> urn_Topic_W1_W2_Value;

	double[] nwsum;

	int[] ndsum;

	int iterations;

	double[][] thetasum;

	double[][] phisum;

	int numstats;

	int BURN_IN = 200;

	int SAMPLE_LAG = 20;

	double[][] weight;

	public WeZlabelLDA(int[][] documents, int V,
			Map<Integer, Map<Integer, Double>> urn_Topic_W1_W2_Value) {

		this.documents = documents;
		this.V = V;
		this.urn_Topic_W1_W2_Value = urn_Topic_W1_W2_Value;

	}

	public void initialState() {

		int D = documents.length;
		nw = new double[V][K];
		nd = new int[D][K];
		nwsum = new double[K];
		ndsum = new int[D];

		weight = new double[V][K];

		z = new int[D][];

		for (int d = 0; d < D; d++) {

			int Nd = documents[d].length;

			z[d] = new int[Nd];

			for (int n = 0; n < Nd; n++) {

				int topic = (int) (Math.random() * K);

				z[d][n] = topic;

				updateCount(d, topic, documents[d][n], +1);

			}
		}

	}

	public void markovChain(int K, double alpha, double beta, int iterations) {

		this.K = K;
		this.alpha = alpha;
		this.beta = beta;
		this.iterations = iterations;

		if (SAMPLE_LAG > 0) {
			thetasum = new double[documents.length][K];
			phisum = new double[K][V];
			numstats = 0;
		}

		initialState();

		for (int i = 0; i < this.iterations; i++) {

			gibbs();

			if (i >= BURN_IN && SAMPLE_LAG > 0 && i % SAMPLE_LAG == 0) {

				updateParams();
			}
		}
	}

	public void gibbs() {

		for (int d = 0; d < z.length; d++) {
			for (int n = 0; n < z[d].length; n++) {

				int topic = sampleFullConditional(d, n);
				z[d][n] = topic;

			}
		}

	}

	int sampleFullConditional(int d, int n) {

		int topic = z[d][n];

		int word = documents[d][n];

		updateCount(d, topic, word, -1);

		double[] p = new double[K];

		if (urn_Topic_W1_W2_Value.containsKey(word)) {

			weight[word] = compute_q(word);

		} else {

			for (int k = 0; k < K; k++) {
				weight[word][k] = 1;
			}

		}

		for (int k = 0; k < K; k++) {

			p[k] = weight[word][k] * (nd[d][k] + alpha)
					/ (ndsum[d] + K * alpha) * (nw[documents[d][n]][k] + beta)
					/ (nwsum[k] + V * beta);
		}

		for (int k = 1; k < K; k++) {
			p[k] += p[k - 1];
		}
		double u = Math.random() * p[K - 1];
		for (int t = 0; t < K; t++) {
			if (u < p[t]) {
				topic = t;
				break;
			}
		}

		updateCount(d, topic, word, +1);

		return topic;

	}

	void updateCount(int d, int topic, int word, int flag) {

		nd[d][topic] += flag;
		ndsum[d] += flag;

		if (urn_Topic_W1_W2_Value.containsKey(word)) {
			Map<Integer, Double> urn_W2_value = urn_Topic_W1_W2_Value.get(word);
			for (Map.Entry<Integer, Double> entry : urn_W2_value.entrySet()) {
				int w2 = entry.getKey();
				double count = flag * entry.getValue();
				nw[w2][topic] += count;
				nwsum[topic] += count;
			}
		}

		nw[word][topic] += flag;
		nwsum[topic] += flag;
	}

	/**
	 * Add to the statistics the values of theta and phi for the current state.
	 */
	void updateParams() {
		for (int m = 0; m < documents.length; m++) {
			for (int k = 0; k < K; k++) {
				thetasum[m][k] += (nd[m][k] + alpha) / (ndsum[m] + K * alpha);
			}
		}
		for (int k = 0; k < K; k++) {
			for (int w = 0; w < V; w++) {
				phisum[k][w] += (nw[w][k] + beta) / (nwsum[k] + V * beta);
			}
		}
		numstats++;

	}

	public double[][] estimateTheta() {

		double[][] theta = new double[documents.length][K];

		for (int m = 0; m < documents.length; m++) {
			for (int k = 0; k < K; k++) {
				theta[m][k] = thetasum[m][k] / numstats;
			}
		}

		return theta;
	}

	public double[][] estimatePhi() {

		double[][] phi = new double[K][V];

		for (int k = 0; k < K; k++) {

			for (int w = 0; w < V; w++) {
				phi[k][w] = phisum[k][w] / numstats;
			}
		}

		return phi;
	}

	public double[] compute_q(int word) {

		double[] q_score = new double[K];

		Map<Integer, Double> urn_W2_value = urn_Topic_W1_W2_Value.get(word);

		for (int word_2 : urn_W2_value.keySet()) {

			for (int k = 0; k < K; k++) {
				q_score[k] = nw[word_2][k] * urn_W2_value.get(word_2);
				// q_score[k] = 1;
			}
		}

		return q_score;

	}

}
