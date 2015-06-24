package topic;

import java.util.Map;

/**
 * @author Liang Yao
 * 
 * @comment 利用词向量构建相关度矩阵
 *
 */
public class WeGpuLDA {

	int[][] documents;

	int V;

	int K;

	double alpha;

	double beta;

	int[][] z;

	double[][] nw;

	int[][] nd;

	float[][] A; // 词相关度矩阵

	// 每个主题下的must-link及其相关度

	Map<Integer, Map<Integer, Map<Integer, Double>>> urn_Topic_W1_W2_Value;

	double[] nwsum;

	int[] ndsum;

	int iterations;

	double miu;

	public WeGpuLDA(int[][] documents, int V, float[][] A) {

		this.documents = documents;
		this.V = V;
		this.A = A;

		this.miu = 0.3;
	}

	public WeGpuLDA(
			int[][] documents,
			int V,
			Map<Integer, Map<Integer, Map<Integer, Double>>> urn_Topic_W1_W2_Value) {

		this.documents = documents;
		this.V = V;
		this.urn_Topic_W1_W2_Value = urn_Topic_W1_W2_Value;

		this.miu = 0.3;
	}

	public void initialState() {

		int D = documents.length;
		nw = new double[V][K];
		nd = new int[D][K];
		nwsum = new double[K];
		ndsum = new int[D];

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

		initialState();

		for (int i = 0; i < this.iterations; i++) {

			gibbs();
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

		updateCount(d, topic, documents[d][n], -1);

		double[] p = new double[K];

		for (int k = 0; k < K; k++) {

			p[k] = (nd[d][k] + alpha) / (ndsum[d] + K * alpha)
					* (nw[documents[d][n]][k] + beta) / (nwsum[k] + V * beta);
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

		updateCount(d, topic, documents[d][n], +1);

		return topic;

	}

	void updateCount(int d, int topic, int word, int flag) {

		nd[d][topic] += flag;
		ndsum[d] += flag;

		// 根据罐子更新相关词的权重
		if (urn_Topic_W1_W2_Value.containsKey(topic)) {

			Map<Integer, Map<Integer, Double>> urn_W1_W2_Value = urn_Topic_W1_W2_Value
					.get(topic);
			if (urn_W1_W2_Value.containsKey(word)) {
				Map<Integer, Double> urn_W2_value = urn_W1_W2_Value.get(word);
				for (Map.Entry<Integer, Double> entry : urn_W2_value.entrySet()) {
					int w2 = entry.getKey();
					double count = flag * entry.getValue();
					nw[w2][topic] += count;
					nwsum[topic] += count;
				}
			}
		}

		nw[word][topic] += flag;
		nwsum[topic] += flag;
	}

	public double[][] estimateTheta() {
		double[][] theta = new double[documents.length][K];
		for (int d = 0; d < documents.length; d++) {
			for (int k = 0; k < K; k++) {
				theta[d][k] = (nd[d][k] + alpha) / (ndsum[d] + K * alpha);
			}
		}
		return theta;
	}

	public double[][] estimatePhi() {
		double[][] phi = new double[K][V];
		for (int k = 0; k < K; k++) {
			for (int w = 0; w < V; w++) {
				phi[k][w] = (nw[w][k] + beta) / (nwsum[k] + V * beta);
			}
		}
		return phi;
	}

}
