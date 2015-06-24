package topic;

/**
 * @author Liang Yao
 * 
 * @comment 目前是对所有词的簇，效果不行
 *
 */
public class WeClusterLDA {

	int[][] documents;

	int V;

	int K;

	double alpha;

	double beta;

	double gamma;

	int[][] z;

	/*
	 * 三个后验概率累计
	 */

	double[][] thetasum;

	double[][] phisum;

	double[][][] etasum;

	/*
	 * 计数器
	 */

	double[][][] ntcw; // 主题-簇-词计数器

	double[][] ntcsum; // 主题-簇-所有词计数器

	double[][] nc;

	double[] ncsum;

	int[][] nd;

	int[] ndsum;

	int[] word_cluster; // 词所属于的簇标号

	int cluster; // 词向量簇数

	int numstats;

	int BURN_IN = 200;

	int SAMPLE_LAG = 20;

	int iterations;

	public WeClusterLDA(int[][] docs, int V, int[] assignment, int cluster) {

		this.documents = docs;
		this.V = V;

		this.word_cluster = assignment;
		this.cluster = cluster;

	}

	public void initialState() {

		int D = documents.length;
		nc = new double[cluster][K];
		nd = new int[D][K];
		ncsum = new double[K];
		ndsum = new int[D];

		ntcw = new double[K][cluster][V];

		ntcsum = new double[K][cluster];

		z = new int[D][];

		for (int d = 0; d < D; d++) {

			int Nd = documents[d].length;

			z[d] = new int[Nd];

			for (int n = 0; n < Nd; n++) {

				int topic = (int) (Math.random() * K);

				z[d][n] = topic;

				// 初始化为词向量聚类之后结果
				updateCount(d, topic, word_cluster[documents[d][n]],
						documents[d][n], +1);

			}
		}

	}

	public void markovChain(int K, double alpha, double beta, double gamma,
			int iterations) {

		this.K = K;
		this.alpha = alpha;
		this.beta = beta;
		this.gamma = gamma;
		this.iterations = iterations;

		if (SAMPLE_LAG > 0) {

			thetasum = new double[documents.length][K];
			phisum = new double[K][cluster];

			etasum = new double[K][cluster][V];
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

		double[] p = new double[K * cluster];

		int word = documents[d][n];

		updateCount(d, topic, word_cluster[word], documents[d][n], -1);

		for (int k = 0; k < K; k++) {

			p[k] = (nd[d][k] + alpha) / (ndsum[d] + K * alpha)
					* (nc[word_cluster[word]][k] + beta)
					/ (ncsum[k] + K * beta)
					* (ntcw[topic][word_cluster[word]][word] + gamma)
					/ (ntcsum[topic][word_cluster[word]] + V * gamma);
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

		int new_cluster = topic % cluster;

		int new_topic = topic / cluster;

		word_cluster[word] = new_cluster;

		updateCount(d, new_topic, new_cluster, documents[d][n], +1);

		return topic;

	}

	public void updateCount(int d, int topic, int cluster, int word, int flag) {

		nd[d][topic] += flag;
		ndsum[d] += flag;

		nc[cluster][topic] += flag;
		ncsum[topic] += flag;

		ntcw[topic][cluster][word] += flag;
		ntcsum[topic][cluster] += flag;

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
			for (int c = 0; c < cluster; c++) {
				phisum[k][c] += (nc[c][k] + beta) / (ncsum[k] + cluster * beta);
			}
		}

		for (int k = 0; k < K; k++) {

			for (int c = 0; c < cluster; c++) {
				// 簇的大小
				for (int v = 0; v < V; v++) {
					etasum[k][c][v] += (ntcw[k][c][v] + gamma)
							/ (ntcsum[k][c] + V * gamma);
				}

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
			for (int c = 0; c < cluster; c++) {
				phi[k][c] = phisum[k][c] / numstats;
			}
		}

		return phi;
	}

	public double[][][] eatimateEta() {

		double[][][] eta = new double[K][cluster][V];

		for (int k = 0; k < K; k++) {

			for (int c = 0; c < cluster; c++) {
				// 簇的大小
				for (int v = 0; v < V; v++) {
					eta[k][c][v] = etasum[k][c][v] / numstats;
				}

			}
		}

		return eta;
	}

	/**
	 * 这才是主题-词分布 omega[t][w] = sum_{c} phi[t][c] * eta[t][c][w_i]
	 * 
	 * @return
	 */
	public double[][] estimateOmega() {

		double[][] phi = estimatePhi();

		double[][][] eta = eatimateEta();

		double[][] omega = new double[K][V];

		for (int k = 0; k < K; k++) {

			for (int c = 0; c < cluster; c++) {
				// 簇的大小
				for (int v = 0; v < V; v++) {

					omega[k][v] += phi[k][c] * eta[k][c][v];
				}

			}
		}

		return omega;

	}
}
