package topic;

import util.Corpus;

/**
 * GPU-LDA, In EMNLP'11
 *
 * @author Liang Yao
 * @email yaoliangzju@gmail.com
 *
 */
public class GPU {

	int[][] documents;

	int V;

	int K;

	double alpha;

	double beta;

	int[][] z;

	double[][] nw;

	int[][] nd;

	float[][] A; // 词相关度矩阵

	double[] nwsum;

	int[] ndsum;

	int iterations;

	public GPU(int[][] documents, int V) {

		this.documents = documents;
		this.V = V;
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

				nw[documents[d][n]][topic]++;

				nwsum[topic]++;

				// 自己+1,别人加权
				for (int v = 0; v < V; v++) {

					if (v != documents[d][n]) {

						nw[v][topic] += A[v][documents[d][n]];

						nwsum[topic] += A[v][documents[d][n]];

					}

				}

				nd[d][topic]++;

			}
			ndsum[d] = Nd;
		}

	}

	public void markovChain(int K, double alpha, double beta, int iterations) {

		this.K = K;
		this.alpha = alpha;
		this.beta = beta;
		this.iterations = iterations;

		initialState();

		for (int i = 0; i < this.iterations; i++) {
			// System.out.println("iteration : " + i);
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
		nw[documents[d][n]][topic]--;
		nd[d][topic]--;
		nwsum[topic]--;
		ndsum[d]--;

		for (int v = 0; v < V; v++) {

			if (v != documents[d][n]) {

				nw[v][topic] -= A[v][documents[d][n]];

				nwsum[topic] -= A[v][documents[d][n]];

			}
		}

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
		nw[documents[d][n]][topic]++;
		nd[d][topic]++;
		nwsum[topic]++;
		ndsum[d]++;

		for (int v = 0; v < V; v++) {

			if (v != documents[d][n]) {

				nw[v][topic] += A[v][documents[d][n]];

				nwsum[topic] += A[v][documents[d][n]];

			}
		}

		return topic;

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

	/**
	 * 计算词相关度矩阵A
	 */
	public void setSchema() {

		A = new float[V][V];

		for (int i = 0; i < V; i++) {

			double idf = Corpus.IDF(documents, i);

			int df = Corpus.DocumentFrequency(documents, i);

			for (int j = 0; j < V; j++) {

				if (i == j) {
					A[i][j] = (float) (idf * df);
				} else {
					A[i][j] = (float) (idf * Corpus.DocumentFrequency(
							documents, i, j));
				}

			}

		}

		// 正则化到和为1

		for (int i = 0; i < V; i++) {

			double sum = 0;

			for (int j = 0; j < V; j++) {
				sum += A[i][j];
			}

			for (int j = 0; j < V; j++) {

				A[i][j] = (float) (A[i][j] / sum);

			}

		}

	}
}
