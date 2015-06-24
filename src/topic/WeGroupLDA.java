package topic;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import util.Common;

public class WeGroupLDA {

	int[][] documents;

	int V;

	int K;

	double alpha;

	double beta;

	int[][] z;

	double[][] nw;

	int[][] nd;

	// 每个词的must-link及其相关度

	Map<Integer, Map<Integer, Double>> urn_Topic_W1_W2_Value;

	double[][] topic_vector;

	Map<String, double[]> word_vector;

	List<String> vocab;

	double[] nwsum;

	int[] ndsum;

	int iterations;

	double[][] thetasum;

	double[][] phisum;

	int numstats;

	int BURN_IN = 200;

	int SAMPLE_LAG = 20;

	int word_vector_size = 100;

	public WeGroupLDA(int[][] documents, int V,
			Map<Integer, Map<Integer, Double>> urn_Topic_W1_W2_Value,
			List<String> vocab, Map<String, double[]> word_vector, int K) {

		this.documents = documents;
		this.V = V;
		this.urn_Topic_W1_W2_Value = urn_Topic_W1_W2_Value;
		this.word_vector = word_vector;
		this.vocab = vocab;
		this.K = K;

		topic_vector = new double[K][word_vector_size];

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

		updateCount(d, topic, documents[d][n], -1);

		double[] p = new double[K];

		for (int k = 0; k < K; k++) {

			p[k] = (nd[d][k] + alpha) / (ndsum[d] + K * alpha)
					* (nw[documents[d][n]][k] + beta) / (nwsum[k] + V * beta);
		}
		topic = sampleMultinomialDistribution(p);

		updateCount(d, topic, documents[d][n], +1);

		return topic;

	}

	void updateCount(int d, int topic, int word, int flag) {

		nd[d][topic] += flag;
		ndsum[d] += flag;

		double[] word_1_vector = word_vector.get(vocab.get(word));

		for (int dim = 0; dim < word_vector_size; dim++) {

			topic_vector[topic][dim] += flag * word_1_vector[dim];
		}
		// 根据罐子更新相关词的权重

		if (urn_Topic_W1_W2_Value.containsKey(word)) {
			Map<Integer, Double> urn_W2_value = urn_Topic_W1_W2_Value.get(word);

			double[] p = new double[urn_W2_value.keySet().size()];

			List<Integer> must_words = new ArrayList<>(urn_W2_value.keySet());

			double[] current_topic_vector = topic_vector[topic];

			for (int word_2 : must_words) {

				double[] word_2_vector = word_vector.get(vocab.get(word_2));

				double[] vector_sum = Common.add(word_1_vector, word_2_vector);

				double cosine_similarity = 1 - Common.cosine_distance(
						vector_sum, current_topic_vector);

				p[must_words.indexOf(word_2)] = cosine_similarity;

				// p[must_words.indexOf(word_2)] = (nw[word_2][topic] + beta)
				// / (nwsum[topic] + V * beta);

			}

			int selected_word_index = sampleMultinomialDistribution(p);

			int selected_word = must_words.get(selected_word_index);

			double relatedness = urn_W2_value.get(selected_word);

			double count = flag * relatedness;

			nw[selected_word][topic] += count;
			nwsum[topic] += count;

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

	int sampleMultinomialDistribution(double[] p) {

		int state = 0;

		for (int k = 1; k < p.length; k++) {
			p[k] += p[k - 1];
		}
		double u = Math.random() * p[p.length - 1];
		for (int t = 0; t < p.length; t++) {
			if (u < p[t]) {
				state = t;
				break;
			}
		}

		return state;
	}

}
