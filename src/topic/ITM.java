package topic;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * @author Liang Yao
 * 
 * @comment Interactive Topic Modeling (ACL'11)
 *
 */
public class ITM {

	int[][] documents;

	int V;

	int K;

	double alpha;

	double beta;

	double eta;

	int[][] z;

	/*
	 * 三个后验概率累计
	 */

	double[][] thetasum;

	double[][] phisum;

	double[][] phi_group_sum;

	double[][][] etasum;

	/*
	 * 计数器
	 */

	double[][][] ntgw; // 主题-组-词计数器

	double[][] ntgsum; // 主题-组-所有词计数器

	double[][] ntg;

	double[][] ntw;

	double[] ntsum;

	int[][] nd;

	int[] ndsum;

	// 迭代相关

	int BURN_IN = 200;

	int SAMPLE_LAG = 20;

	int numstats;

	int iterations;

	/*
	 * 知识
	 */

	int num_group; // 共有多少组知识

	Map<Integer, Map<Integer, Map<Integer, Double>>> knowledge;

	Set<Integer> constrain_word; // 所有有限制的词

	Map<Integer, Integer> word_group; // 词属于的组

	Map<Integer, Integer> num_words_in_group; // 这一组限制中词的数量

	/*
	 * 构造方法
	 */

	public ITM(int[][] docs, int V,
			Map<Integer, Map<Integer, Map<Integer, Double>>> knowledge) {

		this.documents = docs;
		this.V = V;

		this.knowledge = knowledge;

		num_group = knowledge.keySet().size();

		word_group = new HashMap<>();

		num_words_in_group = new HashMap<>();

		// 获得所有知识涉及的词
		constrain_word = new HashSet<>();

		for (int key : knowledge.keySet()) {

			Set<Integer> words_in_this_group = new HashSet<>();

			Map<Integer, Map<Integer, Double>> urn_W1_W2_Value = knowledge
					.get(key);

			for (int word_1 : urn_W1_W2_Value.keySet()) {

				constrain_word.add(word_1);

				words_in_this_group.add(word_1);

				word_group.put(word_1, key);

				Map<Integer, Double> urn_W2_value = urn_W1_W2_Value.get(word_1);

				for (int word_2 : urn_W2_value.keySet()) {

					constrain_word.add(word_2);
					word_group.put(word_2, key);

					words_in_this_group.add(word_2);
				}

			}

			num_words_in_group.put(key, words_in_this_group.size());

		}

	}

	public void initialState() {

		int D = documents.length;

		nd = new int[D][K];
		ndsum = new int[D];

		ntw = new double[V][K];

		ntg = new double[num_group][K];
		ntsum = new double[K];

		ntgw = new double[K][num_group][V];

		z = new int[D][];

		for (int d = 0; d < D; d++) {

			int Nd = documents[d].length;

			z[d] = new int[Nd];

			for (int n = 0; n < Nd; n++) {

				int topic = (int) (Math.random() * K);

				z[d][n] = topic;

				// 初始化为词向量聚类之后结果
				updateCount(d, topic, documents[d][n], +1);

			}
		}

	}

	public void markovChain(int K, double alpha, double beta, double eta,
			int iterations) {

		this.K = K;
		this.alpha = alpha;
		this.beta = beta;
		this.eta = eta;
		this.iterations = iterations;

		if (SAMPLE_LAG > 0) {

			thetasum = new double[documents.length][K];
			phisum = new double[K][V];

			phi_group_sum = new double[K][num_group];

			etasum = new double[K][num_group][V];
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

		int group = 0;

		if (word_group.containsKey(word))

			group = word_group.get(word);

		int group_size = num_words_in_group.get(group);

		updateCount(d, topic, documents[d][n], -1);

		double[] p = new double[K];

		if (constrain_word.contains(word)) {

			for (int k = 0; k < K; k++) {

				p[k] = (nd[d][k] + alpha) / (ndsum[d] + K * alpha)
						* (ntg[group][k] + group_size * beta)
						/ (ntsum[k] + V * beta) * (ntgw[k][group][word] + eta)
						/ (ntg[group][k] + group_size * eta);
			}

		} else {

			for (int k = 0; k < K; k++) {

				p[k] = (nd[d][k] + alpha) / (ndsum[d] + K * alpha)
						* (ntw[word][k] + beta) / (ntsum[k] + V * beta);
			}

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
			for (int g = 0; g < num_group; g++) {
				int group_size = num_words_in_group.get(g);
				phi_group_sum[k][g] += (ntg[g][k] + group_size * beta)
						/ (ntsum[k] + V * beta);
			}
		}

		for (int k = 0; k < K; k++) {
			for (int w = 0; w < V; w++) {
				phisum[k][w] += (ntw[w][k] + beta) / (ntsum[k] + V * beta);
			}
		}

		for (int k = 0; k < K; k++) {

			for (int g = 0; g < num_group; g++) {
				// 组的大小

				int group_size = num_words_in_group.get(g);
				for (int v = 0; v < V; v++) {
					etasum[k][g][v] += (ntgw[k][g][v] + eta)
							/ (ntg[g][k] + group_size * eta);
				}

			}
		}

		numstats++;
	}

	public void updateCount(int d, int topic, int word, int flag) {

		nd[d][topic] += flag;
		ndsum[d] += flag;

		if (constrain_word.contains(word)) {

			int group = word_group.get(word);
			ntg[group][topic] += flag;
			ntgw[topic][group][word] += flag;

		} else {
			ntw[word][topic] += flag;
		}

		ntsum[topic] += flag;

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
				if (constrain_word.contains(w)) {

					for (int g = 0; g < num_group; g++) {
						phi[k][w] = phi_group_sum[k][g] * etasum[k][g][w]
								/ numstats;
					}

				} else {
					phi[k][w] = phisum[k][w] / numstats;
				}

			}
		}

		return phi;
	}

}
