package test;

import java.util.List;
import java.util.Map;

import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.DefaultDataset;
import net.sf.javaml.core.DenseInstance;
import net.sf.javaml.core.Instance;
import topic.WeClusterLDA;
import util.Common;
import util.Corpus;
import util.ReadWriteFile;
import wordcluster.Kmeans;
import wordvector.Word2Vec;

public class RunWeClusterLDA {

	public static void main(String[] args) throws Exception {

		// 读词向量
		String filename = "file//amazon_word.vec";

		Word2Vec w2v = new Word2Vec(filename);

		Map<String, double[]> word_vector = w2v.getWordVector();

		String domain = "Alarm Clock";

		runWeClusterLDA(domain, word_vector);
	}

	/**
	 * 
	 * 对指定领域执行WeClusterLDA
	 * 
	 * @param domain
	 * @param word_vector
	 *            词向量
	 * @return
	 * @throws Exception
	 */
	public static double runWeClusterLDA(String domain,
			Map<String, double[]> word_vector) throws Exception {

		List<String> vocab = Corpus.getVocab("data//" + domain + "//" + domain
				+ ".vocab");

		int[][] docs = Corpus.getDocuments("data//" + domain + "//" + domain
				+ ".docs");

		// 用Java ML 的 Kmeans

		Dataset data = new DefaultDataset();

		for (String word : vocab) {

			double[] vector = word_vector.get(word);

			Instance instance = new DenseInstance(vector);

			data.add(instance);
		}
		System.out.println(data.size());

		int[] assignment = Kmeans.RunKmeansCosine(data, 100);

		// assignment = Kmedoids.RunKmedoidsCosine(data, 100);

		int K = 15;
		double alpha = 1;
		double beta = 0.1;

		double gamma = 1000;
		int iterations = 2000;

		int top_word_count = 30;

		WeClusterLDA lda = new WeClusterLDA(docs, vocab.size(), assignment, 100);

		lda.markovChain(K, alpha, beta, gamma, iterations);

		double[][] phi = lda.estimateOmega();

		double[][] phi_copy = Common.makeCopy(phi);

		// 将每个主题的前10个词写文件
		double[][] phi_for_write = Common.makeCopy(phi);

		StringBuilder sb = new StringBuilder();

		for (double[] phi_t : phi_for_write) {

			for (int i = 0; i < 10; i++) {

				int max_index = Common.maxIndex(phi_t);

				sb.append(vocab.get(max_index) + "\t");

				phi_t[max_index] = 0;

			}
			sb.append("\n");

		}

		String filename = "file//" + domain + "_we_cluster_lda.txt";

		// 语义一致性
		double average_coherence = Corpus.average_coherence(docs, phi_copy,
				top_word_count);

		System.out.println("average coherence : " + average_coherence);

		sb.append("average coherence\t" + average_coherence);

		ReadWriteFile.writeFile(filename, sb.toString());

		double[][] theta = lda.estimateTheta();

		// perplexity
		double perplexity = Corpus.perplexity(theta, phi, docs);
		System.out.println("perplexity : " + perplexity);

		return average_coherence;

	}

}
