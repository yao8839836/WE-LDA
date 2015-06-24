package test;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import topic.WeLDA;
import util.Common;
import util.Corpus;
import util.ReadWriteFile;

public class RunWeLDA {

	public static void main(String[] args) throws IOException {

		String domain = "Alarm Clock";

		runWeLDA(domain);
	}

	/**
	 * 对指定领域执行WE-LDA
	 * 
	 * @param domain
	 * @return
	 * @throws IOException
	 */
	public static double runWeLDA(String domain) throws IOException {

		List<String> vocab = Corpus.getVocab("data//" + domain + "//" + domain
				+ ".vocab");

		int[][] docs = Corpus.getDocuments("data//" + domain + "//" + domain
				+ ".docs");

		int K = 15;
		double alpha = 1;
		double beta = 0.1;
		int iterations = 2000;

		int top_word_count = 10;

		int V = vocab.size();

		Map<Integer, Map<Integer, Map<Integer, Double>>> urn_Topic_W1_W2_Value = readKnowledge(
				K, vocab);

		// 执行WE-LDA

		WeLDA gpu = new WeLDA(docs, V, urn_Topic_W1_W2_Value);

		gpu.markovChain(K, alpha, beta, iterations);

		double[][] phi = gpu.estimatePhi();

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

		String filename = "file//" + domain + ".txt";

		// 语义一致性
		double average_coherence = Corpus.average_coherence(docs, phi_copy,
				top_word_count);

		System.out.println("average coherence : " + average_coherence);

		sb.append("average coherence\t" + average_coherence);

		ReadWriteFile.writeFile(filename, sb.toString());

		double[][] theta = gpu.estimateTheta();

		// perplexity
		double perplexity = Corpus.perplexity(theta, phi, docs);
		System.out.println("perplexity : " + perplexity);

		return average_coherence;

	}

	/**
	 * 读取词向量知识，分组
	 * 
	 * @param K
	 *            主题数，簇数
	 * @param vocab
	 *            词表
	 * @return
	 * @throws IOException
	 */
	public static Map<Integer, Map<Integer, Map<Integer, Double>>> readKnowledge(
			int K, List<String> vocab) throws IOException {

		Map<Integer, Map<Integer, Map<Integer, Double>>> urn_Topic_W1_W2_Value = new HashMap<>();

		for (int k = 0; k < K; k++) {

			// WE must link, 分组

			File f = new File("file//knowledge//Alarm Clock.txt");
			BufferedReader reader = new BufferedReader(new InputStreamReader(
					new FileInputStream(f), "UTF-8"));
			String line = "";

			urn_Topic_W1_W2_Value.put(k,
					new HashMap<Integer, Map<Integer, Double>>());

			Map<Integer, Map<Integer, Double>> urn_W1_W2_Value = urn_Topic_W1_W2_Value
					.get(k);

			while ((line = reader.readLine()) != null) {

				String[] temp = line.split("\t");

				int len = temp.length;

				if (k == Integer.parseInt(temp[len - 1])) {

					int word_i = vocab.indexOf(temp[0]);

					int word_j = vocab.indexOf(temp[1]);

					if (!urn_W1_W2_Value.containsKey(word_i)) {
						urn_W1_W2_Value.put(word_i,
								new HashMap<Integer, Double>());
					}

					Map<Integer, Double> urn_W2_Value = urn_W1_W2_Value
							.get(word_i);

					double pmi = Double.parseDouble(temp[3]);

					double vector_sim = Double.parseDouble(temp[2]);

					urn_W2_Value.put(word_j, 0.8 * pmi * vector_sim);

					if (!urn_W1_W2_Value.containsKey(word_j)) {
						urn_W1_W2_Value.put(word_j,
								new HashMap<Integer, Double>());
					}
					urn_W2_Value = urn_W1_W2_Value.get(word_j);
					urn_W2_Value.put(word_j, 0.8 * pmi * vector_sim);

				}

			}
			reader.close();

		}

		return urn_Topic_W1_W2_Value;
	}
}
