package test;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import topic.WeGroupLDA;
import util.Common;
import util.Corpus;
import util.ReadWriteFile;
import wordvector.Word2Vec;

public class RunWeGroupLDA {

	public static void main(String[] args) throws IOException {

		String domain = "Alarm Clock";

		// 读词向量
		String filename = "file//amazon_word.vec";

		Word2Vec w2v = new Word2Vec(filename);

		Map<String, double[]> word_vector = w2v.getWordVector();

		runWeGroupLDA(domain, word_vector);

	}

	/**
	 * 对指定领域执行WE-Group-LDA
	 * 
	 * @param domain
	 * @return
	 * @throws IOException
	 */
	public static double runWeGroupLDA(String domain,
			Map<String, double[]> word_vector) throws IOException {

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

		Map<Integer, Map<Integer, Double>> mustlinks = readMustLinks(vocab);

		// 执行WE-LDA

		WeGroupLDA gpu = new WeGroupLDA(docs, V, mustlinks, vocab, word_vector,
				K);

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

		String filename = "file//" + domain + "_gibbs2.txt";

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
	 * 
	 * @param vocab
	 *            词表
	 * @return
	 * @throws IOException
	 */
	public static Map<Integer, Map<Integer, Map<Integer, Double>>> readKnowledge(
			List<String> vocab) throws IOException {

		File f = new File("file//knowledge_group//Alarm Clock.txt");
		BufferedReader reader = new BufferedReader(new InputStreamReader(
				new FileInputStream(f), "UTF-8"));
		String line = "";

		// 先全部读进来，再分组

		StringBuilder sb = new StringBuilder();

		while ((line = reader.readLine()) != null) {

			sb.append(line + "\n");

		}
		reader.close();

		// 分组

		Map<Integer, Map<Integer, Map<Integer, Double>>> urn_Topic_W1_W2_Value = new HashMap<>();

		String[] knowledge = sb.toString().split("\n\n");

		int group_no = 0;

		for (String group : knowledge) {

			String[] must_links = group.split("\n");

			urn_Topic_W1_W2_Value.put(group_no,
					new HashMap<Integer, Map<Integer, Double>>());

			Map<Integer, Map<Integer, Double>> urn_W1_W2_Value = urn_Topic_W1_W2_Value
					.get(group_no);

			for (String must_link : must_links) {

				String[] temp = must_link.split("\t");

				int word_i = vocab.indexOf(temp[0]);

				int word_j = vocab.indexOf(temp[1]);

				if (!urn_W1_W2_Value.containsKey(word_i)) {
					urn_W1_W2_Value.put(word_i, new HashMap<Integer, Double>());
				}

				Map<Integer, Double> urn_W2_Value = urn_W1_W2_Value.get(word_i);

				double pmi = Double.parseDouble(temp[3]);

				double vector_sim = Double.parseDouble(temp[2]);

				urn_W2_Value.put(word_j, 0.8 * pmi * vector_sim);

				if (!urn_W1_W2_Value.containsKey(word_j)) {
					urn_W1_W2_Value.put(word_j, new HashMap<Integer, Double>());
				}
				urn_W2_Value = urn_W1_W2_Value.get(word_j);
				urn_W2_Value.put(word_j, 0.8 * pmi * vector_sim);

			}

			group_no++;

		}

		return urn_Topic_W1_W2_Value;
	}

	/**
	 * 读取词向量知识，不分组，不聚类
	 * 
	 * @param vocab
	 *            词表
	 * @return
	 * @throws IOException
	 */
	public static Map<Integer, Map<Integer, Double>> readMustLinks(
			List<String> vocab) throws IOException {

		File f = new File("file//knowledge//Alarm Clock.txt");
		BufferedReader reader = new BufferedReader(new InputStreamReader(
				new FileInputStream(f), "UTF-8"));
		String line = "";

		Map<Integer, Map<Integer, Double>> urn_W1_W2_Value = new HashMap<>();

		while ((line = reader.readLine()) != null) {

			String[] temp = line.split("\t");
			if (temp.length < 2)
				continue;

			int word_i = vocab.indexOf(temp[0]);

			int word_j = vocab.indexOf(temp[1]);

			if (!urn_W1_W2_Value.containsKey(word_i)) {
				urn_W1_W2_Value.put(word_i, new HashMap<Integer, Double>());
			}

			Map<Integer, Double> urn_W2_Value = urn_W1_W2_Value.get(word_i);

			double pmi = Double.parseDouble(temp[3]);

			double vector_sim = Double.parseDouble(temp[2]);

			urn_W2_Value.put(word_j, 0.8 * pmi * vector_sim);

			if (!urn_W1_W2_Value.containsKey(word_j)) {
				urn_W1_W2_Value.put(word_j, new HashMap<Integer, Double>());
			}
			urn_W2_Value = urn_W1_W2_Value.get(word_j);
			urn_W2_Value.put(word_j, 0.8 * pmi * vector_sim);

		}

		reader.close();
		return urn_W1_W2_Value;
	}

}
