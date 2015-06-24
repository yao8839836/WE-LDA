package test;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import knowledge.KnowledgeExtract;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.DefaultDataset;
import net.sf.javaml.core.DenseInstance;
import net.sf.javaml.core.Instance;
import topic.WeLDA;
import util.Common;
import util.Corpus;
import util.ReadWriteFile;
import wordcluster.Kmeans;
import wordvector.Word2Vec;
import wordvector.WordEntry;

public class MultiDomainWeLDA {

	public static void main(String[] args) throws Exception {

		/*
		 * 读所有的领域名
		 */

		File[] files = new File("data//Electronics//").listFiles();

		List<String> domain_list = new ArrayList<String>();

		for (File f : files) {
			String file_path = f.toString();
			String domain = file_path.substring(
					file_path.lastIndexOf("\\") + 1, file_path.length());

			domain_list.add(domain);
		}

		// 读词向量

		String filename = "file//amazon_word.vec";

		Word2Vec w2v = new Word2Vec(filename);

		double coherence = 0;

		StringBuilder sb = new StringBuilder();

		for (String domain : domain_list) {

			double domain_coherence = runWeLDA(domain, w2v);
			coherence += domain_coherence;
			sb.append(domain + "\t" + domain_coherence + "\n");
			System.out.println(domain + "\t" + domain_coherence + "\n");

		}

		sb.append("average : " + coherence / domain_list.size() + "\n");

		filename = "file//we_lda_coherence_top10.txt";

		ReadWriteFile.writeFile(filename, sb.toString());

	}

	/**
	 * 对指定领域执行WE-LDA
	 * 
	 * @param domain
	 *            领域名
	 * @param w2v
	 *            词向量类
	 * @return
	 * @throws Exception
	 */
	public static double runWeLDA(String domain, Word2Vec w2v) throws Exception {

		// 词表，语料库
		List<String> vocab = Corpus.getVocab("data//Electronics//" + domain
				+ "//" + domain + ".vocab");

		int[][] docs = Corpus.getDocuments("data//Electronics//" + domain
				+ "//" + domain + ".docs");

		/*
		 * 参数设置
		 */

		int K = 15;
		double alpha = 1;
		double beta = 0.1;
		int iterations = 2000;

		int top_word_count = 10;

		int V = vocab.size();

		int top_seed_words_num = 30;

		/*
		 * 读种子词集合
		 */

		String filename = "data//Electronics//" + domain + "//" + domain
				+ ".twords";

		Set<String> top_words = KnowledgeExtract.getTopWords(filename,
				top_seed_words_num);

		// 生成nust-link知识库,写文件

		String content = generateKnowledgeBase(top_words, vocab, docs, w2v);

		filename = "file//knowledge//" + domain + ".txt";
		ReadWriteFile.writeFile(filename, content);

		runKmeans(domain, w2v, K);

		// 读must-link知识库
		Map<Integer, Map<Integer, Map<Integer, Double>>> urn_Topic_W1_W2_Value = readKnowledge(
				K, vocab, domain);

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

		filename = "output//WE-LDA//" + domain + ".txt";

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
	 * 
	 * 根据种子词(LDA top words)生成must-link知识库
	 * 
	 * @param top_words
	 * @param vocab
	 *            词表
	 * @param docs
	 *            语料库
	 * @param w2v
	 *            词向量类
	 * @return 知识库字符串
	 */

	public static String generateKnowledgeBase(Set<String> top_words,
			List<String> vocab, int[][] docs, Word2Vec w2v) {

		StringBuilder sb = new StringBuilder();

		Set<Set<String>> visited = new HashSet<>();

		for (String word : top_words) {

			// 相似的词
			Set<WordEntry> word_set = w2v.distance(word);

			for (WordEntry sim_word : word_set) {

				if (vocab.contains(sim_word.name)) {

					int word_i = vocab.indexOf(word);
					int word_j = vocab.indexOf(sim_word.name);

					double pmi = Corpus.PMI(docs, word_i, word_j);

					// 词的点互信息
					if (pmi > 0) {

						Set<String> must = new HashSet<>();

						must.add(word);
						must.add(sim_word.name);

						if (visited.contains(must))
							continue;

						sb.append(word + "\t" + sim_word.name + "\t"
								+ sim_word.score + "\t" + pmi + "\n");
						visited.add(must);

					}

				}

			}

		}

		return sb.toString();
	}

	/**
	 * 读取词向量知识，分组
	 * 
	 * @param K
	 *            主题数，簇数
	 * @param vocab
	 *            词表
	 * @param domain
	 * @return
	 * @throws IOException
	 */

	public static Map<Integer, Map<Integer, Map<Integer, Double>>> readKnowledge(
			int K, List<String> vocab, String domain) throws IOException {

		Map<Integer, Map<Integer, Map<Integer, Double>>> urn_Topic_W1_W2_Value = new HashMap<>();

		for (int k = 0; k < K; k++) {

			// WE must link, 分组

			File f = new File("file//knowledge//" + domain + ".txt");
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

					// miu in paper

					double miu = 0.8;

					urn_W2_Value.put(word_j, miu * pmi * vector_sim);

					if (!urn_W1_W2_Value.containsKey(word_j)) {
						urn_W1_W2_Value.put(word_j,
								new HashMap<Integer, Double>());
					}
					urn_W2_Value = urn_W1_W2_Value.get(word_j);
					urn_W2_Value.put(word_j, miu * pmi * vector_sim);

				}

			}
			reader.close();

		}

		return urn_Topic_W1_W2_Value;
	}

	/**
	 * 对must-link聚类，用两个词的向量和作为特征，结果写回文件
	 * 
	 * @param domain
	 *            领域名
	 * @param w2v
	 *            词向量
	 * 
	 * @param K
	 *            主题数，must-link簇数
	 * @throws Exception
	 */

	public static void runKmeans(String domain, Word2Vec w2v, int K)
			throws Exception {

		File f = new File("file//knowledge//" + domain + ".txt");
		BufferedReader reader = new BufferedReader(new InputStreamReader(
				new FileInputStream(f), "UTF-8"));
		String line = "";

		Map<String, double[]> word_vector = w2v.getWordVector();

		// 用Java ML 的 Kmeans

		Dataset data = new DefaultDataset();

		while ((line = reader.readLine()) != null) {

			String[] temp = line.split("\t");

			String word_1 = temp[0];

			String word_2 = temp[1];

			double[] vector_1 = word_vector.get(word_1);
			double[] vector_2 = word_vector.get(word_2);

			double[] vector = Common.add(vector_1, vector_2);

			Instance instance = new DenseInstance(vector);

			data.add(instance);

		}

		reader.close();

		int[] assignment = Kmeans.RunKmeansCosine(data, K);

		// int[] assignment = Kmedoids.RunKmedoidsCosine(data, K);

		// 将簇标号写到每一行末尾

		reader = new BufferedReader(new InputStreamReader(
				new FileInputStream(f), "UTF-8"));

		StringBuilder sb = new StringBuilder();

		int line_no = 0;

		while ((line = reader.readLine()) != null) {

			sb.append(line + "\t" + assignment[line_no] + "\n");
			line_no++;

		}
		reader.close();

		ReadWriteFile.writeFile("file//knowledge//" + domain + ".txt",
				sb.toString());

	}

}
