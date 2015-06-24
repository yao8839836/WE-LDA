package test;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import topic.WeZlabelLDA;
import util.Common;
import util.Corpus;
import util.ReadWriteFile;

public class MultiDomainWeZLDA {

	public static void main(String[] args) throws IOException {

		/*
		 * 读所有的领域名
		 */

		File[] files = new File("data//").listFiles();

		List<String> domain_list = new ArrayList<String>();

		for (File f : files) {
			String file_path = f.toString();
			String domain = file_path.substring(file_path.indexOf("\\") + 1,
					file_path.length());

			domain_list.add(domain);
		}

		double coherence = 0;

		StringBuilder sb = new StringBuilder();

		for (String domain : domain_list) {

			double domain_coherence = runWeZlabelLDA(domain);
			coherence += domain_coherence;
			sb.append(domain + "\t" + domain_coherence + "\n");
			System.out.println(domain + "\t" + domain_coherence + "\n");

		}

		sb.append("average : " + coherence / domain_list.size() + "\n");

		String filename = "file//we_zlabel_lda_coherence_top30.txt";

		ReadWriteFile.writeFile(filename, sb.toString());

	}

	/**
	 * 对指定领域执行WE-Zlabel-GPU-LDA
	 * 
	 * @param domain
	 * @return
	 * @throws IOException
	 */
	public static double runWeZlabelLDA(String domain) throws IOException {

		List<String> vocab = Corpus.getVocab("data//" + domain + "//" + domain
				+ ".vocab");

		int[][] docs = Corpus.getDocuments("data//" + domain + "//" + domain
				+ ".docs");

		int K = 15;
		double alpha = 1;
		double beta = 0.1;
		int iterations = 2000;

		int top_word_count = 30;

		int V = vocab.size();

		Map<Integer, Map<Integer, Double>> mustlinks = readMustLinks(vocab,
				domain);

		WeZlabelLDA zlabel = new WeZlabelLDA(docs, V, mustlinks);

		zlabel.markovChain(K, alpha, beta, iterations);

		double[][] phi = zlabel.estimatePhi();

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

		String filename = "output//WE-zlabel-GPU-LDA//" + domain + ".txt";

		// 语义一致性
		double average_coherence = Corpus.average_coherence(docs, phi_copy,
				top_word_count);

		System.out.println("average coherence : " + average_coherence);

		sb.append("average coherence\t" + average_coherence);

		ReadWriteFile.writeFile(filename, sb.toString());

		double[][] theta = zlabel.estimateTheta();

		// perplexity
		double perplexity = Corpus.perplexity(theta, phi, docs);
		System.out.println("perplexity : " + perplexity);

		return average_coherence;

	}

	/**
	 * 读取指定领域词向量知识，不分组，不聚类
	 * 
	 * @param vocab
	 * @param domain
	 * @return
	 * @throws IOException
	 */
	public static Map<Integer, Map<Integer, Double>> readMustLinks(
			List<String> vocab, String domain) throws IOException {

		File f = new File("file//knowledge//" + domain + ".txt");
		BufferedReader reader = new BufferedReader(new InputStreamReader(
				new FileInputStream(f), "UTF-8"));
		String line = "";

		Map<Integer, Map<Integer, Double>> urn_W1_W2_Value = new HashMap<>();

		while ((line = reader.readLine()) != null) {

			String[] temp = line.split("\t");

			int word_i = vocab.indexOf(temp[0]);

			int word_j = vocab.indexOf(temp[1]);

			if (!urn_W1_W2_Value.containsKey(word_i)) {
				urn_W1_W2_Value.put(word_i, new HashMap<Integer, Double>());
			}

			Map<Integer, Double> urn_W2_Value = urn_W1_W2_Value.get(word_i);

			double pmi = Double.parseDouble(temp[3]);

			double vector_sim = Double.parseDouble(temp[2]);

			double miu = 0.8;

			urn_W2_Value.put(word_j, miu * pmi * vector_sim);

			if (!urn_W1_W2_Value.containsKey(word_j)) {
				urn_W1_W2_Value.put(word_j, new HashMap<Integer, Double>());
			}
			urn_W2_Value = urn_W1_W2_Value.get(word_j);
			urn_W2_Value.put(word_j, miu * pmi * vector_sim);

		}

		reader.close();
		return urn_W1_W2_Value;
	}
}
