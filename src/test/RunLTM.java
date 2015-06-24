package test;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import topic.WeGpuLDA;
import util.Common;
import util.Corpus;
import util.ReadWriteFile;

public class RunLTM {

	public static void main(String[] args) throws IOException {

		String domain = "Alarm Clock";

		runLTM(domain);

	}

	/**
	 * 对指定文档集执行LTM
	 * 
	 * @param domain
	 * @return
	 * @throws IOException
	 */
	public static double runLTM(String domain) throws IOException {

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

		/*
		 * 从文件中读取LTM产生的知识
		 */

		// float[][][] A = new float[K][V][V];

		Map<Integer, Map<Integer, Map<Integer, Double>>> urn_Topic_W1_W2_Value = new HashMap<>();

		List<List<String>> knowledge = readKnowledgeGroup();

		System.out.println(knowledge.size());

		for (List<String> must_links : knowledge) {

			int topic = knowledge.indexOf(must_links);

			urn_Topic_W1_W2_Value.put(topic,
					new HashMap<Integer, Map<Integer, Double>>());

			Map<Integer, Map<Integer, Double>> urn_W1_W2_Value = urn_Topic_W1_W2_Value
					.get(topic);

			for (String must : must_links) {

				String[] temp = must.split(" ");

				System.out.println(temp[0] + " " + temp[1]);

				int word_i = vocab.indexOf(temp[0]);

				int word_j = vocab.indexOf(temp[1]);

				if (word_i != -1 && word_j != -1) {

					float gpu = (float) (Corpus.PMI(docs, word_i, word_j) * 0.3);

					if (gpu < 0)
						continue;

					if (!urn_W1_W2_Value.containsKey(word_i)) {
						urn_W1_W2_Value.put(word_i,
								new HashMap<Integer, Double>());
					}

					Map<Integer, Double> urn_W2_Value = urn_W1_W2_Value
							.get(word_i);
					urn_W2_Value.put(word_j, (double) gpu);

					if (!urn_W1_W2_Value.containsKey(word_j)) {
						urn_W1_W2_Value.put(word_j,
								new HashMap<Integer, Double>());
					}
					urn_W2_Value = urn_W1_W2_Value.get(word_j);
					urn_W2_Value.put(word_j, (double) gpu);

					// A[topic][word_j][word_i] = A[topic][word_i][word_j];

				}

			}

		}
		System.out.println(urn_Topic_W1_W2_Value);
		// 执行LTM

		WeGpuLDA gpu = new WeGpuLDA(docs, V, urn_Topic_W1_W2_Value);

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

		String filename = "output//GPU-LDA//" + domain + ".txt";

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
	 * 读取LTM知识，分组
	 * 
	 * @return
	 * @throws IOException
	 */
	public static List<List<String>> readKnowledgeGroup() throws IOException {

		// LTM must link, 分组

		File f = new File("file//Alarm Clock.knowl");
		BufferedReader reader = new BufferedReader(new InputStreamReader(
				new FileInputStream(f), "UTF-8"));
		String line = "";

		StringBuilder sb = new StringBuilder();

		while ((line = reader.readLine()) != null) {

			sb.append(line + "\n");
		}
		reader.close();

		List<List<String>> knowledge = new ArrayList<>();

		String[] temp = sb.toString().split("\n\n");

		for (String topic : temp) {

			String[] must_link = topic.split("\n");

			List<String> group = Arrays.asList(must_link);

			knowledge.add(group);

		}

		return knowledge;
	}

}
