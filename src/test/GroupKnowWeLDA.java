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

import util.Common;
import util.Corpus;
import util.ReadWriteFile;
import wordvector.Word2Vec;

public class GroupKnowWeLDA {

	public static void main(String[] args) throws IOException {

		/*
		 * 读领域名
		 */

		File[] files = new File("data//").listFiles();

		List<String> domain_list = new ArrayList<String>();

		for (File f : files) {
			String file_path = f.toString();
			String domain = file_path.substring(file_path.indexOf("\\") + 1,
					file_path.length());

			domain_list.add(domain);
		}

		// 读词向量
		String filename = "file//amazon_word.vec";

		Word2Vec w2v = new Word2Vec(filename);

		Map<String, double[]> word_vector = w2v.getWordVector();

		// 每个领域写文件

		for (String domain : domain_list) {

			String knowledge = generateGroupKnowledge(domain, word_vector);

			filename = "file//knowledge_group//" + domain + ".txt";

			ReadWriteFile.writeFile(filename, knowledge);

			System.out.println(domain);

		}

	}

	/**
	 * 
	 * 从词向量知识文件中读取must-link分组
	 * 
	 * @param domain
	 * @return must-link Map
	 * @throws IOException
	 */
	public static Map<String, Integer> getKnowledgeMap(String domain)
			throws IOException {

		File f = new File("file//knowledge//" + domain + ".txt");
		BufferedReader reader = new BufferedReader(new InputStreamReader(
				new FileInputStream(f), "UTF-8"));
		String line = "";

		Map<String, Integer> group = new HashMap<>();

		Set<String> vocab = new HashSet<>();

		while ((line = reader.readLine()) != null) {

			String[] temp = line.split("\t");

			vocab.add(temp[0]);

			vocab.add(temp[1]);

		}

		reader.close();
		// 读完一遍,构建词表
		List<String> words = new ArrayList<>(vocab);

		for (String word : words) {

			group.put(word, words.indexOf(word));

		}

		reader = new BufferedReader(new InputStreamReader(
				new FileInputStream(f), "UTF-8"));

		while ((line = reader.readLine()) != null) {

			String[] temp = line.split("\t");

			int label_1 = group.get(temp[0]);

			int label_2 = group.get(temp[1]);

			if (label_1 <= label_2) {

				for (String word : group.keySet()) {

					if (group.get(word) == label_2) {

						group.put(word, label_1);
					}

				}

			} else {

				for (String word : group.keySet()) {

					if (group.get(word) == label_1) {

						group.put(word, label_2);
					}

				}

			}

		}

		reader.close();

		return group;
	}

	/**
	 * 为每个领域产生分组的word2vec知识，字符串，用来写文件
	 * 
	 * @param domain
	 * @return
	 * @throws IOException
	 */
	public static String generateGroupKnowledge(String domain,
			Map<String, double[]> word_vector) throws IOException {

		List<String> vocab = Corpus.getVocab("data//" + domain + "//" + domain
				+ ".vocab");

		int[][] docs = Corpus.getDocuments("data//" + domain + "//" + domain
				+ ".docs");

		Map<String, Integer> word_map = getKnowledgeMap(domain);

		Set<Integer> group_id = new HashSet<>();

		for (int e : word_map.values()) {
			group_id.add(e);
		}

		StringBuilder sb = new StringBuilder();

		for (int e : group_id) {

			Set<String> words = new HashSet<>();

			// 词group
			for (String word : word_map.keySet()) {

				if (word_map.get(word) == e) {
					words.add(word);
				}
			}

			// 有了A,B，不要B,A

			Set<Set<String>> must_links = new HashSet<>();
			// 两两词向量距离，PMI
			for (String word_1 : words) {

				for (String word_2 : words) {

					if (!word_1.equals(word_2)) {

						Set<String> must_link = new HashSet<>();

						must_link.add(word_1);
						must_link.add(word_2);

						if (must_links.contains(must_link))
							continue;
						else
							must_links.add(must_link);

						int word_i = vocab.indexOf(word_1);

						int word_j = vocab.indexOf(word_2);

						double pmi = Corpus.PMI(docs, word_i, word_j);

						if (pmi > 0) {

							double[] vector_1 = word_vector.get(word_1);

							double[] vector_2 = word_vector.get(word_2);

							double distance = Common.cosine_distance(vector_1,
									vector_2);

							if (distance < 1)

								sb.append(word_1 + "\t" + word_2 + "\t"
										+ (1 - distance) + "\t" + pmi + "\n");
						}

					}

				}
			}
			sb.append("\n");
		}
		return sb.toString();
	}

}
