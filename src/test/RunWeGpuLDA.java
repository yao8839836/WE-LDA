package test;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import knowledge.KnowledgeExtract;
import topic.WeGpuLDA;
import util.Common;
import util.Corpus;
import util.ReadWriteFile;
import wordvector.Word2Vec;
import wordvector.WordEntry;

public class RunWeGpuLDA {

	public static void main(String[] args) throws IOException {

		File[] files = new File("data//").listFiles();

		List<String> domain_list = new ArrayList<String>();

		for (File f : files) {
			String file_path = f.toString();
			String domain = file_path.substring(file_path.indexOf("\\") + 1,
					file_path.length());

			domain_list.add(domain);
		}

		double coherence = 0;

		Set<String> all_top_words = getAllTopWords(domain_list);

		StringBuilder sb = new StringBuilder();

		for (String domain : domain_list) {

			double domain_coherence = runWeGpuLDA(domain, all_top_words);
			coherence += domain_coherence;
			sb.append(domain + "\t" + domain_coherence + "\n");
			System.out.println(domain + "\t" + domain_coherence + "\n");
		}

		sb.append("average : " + coherence / domain_list.size() + "\n");

		String filename = "file//we_gpu_pmi_coherence.txt";

		ReadWriteFile.writeFile(filename, sb.toString());

	}

	/**
	 * 对指定文档集执行WE-GPU-LDA
	 * 
	 * @param domain
	 * @return
	 * @throws IOException
	 */
	public static double runWeGpuLDA(String domain, Set<String> all_top_words)
			throws IOException {

		// 读语料库

		List<String> vocab = Corpus.getVocab("data//" + domain + "//" + domain
				+ ".vocab");

		int[][] docs = Corpus.getDocuments("data//" + domain + "//" + domain
				+ ".docs");

		// 读词向量

		String filename = "file//amazon_word.vec";

		Word2Vec w2v = new Word2Vec(filename);

		// 读Top words

		filename = "data//" + domain + "//" + domain + ".twords";

		Set<String> top_words = KnowledgeExtract.getTopWords(filename, 30);

		// 获取知识，计算相关度矩阵，写文件

		int V = vocab.size();

		float[][] A = new float[V][V];

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
						A[word_i][word_j] = (float) (sim_word.score);

						A[word_j][word_i] = A[word_i][word_j];

						visited.add(must);
					}

				}

			}

		}

		filename = "file//knowledge//" + domain + ".txt";
		ReadWriteFile.writeFile(filename, sb.toString());

		int K = 15;
		double alpha = 1;
		double beta = 0.1;
		int iterations = 2000;

		int top_word_count = 30;

		WeGpuLDA gpu = new WeGpuLDA(docs, V, A);

		gpu.markovChain(K, alpha, beta, iterations);

		double[][] phi = gpu.estimatePhi();

		double[][] phi_copy = Common.makeCopy(phi);

		// 将每个主题的前10个词写文件
		double[][] phi_for_write = Common.makeCopy(phi);

		sb = new StringBuilder();

		for (double[] phi_t : phi_for_write) {

			for (int i = 0; i < 10; i++) {

				int max_index = Common.maxIndex(phi_t);

				sb.append(vocab.get(max_index) + "\t");

				phi_t[max_index] = 0;

			}
			sb.append("\n");

		}

		filename = "output//WE-GPU-PMI-LDA//" + domain + ".txt";

		// 语义一致性
		double average_coherence = Corpus.average_coherence(docs, phi_copy,
				top_word_count);

		// 多线程每个领域将coherence写在自己文件
		sb.append("average coherence\t" + average_coherence);

		System.out.println("average coherence : " + average_coherence);

		ReadWriteFile.writeFile(filename, sb.toString());

		double[][] theta = gpu.estimateTheta();

		// perplexity
		double perplexity = Corpus.perplexity(theta, phi, docs);
		System.out.println("perplexity : " + perplexity);

		return average_coherence;
	}

	/**
	 * 返回所有domain的Top words
	 * 
	 * @param domain_list
	 * @return
	 * @throws IOException
	 */
	public static Set<String> getAllTopWords(List<String> domain_list)
			throws IOException {

		Set<String> words = new HashSet<>();

		for (String domain : domain_list) {

			String filename = "data//" + domain + "//" + domain + ".twords";

			Set<String> top_words = KnowledgeExtract.getTopWords(filename, 15);

			words.addAll(top_words);

		}

		return words;
	}
}
