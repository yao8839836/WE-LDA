package test;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import topic.GPU;
import util.Common;
import util.Corpus;
import util.ReadWriteFile;

public class RunGpuLDA {

	public static void main(String[] args) throws IOException {

		File[] files = new File("data//Non-Electronics//").listFiles();

		List<String> domain_list = new ArrayList<String>();

		for (File f : files) {
			String file_path = f.toString();
			String domain = file_path.substring(
					file_path.lastIndexOf("\\") + 1, file_path.length());

			domain_list.add(domain);
		}

		double coherence = 0;

		StringBuilder sb = new StringBuilder();

		for (String domain : domain_list) {

			double domain_coherence = runGpuLDA(domain);
			coherence += domain_coherence;
			sb.append(domain + "\t" + domain_coherence + "\n");
			System.out.println(domain + "\t" + domain_coherence + "\n");
		}

		sb.append("average : " + coherence / domain_list.size() + "\n");

		String filename = "file//gpu_20_top10_coherence_non.txt";

		ReadWriteFile.writeFile(filename, sb.toString());

	}

	/**
	 * 对指定文档集执行GPU-LDA
	 * 
	 * @param domain
	 * @return
	 * @throws IOException
	 */
	public static double runGpuLDA(String domain) throws IOException {

		List<String> vocab = Corpus.getVocab("data//Non-Electronics//" + domain
				+ "//" + domain + ".vocab");

		int[][] docs = Corpus.getDocuments("data//Non-Electronics//" + domain
				+ "//" + domain + ".docs");

		int K = 20;
		double alpha = 1;
		double beta = 0.1;
		int iterations = 2000;

		int top_word_count = 10;

		GPU gpu = new GPU(docs, vocab.size());

		gpu.setSchema();

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

		ReadWriteFile.writeFile(filename, sb.toString());

		double[][] theta = gpu.estimateTheta();

		// perplexity
		double perplexity = Corpus.perplexity(theta, phi, docs);
		System.out.println("perplexity : " + perplexity);

		return average_coherence;
	}

}
