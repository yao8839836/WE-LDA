package test;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Set;

import knowledge.KnowledgeExtract;
import Jama.Matrix;
import Jama.SingularValueDecomposition;
import util.Common;
import util.Corpus;
import util.ReadWriteFile;
import wordvector.Word2Vec;
import wordvector.WordEntry;

public class TestWordVector {

	public static void main(String[] args) throws IOException {

		String filename = "file//amazon_word.vec";

		/*
		 * 读词向量测试
		 */

		Word2Vec w2v = new Word2Vec(filename);

		Map<String, double[]> word_vector = w2v.getWordVector();

		double[] vector = word_vector.get("android");

		for (double e : vector) {

			System.out.println(e);
		}

		/*
		 * PCA测试，可视化一下
		 */
		double[][] word_matrix = new double[word_vector.keySet().size()][100];

		int index = 0;

		for (String word : word_vector.keySet()) {

			double[] word_vec = word_vector.get(word);

			for (int i = 0; i < word_vec.length; i++) {

				word_matrix[index][i] = word_vec[i];
			}

			index++;

		}

		Matrix embedding = new Matrix(word_matrix);

		System.out.println("Embedding = U S V^T");

		SingularValueDecomposition s = embedding.svd();

		Matrix U = s.getU();

		Matrix S = s.getS();

		// 按行压缩，降维，2维，可以可视化

		Matrix U_sub = U.getMatrix(0, U.getRowDimension() - 1, 0, 1);

		Matrix compress = U_sub.times(S.getMatrix(0, 1, 0, 1));

		double[][] matrix = compress.getArray();

		StringBuilder sb = new StringBuilder();

		for (double[] row : matrix) {

			for (double e : row) {

				sb.append(e + "\t");
			}
			sb.append("\n");
		}

		ReadWriteFile.writeFile("file//wordvec_2d.txt", sb.toString()
				.replaceAll("\t\n", "\n"));

		/*
		 * 测试词的距离
		 */

		double[] vector_1 = word_vector.get("laptop");
		double[] vector_2 = word_vector.get("wifi");

		double euclidean = Common.euclidean_distance(vector_1, vector_2);

		double cosine = Common.cosine_distance(vector_1, vector_2);

		System.out.println("欧氏距离： " + euclidean);

		System.out.println("余弦距离： " + cosine);

		/*
		 * 测试最相近的词
		 */

		Set<WordEntry> word_set = w2v.distance("charge");

		System.out.println(word_set);

		/*
		 * 只显示当前语料库中相似的词
		 */

		String domain = "Battery";

		List<String> vocab = Corpus.getVocab("data//" + domain + "//" + domain
				+ ".vocab");

		int[][] docs = Corpus.getDocuments("data//" + domain + "//" + domain
				+ ".docs");

		for (WordEntry word : word_set) {

			if (vocab.contains(word.name))
				System.out.print(word.name
						+ "\t"
						+ word.score
						+ " "
						+ Corpus.PMI(docs, vocab.indexOf(word.name),
								vocab.indexOf("charge")) + ", ");

		}

		System.out.println();

		/*
		 * 测试读Top words
		 */

		filename = "data//" + domain + "//" + domain + ".twords";
		Set<String> top_words = KnowledgeExtract.getTopWords(filename, 15);
		System.out.println(top_words);

		/*
		 * 只显示Top words中相似的词
		 */

		for (WordEntry word : word_set) {

			if (top_words.contains(word.name))
				System.out.print(word.name
						+ "\t"
						+ word.score
						+ " "
						+ Corpus.PMI(docs, vocab.indexOf(word.name),
								vocab.indexOf("charge")) + ", ");

		}

	}

}
