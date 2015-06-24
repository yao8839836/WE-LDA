package test;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.Map;

import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.DefaultDataset;
import net.sf.javaml.core.DenseInstance;
import net.sf.javaml.core.Instance;
import util.Common;
import util.ReadWriteFile;
import wordcluster.Kmeans;
import wordvector.Word2Vec;

public class RunKmeans {

	public static void main(String[] args) throws Exception {
		/*
		 * 读word2vec must link
		 */
		File f = new File("file//knowledge//Alarm Clock.txt");
		BufferedReader reader = new BufferedReader(new InputStreamReader(
				new FileInputStream(f), "UTF-8"));
		String line = "";

		// 读词向量

		String filename = "file//amazon_word.vec";

		Word2Vec w2v = new Word2Vec(filename);

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
		System.out.println(data.size());

		int[] assignment = Kmeans.RunKmeansCosine(data, 15);

		// assignment = Kmedoids.RunKmedoidsCosine(data, 15);

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

		ReadWriteFile.writeFile("file//knowledge//Alarm Clock.txt",
				sb.toString());

	}
}
