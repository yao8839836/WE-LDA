package wordvector;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

public class Word2Vec {

	Map<String, double[]> wordMap;

	int topNSize = 20;

	/**
	 * 构造方法,从文件中得到词向量
	 * 
	 * @param filename
	 *            存储词向量的文件
	 * @throws IOException
	 */
	public Word2Vec(String filename) throws IOException {

		File f = new File(filename);
		BufferedReader reader = new BufferedReader(new InputStreamReader(
				new FileInputStream(f), "UTF-8"));
		String line = "";

		reader.readLine();

		wordMap = new HashMap<>();

		while ((line = reader.readLine()) != null) {

			String[] temp = line.split(" ");

			double[] vector = new double[temp.length - 1];

			for (int i = 0; i < vector.length; i++) {

				vector[i] = Double.parseDouble(temp[i + 1]);
			}

			wordMap.put(temp[0], vector);

		}

		reader.close();

	}

	/**
	 * 返回与查询词余弦距离最近的词
	 * 
	 * @param queryWord
	 *            查询词
	 * @return 相近词及其余弦相似度的集合
	 */
	public Set<WordEntry> distance(String queryWord) {

		double[] center = wordMap.get(queryWord);
		if (center == null) {
			return Collections.emptySet();
		}

		int resultSize = wordMap.size() < topNSize ? wordMap.size() : topNSize;
		TreeSet<WordEntry> result = new TreeSet<WordEntry>();

		double norm = 0;
		for (int i = 0; i < center.length; i++) {
			norm += center[i] * center[i];
		}
		norm = Math.sqrt(norm);

		double min = Double.MIN_VALUE;

		for (Map.Entry<String, double[]> entry : wordMap.entrySet()) {

			double[] vector = entry.getValue();
			double dist = 0;
			for (int i = 0; i < vector.length; i++) {
				dist += center[i] * vector[i];
			}
			double norm1 = 0;
			for (int i = 0; i < vector.length; i++) {

				norm1 += vector[i] * vector[i];
			}
			norm1 = Math.sqrt(norm1);
			dist = (double) (dist / (norm * norm1));
			if (dist > min) {
				result.add(new WordEntry(entry.getKey(), dist));
				if (resultSize < result.size()) {
					result.pollLast();
				}
				min = result.last().score;
			}
		}
		result.pollFirst();// 本身

		return result;
	}

	/**
	 * 获取词向量表
	 * 
	 * @return 词向量表
	 */
	public Map<String, double[]> getWordVector() {

		return wordMap;
	}

}
