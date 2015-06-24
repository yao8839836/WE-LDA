package knowledge;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashSet;
import java.util.Set;

public class KnowledgeExtract {

	/**
	 * 从指定文件中读取LDA产生的Top words
	 * 
	 * @param filename
	 *            文件名
	 * @param top_n
	 *            前多少个词
	 * @return 词集合
	 * @throws IOException
	 * 
	 */
	public static Set<String> getTopWords(String filename, int top_n)
			throws IOException {

		File f = new File(filename);
		BufferedReader reader = new BufferedReader(new InputStreamReader(
				new FileInputStream(f), "UTF-8"));
		String line = "";

		reader.readLine();

		Set<String> top_words = new HashSet<>();

		int line_count = 0;

		while ((line = reader.readLine()) != null && line_count < top_n) {

			String[] temp = line.split("\t");

			for (String word : temp) {

				top_words.add(word);
			}

			line_count++;

		}

		reader.close();

		return top_words;
	}

}
