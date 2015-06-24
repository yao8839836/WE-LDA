package test;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import knowledge.KnowledgeExtract;
import util.Corpus;
import util.ReadWriteFile;
import wordvector.Word2Vec;

public class RunKnowlGen {

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

		// 读词向量

		String filename = "file//amazon_word.vec";

		Word2Vec w2v = new Word2Vec(filename);

		for (String domain : domain_list) {

			generateKnowledgeFile(domain, w2v);
			System.out.println(domain);
		}

	}

	/**
	 * 对指定领域生成知识文件
	 * 
	 * @param domain
	 * @param w2v
	 * @throws IOException
	 */
	public static void generateKnowledgeFile(String domain, Word2Vec w2v)
			throws IOException {

		// 词表，语料库
		List<String> vocab = Corpus.getVocab("data//" + domain + "//" + domain
				+ ".vocab");

		int[][] docs = Corpus.getDocuments("data//" + domain + "//" + domain
				+ ".docs");
		int top_seed_words_num = 15;

		String filename = "data//" + domain + "//" + domain + ".twords";

		Set<String> top_words = KnowledgeExtract.getTopWords(filename,
				top_seed_words_num);

		// 生成nust-link知识库,写文件

		String content = MultiDomainWeLDA.generateKnowledgeBase(top_words,
				vocab, docs, w2v);

		filename = "file//knowledge_welda//" + domain + ".txt";
		ReadWriteFile.writeFile(filename, content);

	}

}
