# WE-LDA
The implementation of our paper:

Liang Yao, Yin Zhang, Qinfei Chen, Hongze Qian, Baogang Wei and Zhifeng Hu. "Mining Coherent Topics in Documents using Word Embeddings and Large-scale Text Data." Engineering Applications of Artificial Intelligence 64 (2017): 432-439.

Require Java 7+ and Eclipse. I use Java 8 in the project.

# Main entries

/src/test/MultiDomainWeLDA.java

Run this file can produce topics and compute topic coherence  for each domain.

In the Main method:

     File[] files = new File("data//Electronics//").listFiles(); 

is to read multiple domain data.

The code snippet below is to run WE-LDA, generate topics and compute topic coherence for each domain. 

		for (String domain : domain_list) {

			double domain_coherence = runWeLDA(domain, w2v);
			coherence += domain_coherence;
			sb.append(domain + "\t" + domain_coherence + "\n");
			System.out.println(domain + "\t" + domain_coherence + "\n");

		}

# Dataset

/data/Electronics/

/data/Non-Electronics/

We used the pre-processed version from:

Chen, Z., & Liu, B. (2014a). Mining topics in documents: standing on the shoulders of big data. In KDD (pp. 1116–
1125). ACM.

If you need the original full text reviews, please visit https://github.com/czyuan/AMC and contact the author of the above paper.

# Word Vector Files

/file/amazon_word.vec for Electronics dataset.

/file/amazon_word_non_elect.vec for Non-Electronics dataset.

Both are learned from the 50 domains text in each dataset.

Input files for Skip-Gram:

/file/amazon_docs.txt for Electronics dataset.

/file/amazon_non_elect_docs.txt for Non-Electronics dataset.

# Generated Knowledge

In /file/knowledge/.
    

