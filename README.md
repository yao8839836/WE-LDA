# WE-LDA
Incorporating Word Embeddings into Topic Modeling

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

    

