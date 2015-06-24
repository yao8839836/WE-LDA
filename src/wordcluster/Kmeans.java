package wordcluster;

import net.sf.javaml.clustering.KMeans;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.Instance;
import net.sf.javaml.distance.CosineDistance;
import net.sf.javaml.distance.EuclideanDistance;
import weka.clusterers.SimpleKMeans;
import weka.core.Instances;

public class Kmeans {

	/**
	 * Kmeans in Weka, Euclidean distance
	 * 
	 * @param dataset
	 *            数据集
	 * @param K
	 *            簇数
	 * @return 每个数据的簇
	 * @throws Exception
	 */
	public static int[] RunWekaKmeans(Instances dataset, int K)
			throws Exception {

		int[] assignment = new int[dataset.numInstances()];

		SimpleKMeans km = new SimpleKMeans();

		km.setNumClusters(K);

		km.buildClusterer(dataset);

		for (int i = 0; i < assignment.length; i++) {

			int cluster = km.clusterInstance(dataset.instance(i));

			assignment[i] = cluster;
		}

		return assignment;
	}

	/**
	 * 
	 * 用Java ML中的Kmeans对数据集聚类，采用欧式距离
	 * 
	 * @param data
	 *            数据集
	 * @param K
	 *            簇数
	 * @return 每个数据的簇
	 * @throws Exception
	 */
	public static int[] RunKmeansEuclidean(Dataset data, int K)
			throws Exception {

		KMeans km = new KMeans(K, 100, new EuclideanDistance());
		Dataset[] clusters = km.cluster(data);

		int[] assignments = new int[data.size()];

		for (int i = 0; i < clusters.length; i++) {

			Dataset cluster = clusters[i];

			for (Instance ins : cluster) {

				int index = data.indexOf(ins);

				assignments[index] = i;

			}

		}

		return assignments;

	}

	/**
	 * 用Java ML中的Kmeans对数据集聚类，采用余弦距离
	 * 
	 * @param data
	 *            数据集
	 * @param K
	 *            簇数
	 * @return 每个数据的簇
	 * @throws Exception
	 */
	public static int[] RunKmeansCosine(Dataset data, int K) throws Exception {

		KMeans km = new KMeans(K, 100, new CosineDistance());
		Dataset[] clusters = km.cluster(data);

		int[] assignments = new int[data.size()];

		for (int i = 0; i < clusters.length; i++) {

			Dataset cluster = clusters[i];

			for (Instance ins : cluster) {

				int index = data.indexOf(ins);

				assignments[index] = i;

			}

		}

		return assignments;

	}

}
