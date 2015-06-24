package wordcluster;

import net.sf.javaml.clustering.KMedoids;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.Instance;
import net.sf.javaml.distance.CosineDistance;
import net.sf.javaml.distance.EuclideanDistance;

public class Kmedoids {

	/**
	 * 
	 * 用Java ML中的Kmedoids对数据集聚类，采用欧式距离
	 * 
	 * @param data
	 *            数据集
	 * @param K
	 *            簇数
	 * @return 每个数据的簇
	 * @throws Exception
	 */
	public static int[] RunKmedoidsEuclidean(Dataset data, int K)
			throws Exception {

		KMedoids km = new KMedoids(K, 100, new EuclideanDistance());
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
	 * 用Java ML中的Kmedoids对数据集聚类，采用余弦距离
	 * 
	 * @param data
	 *            数据集
	 * @param K
	 *            簇数
	 * @return 每个数据的簇
	 * @throws Exception
	 */
	public static int[] RunKmedoidsCosine(Dataset data, int K) throws Exception {

		KMedoids km = new KMedoids(K, 100, new CosineDistance());

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
