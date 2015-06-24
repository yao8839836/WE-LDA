package util;

public class Common {

	/**
	 * 返回数组中最大元素的下标
	 * 
	 * @param array
	 *            输入数组
	 * @return 最大元素的下标
	 */
	public static int maxIndex(double[] array) {
		double max = array[0];
		int maxIndex = 0;
		for (int i = 1; i < array.length; i++) {
			if (array[i] > max) {
				max = array[i];
				maxIndex = i;
			}

		}
		return maxIndex;

	}

	/**
	 * 返回数组中最小元素的下标
	 * 
	 * @param array
	 *            输入数组
	 * @return 最小元素的下标
	 */
	public static int minIndex(double[] array) {
		double min = array[0];
		int minIndex = 0;
		for (int i = 1; i < array.length; i++) {
			if (array[i] < min) {
				min = array[i];
				minIndex = i;
			}

		}
		return minIndex;

	}

	/**
	 * 返回数组中的最小值
	 * 
	 * @param array
	 *            输入数组
	 * @return
	 */
	public static double min(double[] array) {

		double min = array[0];

		for (int i = 0; i < array.length; i++) {
			if (array[i] < min)
				min = array[i];
		}

		return min;

	}

	/**
	 * 复制矩阵
	 * 
	 * @param array
	 *            矩阵
	 * @return
	 */
	public static double[][] makeCopy(double[][] array) {

		double[][] copy = new double[array.length][];

		for (int i = 0; i < copy.length; i++) {

			copy[i] = new double[array[i].length];

			for (int j = 0; j < copy[i].length; j++) {
				copy[i][j] = array[i][j];
			}
		}

		return copy;
	}

	/**
	 * 把数据规范化到[0,1]
	 * 
	 * @param array
	 *            输入数组
	 * @return
	 */
	public static double[] normalize(double[] array) {

		double[] normalized_array = new double[array.length];

		double max = array[maxIndex(array)];

		double min = min(array);

		for (int i = 0; i < array.length; i++) {
			if (min == max)
				normalized_array[i] = max;
			else
				normalized_array[i] = 1 * (array[i] - min) / (max - min);
		}

		return normalized_array;
	}

	/**
	 * 求两个向量的欧氏距离
	 * 
	 * @param vector_1
	 * @param vector_2
	 * @return
	 */
	public static double euclidean_distance(double[] vector_1, double[] vector_2) {

		double distance = 0;

		for (int i = 0; i < vector_1.length; i++) {

			distance += (vector_1[i] - vector_2[i])
					* (vector_1[i] - vector_2[i]);
		}

		return Math.sqrt(distance);
	}

	/**
	 * 求两个向量的余弦距离
	 * 
	 * @param vector_1
	 * @param vector_2
	 * @return
	 */
	public static double cosine_distance(double[] vector_1, double[] vector_2) {

		double norm_1 = 0;

		for (int i = 0; i < vector_1.length; i++)
			norm_1 += vector_1[i] * vector_1[i];
		norm_1 = Math.sqrt(norm_1);

		double norm_2 = 0;

		for (int i = 0; i < vector_2.length; i++)
			norm_2 += vector_2[i] * vector_2[i];
		norm_2 = Math.sqrt(norm_2);

		double sim = 0;

		for (int i = 0; i < vector_1.length; i++)
			sim += vector_1[i] * vector_2[i];

		sim = sim / (norm_1 * norm_2);

		if (sim < 0)
			sim = 0;

		return 1 - sim;
	}

	/**
	 * 向量求和
	 * 
	 * @param vector_1
	 * @param vector_2
	 * @return 向量对应位置相加
	 */
	public static double[] add(double[] vector_1, double[] vector_2) {

		double[] vector = new double[vector_1.length];

		for (int i = 0; i < vector_1.length; i++) {

			vector[i] = vector_1[i] + vector_2[i];

		}

		return vector;
	}

}
