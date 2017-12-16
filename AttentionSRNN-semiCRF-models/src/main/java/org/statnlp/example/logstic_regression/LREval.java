package org.statnlp.example.logstic_regression;

import org.statnlp.commons.types.Instance;

public class LREval {

	
	/**
	 * Evaluate the relation
	 * @param results
	 * @return the precision, recall and fscore
	 */
	public static double[] evaluate(Instance[] results) {
		//calculating the precision fist.
		int[] p = new int[RelationType.RELS.size()];
		int[] totalPredict = new int[RelationType.RELS.size()];
		int[] totalInData = new int[RelationType.RELS.size()];
		double[] metrics = new double[3];
		for (Instance inst : results) {
			LRInstance res = (LRInstance)inst;
			RelationType prediction = res.getPrediction();
			RelationType gold = res.getOutput();
			String predForm = prediction.form;
			String goldForm = gold.form;
			int corrPredId = prediction.id;
			int corrGoldId = gold.id;
			if (predForm.endsWith(Config.REV_SUFF)) {
				corrPredId = RelationType.get(predForm.replace(Config.REV_SUFF, "")).id;
			}
			if (goldForm.endsWith(Config.REV_SUFF)) {
				corrGoldId = RelationType.get(goldForm.replace(Config.REV_SUFF, "")).id;
			}
			if (prediction.equals(gold)) {
				p[corrPredId]++;
			}
			totalPredict[corrPredId]++;
			totalInData[corrGoldId]++;
		}
		
		int allP = 0;
		int allPredict = 0;
		int allInData = 0;
		for (int r = 0; r < RelationType.RELS.size(); r++) {
			if (RelationType.get(r).form.equals(Config.NR) || RelationType.get(r).form.endsWith(Config.REV_SUFF)) continue;
			double precision = p[r] * 1.0/totalPredict[r] * 100;
			double recall = p[r] * 1.0 / totalInData[r] * 100;
			double fscore = 2.0 * p[r] / (totalPredict[r] + totalInData[r]) * 100;
			String spacing = "\t";
			System.out.printf("[Result] %s: %sPrec.:%.2f%%\tRec.:%.2f%%\tF1.:%.2f%%\n", 
					RelationType.get(r).form, spacing, precision, recall, fscore);
			allP += p[r];
			allPredict += totalPredict[r];
			allInData += totalInData[r];
		}
		double precision = allP * 1.0/ allPredict * 100;
		double recall = allP * 1.0 / allInData * 100;
		double fscore = 2.0 * allP / (allPredict + allInData) * 100;
		metrics[0] = precision;
		metrics[1] = recall;
		metrics[2] = fscore;
		System.out.printf("[Result] All: \t\tPrec.:%.2f%%\tRec.:%.2f%%\tF1.:%.2f%%\n", 
				precision, recall, fscore);
		return metrics;
	}
	
}
