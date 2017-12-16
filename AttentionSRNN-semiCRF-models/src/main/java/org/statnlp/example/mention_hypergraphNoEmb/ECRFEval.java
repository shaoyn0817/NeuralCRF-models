package org.statnlp.example.mention_hypergraphNoEmb;


import java.io.IOException; 
import java.util.ArrayList;
import java.util.List;
import org.statnlp.commons.types.Instance;
import org.statnlp.hypergraph.decoding.Metric;

public class ECRFEval {
	
	public static boolean windows = false;
	
	/**
	 * 
	 * @param testInsts
	 * @param nerOut: word, true pos, true entity, pred entity
	 * @throws IOException
	 */
	public static Metric evalNER(Instance[] testInsts, String nerOut){

		    int corr = 0;
	     	int totalGold = 0;
		    int totalPred = 0;
		    double f1 = -1;
			for(int index=0;index<testInsts.length;index++){			
					MentionHypergraphInstance instance = (MentionHypergraphInstance)testInsts[index];
					List<Span> predictions = instance.prediction;
					List<Span> trueEntities = instance.getOutput();
					List<Span> goldSpans = instance.output;
					List<Span> predSpans = instance.prediction;
					int curTotalGold = goldSpans.size();
					totalGold += curTotalGold;
					int curTotalPred = predSpans.size();
					totalPred += curTotalPred;
					int curCorr = countOverlaps(goldSpans, predSpans);
					corr += curCorr;
					if(curTotalPred == 0) curTotalPred = 1;
					if(curTotalGold == 0) curTotalGold = 1;
			}
				if(totalPred == 0) totalPred = 1;
				if(totalGold == 0) totalGold = 1;
				double precision = 100.0*corr/totalPred;
				double recall = 100.0*corr/totalGold;
				f1 = 2/((1/precision)+(1/recall));
				if(totalPred == 0) precision = 0.0;
				if(totalGold == 0) recall = 0.0;
				if(totalPred == 0 || totalGold == 0) f1 = 0.0;
		System.out.println(f1);
	    return new NEMetric(f1);
	}
	
	
	
	
	
	
	private static int countOverlaps(List<Span> list1, List<Span> list2){
		int result = 0;
		List<Span> copy = new ArrayList<Span>();
		copy.addAll(list2);
		for(Span span: list1){
			if(copy.contains(span)){
				copy.remove(span);
				result += 1;
			}
		}
		return result;
	}
}
