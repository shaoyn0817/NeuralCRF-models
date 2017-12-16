/**
 * 
 */
package org.statnlp.example;

import java.util.ArrayList;
import java.util.List;

import org.statnlp.commons.types.Instance;
import org.statnlp.commons.types.Label;
import org.statnlp.commons.types.LinearInstance;
import org.statnlp.commons.types.Span;
import org.statnlp.example.weak_semi_crf.WeakSemiCRFFeatureManager;
import org.statnlp.example.weak_semi_crf.WeakSemiCRFInstanceParser;
import org.statnlp.example.weak_semi_crf.WeakSemiCRFNetworkCompiler;
import org.statnlp.util.GenericPipeline;

/**
 * An example of how to implement a custom pipeline by extending the GenericPipeline class.
 */
public class WeakSemiCRFPipeline extends GenericPipeline {
	
	public WeakSemiCRFInstanceParser initInstanceParser(){
		if(instanceParser != null){
			return (WeakSemiCRFInstanceParser)instanceParser;
		}
		return new WeakSemiCRFInstanceParser(this);
	}
	
	public WeakSemiCRFNetworkCompiler initNetworkCompiler(){
		if(networkCompiler != null){
			return (WeakSemiCRFNetworkCompiler)networkCompiler;
		}
		return new WeakSemiCRFNetworkCompiler(this);
	}
	
	public WeakSemiCRFFeatureManager initFeatureManager(){
		if(featureManager != null){
			return (WeakSemiCRFFeatureManager)featureManager;
		}
		return new WeakSemiCRFFeatureManager(this);
	}
	
	public void evaluate(Instance[] predictions){
		int corr = 0;
		int totalGold = 0;
		int totalPred = 0;
		for(Instance inst: predictions){
			@SuppressWarnings("unchecked")
			LinearInstance<Span> instance = (LinearInstance<Span>)inst;
			StringBuilder input = new StringBuilder();
			for(String[] inputs: instance.input){
				input.append(inputs[0]);
			}
			System.out.println("Input:");
			System.out.println(input);
			System.out.println("Gold:");
			System.out.println(instance.output);
			System.out.println("Prediction:");
			System.out.println(instance.prediction);
			List<Span> goldSpans = instance.output;
			List<Span> predSpans = instance.prediction;
			int curTotalGold = goldSpans.size();
			totalGold += curTotalGold;
			int curTotalPred = predSpans.size();
			totalPred += curTotalPred;
			int curCorr = countOverlaps(goldSpans, predSpans);
			corr += curCorr;
			double precision = 100.0*curCorr/curTotalPred;
			double recall = 100.0*curCorr/curTotalGold;
			double f1 = 2/((1/precision)+(1/recall));
			if(curTotalPred == 0) precision = 0.0;
			if(curTotalGold == 0) recall = 0.0;
			if(curTotalPred == 0 || curTotalGold == 0) f1 = 0.0;
			System.out.println(String.format("Correct: %1$3d, Predicted: %2$3d, Gold: %3$3d ", curCorr, curTotalPred, curTotalGold));
			System.out.println(String.format("Overall P: %#5.2f%%, R: %#5.2f%%, F: %#5.2f%%", precision, recall, f1));
			System.out.println();
			printScore(new Instance[]{instance});
			System.out.println();
		}
		System.out.println();
		System.out.println("### Overall score ###");
		System.out.println(String.format("Correct: %1$3d, Predicted: %2$3d, Gold: %3$3d ", corr, totalPred, totalGold));
		double precision = 100.0*corr/totalPred;
		double recall = 100.0*corr/totalGold;
		double f1 = 2/((1/precision)+(1/recall));
		if(totalPred == 0) precision = 0.0;
		if(totalGold == 0) recall = 0.0;
		if(totalPred == 0 || totalGold == 0) f1 = 0.0;
		System.out.println(String.format("Overall P: %#5.2f%%, R: %#5.2f%%, F: %#5.2f%%", precision, recall, f1));
		System.out.println();
		printScore(predictions);
	}
	
	private static List<Span> duplicate(List<Span> list){
		List<Span> result = new ArrayList<Span>();
		for(Span span: list){
			result.add(span);
		}
		return result;
	}
	
	private void printScore(Instance[] instances){
		int size = ((WeakSemiCRFInstanceParser)instanceParser).LABELS.size();
		int[] corrects = new int[size];
		int[] totalGold = new int[size];
		int[] totalPred = new int[size];
		for(Instance inst: instances){
			@SuppressWarnings("unchecked")
			LinearInstance<Span> instance = (LinearInstance<Span>)inst;
			List<Span> predicted = duplicate(instance.getPrediction());
			List<Span> actual = duplicate(instance.getOutput());
			for(Span span: actual){
				if(predicted.contains(span)){
					predicted.remove(span);
					Label label = span.label;
					corrects[label.getId()] += 1;
					totalPred[label.getId()] += 1;
				}
				totalGold[span.label.getId()] += 1;
			}
			for(Span span: predicted){
				totalPred[span.label.getId()] += 1;
			}
		}
		double avgF1 = 0;
		for(int i=0; i<size; i++){
			double precision = (totalPred[i] == 0) ? 0.0 : 1.0*corrects[i]/totalPred[i];
			double recall = (totalGold[i] == 0) ? 0.0 : 1.0*corrects[i]/totalGold[i];
			double f1 = (precision == 0.0 || recall == 0.0) ? 0.0 : 2/((1/precision)+(1/recall));
			avgF1 += f1;
			System.out.println(String.format("%6s: #Corr:%2$3d, #Pred:%3$3d, #Gold:%4$3d, Pr=%5$#5.2f%% Rc=%6$#5.2f%% F1=%7$#5.2f%%", ((WeakSemiCRFInstanceParser)instanceParser).getLabel(i).getForm(), corrects[i], totalPred[i], totalGold[i], precision*100, recall*100, f1*100));
		}
		System.out.printf("Macro average F1: %.2f%%", 100*avgF1/size);
	}
	
	/**
	 * Count the number of overlaps (common elements) in the given lists.
	 * Duplicate objects are counted as separate objects.
	 * @param list1
	 * @param list2
	 * @return
	 */
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
	
	public static void main(String[] args){
		new WeakSemiCRFPipeline().parseArgs(args).execute();
	}
}
