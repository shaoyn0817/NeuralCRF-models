package org.statnlp.example;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.statnlp.commons.types.Instance;
import org.statnlp.example.tree_crf.BinaryTree;
import org.statnlp.example.tree_crf.CNFRule;
import org.statnlp.example.tree_crf.Label;
import org.statnlp.example.tree_crf.TreeCRFFeatureManager;
import org.statnlp.example.tree_crf.TreeCRFInstance;
import org.statnlp.example.tree_crf.TreeCRFNetworkCompiler;
import org.statnlp.hypergraph.DiscriminativeNetworkModel;
import org.statnlp.hypergraph.GlobalNetworkParam;
import org.statnlp.hypergraph.NetworkConfig;
import org.statnlp.hypergraph.NetworkModel;

public class TreeCRFMain {
	
	public static ArrayList<Label> labels;
	public static Map<Label, Set<CNFRule>> rules;
	
	public static void main(String[] args) throws InterruptedException, IOException, ClassNotFoundException{
		boolean serializeModel = false;
		boolean useToyData = false;
		
		String train_filename = "data/ptb-binary.train";
		String test_filename = "data/ptb-binary.test";
		
		TreeCRFInstance[] trainInstances;
		TreeCRFInstance[] testInstances;
		if(useToyData){
			trainInstances = getToyData(true);
			testInstances = getToyData(false);
		} else {
			trainInstances = readPTB(train_filename, true, true);
			testInstances = readPTB(test_filename, true, false);
		}
		
		labels = new ArrayList<Label>();
		labels.addAll(Label.LABELS.values());
		rules = new HashMap<Label, Set<CNFRule>>();
		getRules(trainInstances);
		getRules(testInstances);
		
		NetworkConfig.TRAIN_MODE_IS_GENERATIVE = false;
		NetworkConfig.CACHE_FEATURES_DURING_TRAINING = true;
		NetworkConfig.L2_REGULARIZATION_CONSTANT = 0.01;
		NetworkConfig.OBJTOL = 1e-9;
		NetworkConfig.NUM_THREADS = 4;
		NetworkConfig.AVOID_DUPLICATE_FEATURES = true;
		NetworkConfig.PARALLEL_FEATURE_EXTRACTION = true;
		
		int numIterations = 500;
		
		int size = trainInstances.length;
		
		System.err.println("Read.."+size+" instances.");
		
		TreeCRFFeatureManager fm = new TreeCRFFeatureManager(new GlobalNetworkParam());
		
		TreeCRFNetworkCompiler compiler = new TreeCRFNetworkCompiler(labels, rules, Label.get("ROOT"));
		
		NetworkModel model = DiscriminativeNetworkModel.create(fm, compiler);
		
		if(new File("data/model").exists()){
			System.out.println("Reading object...");
			long startTime = System.currentTimeMillis();
			ObjectInputStream ois = new ObjectInputStream(new FileInputStream("data/model"));
			model = (NetworkModel)ois.readObject();
			ois.close();
			long endTime = System.currentTimeMillis();
			System.out.printf("Done in %.3fs\n", (endTime-startTime)/1000.0);
		} else {
			model.train(trainInstances, numIterations);
			if(serializeModel){
				System.out.println("Writing object...");
				long startTime = System.currentTimeMillis();
				ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("data/model"));
				oos.writeObject(model);
				oos.close();
				long endTime = System.currentTimeMillis();
				System.out.printf("Done in %.3fs\n", (endTime-startTime)/1000.0);
			}
		}
		
		System.out.println("Number of rules: "+rules.size());
		int k = 256;
		Instance[] predictions = model.decode(testInstances, k);
		int corr = 0;
		int totalGold = 0;
		int totalPred = 0;
		for(Instance inst: predictions){
			TreeCRFInstance instance = (TreeCRFInstance)inst;
			List<BinaryTree> topKPreds = instance.getTopKPredictions();
			System.out.println("Gold:");
			System.out.println(instance.output);
			System.out.println("Prediction:");
			System.out.println(instance.prediction);
			System.out.println("Size: "+topKPreds.size());
			System.out.println("K-th prediction:");
			System.out.println(topKPreds.get(topKPreds.size()-1));
			System.out.println();
			List<String> goldConstituents = instance.output.getConstituents();
			List<String> predConstituents = instance.prediction.getConstituents();
			int curTotalGold = goldConstituents.size();
			totalGold += curTotalGold;
			int curTotalPred = predConstituents.size();
			totalPred += curTotalPred;
			int curCorr = countOverlaps(goldConstituents, predConstituents);
			corr += curCorr;
			if(curTotalPred == 0) curTotalPred = 1;
			if(curTotalGold == 0) curTotalGold = 1;
			double precision = 100.0*curCorr/curTotalPred;
			double recall = 100.0*curCorr/curTotalGold;
			double f1 = 2/((1/precision)+(1/recall));
			System.out.println("Correct constituents: "+curCorr);
			System.out.println("Gold constituents: "+curTotalGold);
			System.out.println("Predicted constituents: "+curTotalPred);
			System.out.println(String.format("P: %.2f%%", precision));
			System.out.println(String.format("R: %.2f%%", recall));
			System.out.println(String.format("F: %.2f%%", f1));
		}
		System.out.println("Correct constituents: "+corr);
		System.out.println("Gold constituents: "+totalGold);
		System.out.println("Predicted constituents: "+totalPred);
		if(totalPred == 0) totalPred = 1;
		if(totalGold == 0) totalGold = 1;
		double precision = 100.0*corr/totalPred;
		double recall = 100.0*corr/totalGold;
		double f1 = 2/((1/precision)+(1/recall));
		System.out.println(String.format("P: %.2f%%", precision));
		System.out.println(String.format("R: %.2f%%", recall));
		System.out.println(String.format("F: %.2f%%", f1));
	}
	
	/**
	 * Count the number of overlaps (common elements) in the given lists.
	 * Duplicate objects are counted as separate objects.
	 * @param list1
	 * @param list2
	 * @return
	 */
	private static int countOverlaps(List<String> list1, List<String> list2){
		System.out.println(list1);
		System.out.println(list2);
		int result = 0;
		List<String> copy = new ArrayList<String>();
		copy.addAll(list2);
		for(String string: list1){
			if(copy.contains(string)){
				copy.remove(string);
				result += 1;
			}
		}
		return result;
	}
	
	private static void getRules(TreeCRFInstance[] instances){
		for(TreeCRFInstance instance: instances){
			getRules(instance.output);
		}
	}
	
	private static void getRules(BinaryTree tree){
		if(tree.left == null) return;
		Label leftSide = tree.value.label;
		Label firstRight = tree.left.value.label;
		Label secondRight = tree.right.value.label;
		CNFRule rule = new CNFRule(leftSide, firstRight, secondRight);
		if(!rules.containsKey(leftSide)){
			rules.put(leftSide, new HashSet<CNFRule>());
		}
		rules.get(leftSide).add(rule);
		getRules(tree.left);
		getRules(tree.right);
	}
	
	private static TreeCRFInstance[] getToyData(boolean isLabeled){
		ArrayList<TreeCRFInstance> result = new ArrayList<TreeCRFInstance>();
		ArrayList<BinaryTree> trees = new ArrayList<BinaryTree>();
		trees.add(BinaryTree.parse("(ROOT (NP (DT The)(NN duck))(VP (VBZ is)(VBG swimming)))"));
		trees.add(BinaryTree.parse("(ROOT (NP (DT The)(NNS cats))(VP (VBP are)(VBG running)))"));
		trees.add(BinaryTree.parse("(ROOT (NP (DT A)(NN dog))(VP (VBD was)(VBG swimming)))"));
		trees.add(BinaryTree.parse("(ROOT (NP (DT The)(NNS ducks))(VP (VBP are)(VBG running)))"));
		trees.add(BinaryTree.parse("(ROOT (NP (DT A)(NN dog))(VP (VBD was)(VBG swimming)))"));
		trees.add(BinaryTree.parse("(ROOT (NP (DT The)(NN dog))(VP (VBZ ducks)(RB hastily)))"));
		trees.add(BinaryTree.parse("(ROOT (NP (DT The)(NN dog))(VP (VBZ kills)(NNS cats)))"));
		trees.add(BinaryTree.parse("(ROOT (NP (DT The)(NNS cats))(VP (VBP duck)(RB hastily)))"));
		trees.add(BinaryTree.parse("(ROOT (NN I)(VP (VBP eat)(NP (NN spaghetti)(PP (IN with)(NN fish)))))"));
		trees.add(BinaryTree.parse("(ROOT (NN I)(VP (VP (VBP eat)(NN spaghetti))(PP (IN with)(NN fork))))"));
		trees.add(BinaryTree.parse("(S (NN I)(VBP duck))"));
		trees.add(BinaryTree.parse("(S (NN I)(VP (VBP duck)(RB there)))"));
		int instanceId = 1;
		for(BinaryTree tree: trees){
			TreeCRFInstance instance = new TreeCRFInstance(instanceId, 1);
			instanceId++;
			instance.input = tree.getWords();
			instance.output = tree;
			if(isLabeled){
				instance.setLabeled();
			} else {
				instance.setUnlabeled();
			}
			result.add(instance);
		}
		return result.toArray(new TreeCRFInstance[result.size()]);
	}
	
	private static TreeCRFInstance[] readPTB(String fileName, boolean withLabels, boolean isLabeled) throws IOException{
		InputStreamReader isr = new InputStreamReader(new FileInputStream(fileName), "UTF-8");
		BufferedReader br = new BufferedReader(isr);
		ArrayList<TreeCRFInstance> result = new ArrayList<TreeCRFInstance>();
		int instanceId = 1;
		while(br.ready()){
			String line = br.readLine();
			BinaryTree tree = BinaryTree.parse(line);
			TreeCRFInstance instance = new TreeCRFInstance(instanceId, 1);
			instanceId++;
			instance.input = tree.getWords();
			instance.output = tree;
			if(isLabeled){
				instance.setLabeled();
			} else {
				instance.setUnlabeled();
			}
			result.add(instance);
		}
		br.close();
		return result.toArray(new TreeCRFInstance[result.size()]);
	}

}
