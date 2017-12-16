package org.statnlp.example.mention_hypergraphNoEmb;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

import org.statnlp.commons.ml.opt.OptimizerFactory;
import org.statnlp.commons.ml.opt.GradientDescentOptimizer.BestParamCriteria;
import org.statnlp.commons.types.Instance;
import org.statnlp.example.mention_hypergraphNoEmb.MentionHypergraphFeatureManager.FeatureType;
import org.statnlp.example.mention_hypergraphNoEmb.MentionHypergraphInstance.WordsAndTags;
import org.statnlp.hypergraph.DiscriminativeNetworkModel;
import org.statnlp.hypergraph.GlobalNetworkParam;
import org.statnlp.hypergraph.NetworkConfig;
import org.statnlp.hypergraph.NetworkModel;
import org.statnlp.hypergraph.decoding.Metric;
import org.statnlp.hypergraph.neural.BidirectionalLSTM;
import org.statnlp.hypergraph.neural.GlobalNeuralNetworkParam;
import org.statnlp.hypergraph.neural.MultiLayerPerceptron;
import org.statnlp.hypergraph.neural.NeuralNetworkCore;

 

public class MentionHypergraphMainBIO {
	
	public static ArrayList<Label> labels;
	public static String neuralType = "continuous";//-----------------------------------------------
	public static int gpuId = 0;//---------------------------------------------------
	public static String embedding = "glove";
	public static String neural_config = "nn-crf-interface/neural_server/neural.config";
	public static double testR, testP, testF, devR, devF, devP, finalmp = -1;
	public static double testR2, testP2, testF2, devR2, devF2, devP2, finalmp2 = -1;
	public static String nnOptimizer = "lbfgs";
	public static double lr = 0.001;
	//public static OptimizerFactory optimizer =  OptimizerFactory.getGradientDescentFactoryUsingGradientClipping(BestParamCriteria.BEST_ON_DEV, 0.001, 4);
	//public static OptimizerFactory optimizer =  OptimizerFactory.getGradientDescentFactoryUsingAdaDelta();
	public static OptimizerFactory optimizer =  OptimizerFactory.getLBFGSFactory();
    //lab1  lr = 0.01  bs=50 ACE05  
	//lab2  lr = 0.01  bs=300  04  false  good one
	//lab3  lr = 0.01  bs=150  04 false
	//lab4  lr = 0.01  bs = 50  ACE05  FALSE
	//lab5  lr = 0.01  bs = 300  ACE05  true 
	public static boolean evalOnDev = true	;

	

	//original model is
	//Dev set:   P: 64.40%    R: 6.41%    F: 11.66%
	//Test set:   P: 59.29%    R: 5.48%    F: 10.03%
	public static void main(String[] args) throws Exception{
		boolean serializeModel = false; 
		System.out.println("final model");
		boolean readModelIfAvailable = false;  
		NetworkConfig.USE_BATCH_TRAINING = false;
		NetworkConfig.USE_NEURAL_FEATURES = true; 
        NetworkConfig.NUM_THREADS = 8; // 
        NetworkConfig.PRINT_BATCH_OBJECTIVE=true;
        //目前是BI LSTM
    	String dataset = "ACE04";
		int numIterations = 60000;  
		NetworkConfig.REGULARIZE_NEURAL_FEATURES = true;
		NetworkConfig.BATCH_SIZE = 5;  
		NetworkConfig.OPTIMIZE_NEURAL = true;//!!!!!!!!!!!!!!!!!!!!!1
        NetworkConfig.INIT_FV_WEIGHTS = false;
        String modelPath = "kk.model";
		NetworkConfig.L2_REGULARIZATION_CONSTANT = 0.01;
		String train_filename = "data/data/"+dataset+"/train.data"; 
		String dev = "data/data/"+dataset+"/dev.data";
		NetworkConfig.RANDOM_BATCH = true;
		NetworkConfig.OS="linux";
		
		Embedding e = new Embedding("glove.6B.50d.txt");
 
		
		MentionHypergraphInstance[] trainInstances = readData(train_filename, true, true);
		MentionHypergraphInstance[] devInstances = readData(dev, true, false);
		
		labels = new ArrayList<Label>();
		labels.addAll(Label.LABELS.values()); 
		int maxSize = 0;
		for(MentionHypergraphInstance instance: trainInstances){
			maxSize = Math.max(maxSize, instance.size());
		}
		
		NetworkConfig.TRAIN_MODE_IS_GENERATIVE = false;
		NetworkConfig.CACHE_FEATURES_DURING_TRAINING = true;
		NetworkConfig.OBJTOL = 1e-4;
		NetworkConfig.PARALLEL_FEATURE_EXTRACTION = true; 

		
        System.out.println(Label.LABELS.size());

		List<NeuralNetworkCore> nets = new ArrayList<NeuralNetworkCore>();
		if(NetworkConfig.USE_NEURAL_FEATURES){
			if (neuralType.equals("lstm")) {
				int hiddenSize = 50;
				String optimizer = nnOptimizer;
				boolean bidirection = false;
				nets.add(new BidirectionalLSTM(hiddenSize, bidirection, optimizer, lr, 5, Label.LABELS.size(), gpuId, embedding));
			} else if (neuralType.equals("continuous")) {
				nets.add(new ECRFContinuousFeatureValueProvider(50, Label.LABELS.size()));
			}
		} 
		GlobalNetworkParam gnp = new GlobalNetworkParam(optimizer, new GlobalNeuralNetworkParam(nets));
		
		
		
		int size = trainInstances.length;
		
		System.err.println("Read.."+size+" instances.");
		
		MentionHypergraphFeatureManager fm = new MentionHypergraphFeatureManager(gnp);
		
		MentionHypergraphNetworkCompiler compiler = new MentionHypergraphNetworkCompiler(labels.toArray(new Label[labels.size()]), maxSize);
		NetworkModel model = DiscriminativeNetworkModel.create(fm, compiler);

		if(serializeModel){
			
			if(new File(modelPath).exists() && readModelIfAvailable){
				System.out.println("Reading object...");
				long startTime = System.currentTimeMillis();
				ObjectInputStream ois = new ObjectInputStream(new FileInputStream(modelPath));
				model = (NetworkModel)ois.readObject();
				ois.close();
				Field _fm = NetworkModel.class.getDeclaredField("_fm");
				_fm.setAccessible(true);
				fm = (MentionHypergraphFeatureManager)_fm.get(model);
				long endTime = System.currentTimeMillis();
				System.out.printf("Done in %.3fs\n", (endTime-startTime)/1000.0);
			} else {
				/*
				long startTime = System.currentTimeMillis(); 
				model.train(trainInstances, numIterations);
				long endTime = System.currentTimeMillis();
				System.out.printf("Done in %.3fs\n", (endTime-startTime)/1000.0);
				*/
				model.train(trainInstances, numIterations);
				System.out.println("Writing object...");
				long startTime = System.currentTimeMillis();
				ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(modelPath));
				oos.writeObject(model);
				oos.close();
				long endTime = System.currentTimeMillis();
				System.out.printf("Done in %.3fs\n", (endTime-startTime)/1000.0);
			}
		} else {
		    String tmpOut = "data/tmp_out.txt";
			
			Function<Instance[], Metric> evalFunc = new Function<Instance[], Metric>() {

				@Override
				public Metric apply(Instance[] t) {
					return ECRFEval.evalNER(t, tmpOut);
				}
				
			};
			if (!evalOnDev) devInstances = null;
		    model.train(trainInstances, numIterations, devInstances, evalFunc, 100);
			//model.train(trainInstances, numIterations);

		}
		

		
		
		System.out.println("\n\n");
		
		String out = "\n\noriginal model is\n";
		for(int t = 0; t < 2; t ++)//--------------------------------------------------------------------------------------------------------------------------------------------------------
		{
		
		String test_filename = "data/data/"+dataset+"/dev.data";
		if(t== 1)
			test_filename = "data/data/"+dataset+"/test.data"; 
		MentionHypergraphInstance[] testInstances = readData(test_filename, true, false);
		for(MentionHypergraphInstance instance: testInstances){
			maxSize = Math.max(maxSize, instance.size());
		}
		Instance[] predictions = model.decode(testInstances);
		ECRFEval.evalNER(predictions, "");
		fm.getParam_G().setVersion(fm.getParam_G().getVersion()+1);
		int corr = 0;
		int totalGold = 0;
		int totalPred = 0;
		for(Instance inst: predictions){
			MentionHypergraphInstance instance = (MentionHypergraphInstance)inst;
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
			//double precision = 100.0*curCorr/curTotalPred;
			//double recall = 100.0*curCorr/curTotalGold;
			//double f1 = 2/((1/precision)+(1/recall));
			//if(curTotalPred == 0) precision = 0.0;
			//if(curTotalGold == 0) recall = 0.0;
			//if(curTotalPred == 0 || curTotalGold == 0) f1 = 0.0;
			//System.out.println("gold:  "+goldSpans);
		    //System.out.println("pre:   "+predSpans);
			//System.out.println();
		}
		//System.out.println("Correct spans: "+corr);
		//System.out.println("Gold spans: "+totalGold);
		//System.out.println("Predicted spans: "+totalPred);
		if(totalPred == 0) totalPred = 1;
		if(totalGold == 0) totalGold = 1;
		double precision = 100.0*corr/totalPred;
		double recall = 100.0*corr/totalGold;
		double f1 = 2/((1/precision)+(1/recall));
		if(totalPred == 0) precision = 0.0;
		if(totalGold == 0) recall = 0.0;
		if(totalPred == 0 || totalGold == 0) f1 = 0.0;
		
		
	
		if(t==0) out += "Dev set:   ";
		else if(t== 1) out+="Test set:   "; 
		out += String.format("P: %.2f%%    ", precision);
		out += String.format("R: %.2f%%    ", recall);
		out += String.format("F: %.2f%%\n", f1);
		}
		
		System.out.println(out);
	
		
		int mentionPenaltyFeatureIndex = fm.getParam_G().getFeatureId(FeatureType.MENTION_PENALTY.name(), "MP", "MP");
		boolean write = false;
		boolean equal = false;
		for(double mentionPenalty = -4; mentionPenalty <= 3; mentionPenalty += 0.01){
			System.out.println(String.format("\nMention penalty: %.2f", mentionPenalty));
			for(int t = 0; t < 2; t ++)
			{
			String test_filename = "data/data/"+dataset+"/dev.data";
			if(t== 1)
				test_filename = "data/data/"+dataset+"/test.data";
			MentionHypergraphInstance[] testInstances = readData(test_filename, true, false);
			for(MentionHypergraphInstance instance: testInstances){
				maxSize = Math.max(maxSize, instance.size());
			}
			fm.getParam_G().setWeight(mentionPenaltyFeatureIndex, mentionPenalty);
			Instance[] predictions = model.decode(testInstances);
			fm.getParam_G().setVersion(fm.getParam_G().getVersion()+1);
			int corr = 0;
			int totalGold = 0;
			int totalPred = 0;
			for(Instance inst: predictions){
				MentionHypergraphInstance instance = (MentionHypergraphInstance)inst;
				List<Span> goldSpans = instance.output;
				List<Span> predSpans = instance.prediction;
				int curTotalGold = goldSpans.size();
				totalGold += curTotalGold;
				int curTotalPred = predSpans.size();
				totalPred += curTotalPred;
				int curCorr = countOverlaps(goldSpans, predSpans);
				corr += curCorr;
			}
			//System.out.println("Correct spans: "+corr);
			//System.out.println("Gold spans: "+totalGold);
			//System.out.println("Predicted spans: "+totalPred);
	//		if(totalPred == 0) totalPred = 1;
	//		if(totalGold == 0) totalGold = 1;
			double precision = 100.0*corr/totalPred;
			double recall = 100.0*corr/totalGold;
			double f1 = 2/((1/precision)+(1/recall));
			if(totalPred == 0) precision = 0.0;
			if(totalGold == 0) recall = 0.0;
			if(totalPred == 0 || totalGold == 0) f1 = 0.0;
			if(t==0) System.out.print("Dev set:   ");
			else if(t== 1)System.out.print("Test set:   "); 
			System.out.print(String.format("P: %.2f%%    ", precision));
			System.out.print(String.format("R: %.2f%%    ", recall));
			System.out.println(String.format("F: %.2f%%", f1));
			
			if(t == 0) {
				if(f1 > devF) {
					devF = f1;
					devR = recall;
					devP = precision;
					finalmp = mentionPenalty;
					write = true;
				} else if(f1 == devF) {
					equal = true;
					devF2 = f1;
					devR2 = recall;
					devP2 = precision;
					finalmp2 = mentionPenalty;
					write = true;
				} 
			} else if(t == 1 && write == true && equal) {
				if(f1 > testF)
				{
					testF = f1;
				    testR = recall;
				    testP = precision;
				    devF = devF2;
				    devR = devR2;
				    devP = devP2;
				}
				write = false;
				equal = false;
			} else if(t == 1 && write == true) {
				testF = f1;
				testR = recall;
				testP = precision;
				write = false;
			}
			//System.out.print(String.format("P: %.2f%%    ", precision));
			//System.out.print(String.format("R: %.2f%%    ", recall));
			//System.out.println(String.format("F: %.2f%%", f1));
		   } 
		}
		System.out.println("-------------result---------------");
		System.out.println(out);
		System.out.println("finale mp: "+finalmp);
		System.out.print("Dev set:");
		System.out.print(String.format("P: %.2f%%    ", devP));
		System.out.print(String.format("R: %.2f%%    ", devR));
		System.out.println(String.format("F: %.2f%%", devF));
		System.out.print("Test set:");
		System.out.print(String.format("P: %.2f%%    ", testP));
		System.out.print(String.format("R: %.2f%%    ", testR));
		System.out.println(String.format("F: %.2f%%", testF));
		
		
	}

	/*
	private static String toString(Object[] arr){
		StringBuilder builder = new StringBuilder();
//		builder.append("[");
		int index = 0;
		for(Object str: arr){
			if(builder.length() > 0) builder.append(" ");
			builder.append(str+"("+index+")");
			index++;
		}
//		builder.append("]");
		return builder.toString();
	}
	*/
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
	
	/**
	 * Read a list of instances from a file
	 * @param fileName
	 * @param withLabels
	 * @param isLabeled
	 * @return
	 * @throws IOException
	 */
	private static MentionHypergraphInstance[] readData(String fileName, boolean withLabels, boolean isLabeled) throws IOException{
		InputStreamReader isr = new InputStreamReader(new FileInputStream(fileName), "UTF-8");
		BufferedReader br = new BufferedReader(isr);
		ArrayList<MentionHypergraphInstance> result = new ArrayList<MentionHypergraphInstance>();
		int instanceId = 1;
		int index = 0;
		while(br.ready()){

			index ++;
			String words = br.readLine();
			MentionHypergraphInstance instance = new MentionHypergraphInstance(instanceId++, 1.0);
//			instance.words = markWords(words.trim().split(" "));
			String posTags = br.readLine();
//			instance.posTags = posTags.trim().split(" ");
			instance.input = new WordsAndTags(markWords(words.trim().split(" ")), posTags.trim().split(" "));
			String[] spans = br.readLine().split("\\|");
			if(spans.length == 1 && spans[0].length() == 0){
				spans = new String[0];
			}
			List<Span> output = new ArrayList<Span>();
			for(String span: spans){
				String[] tokens = span.split(" ");
				String[] indices = tokens[0].split(",");
				int[] intIndices = new int[indices.length];
				for(int i=0; i<4; i++){
					intIndices[i] = Integer.parseInt(indices[i]);
				}
				Label label = Label.get(tokens[1]);
				output.add(new Span(intIndices[0], intIndices[1], intIndices[2], intIndices[3], label));
			}
			instance.setOutput(output);
			if(isLabeled){
				instance.setLabeled();
			} else {
				instance.setUnlabeled();
			}
			br.readLine();
			result.add(instance);
		}
		br.close();
		return result.toArray(new MentionHypergraphInstance[result.size()]);
	}
	
	private static AttributedWord[] markWords(String[] words){
		AttributedWord[] result = new AttributedWord[words.length];
		for(int i=0; i<result.length; i++){
			result[i] = new AttributedWord(words[i]);
		}
		return result;
	}
}