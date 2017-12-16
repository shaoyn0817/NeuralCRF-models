package org.statnlp.example;

import static org.statnlp.commons.Utils.print;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

import org.statnlp.InitWeightOptimizerFactory;
import org.statnlp.commons.ml.opt.OptimizerFactory;
import org.statnlp.commons.types.Instance;
import org.statnlp.commons.types.Label;
import org.statnlp.commons.types.LinearInstance;
import org.statnlp.example.linear_crf.LinearCRFFeatureManager;
import org.statnlp.example.linear_crf.LinearCRFNetworkCompiler;
import org.statnlp.example.linear_crf.LinearCRFViewer;
import org.statnlp.hypergraph.DiscriminativeNetworkModel;
import org.statnlp.hypergraph.GlobalNetworkParam;
import org.statnlp.hypergraph.NetworkConfig;
import org.statnlp.hypergraph.NetworkModel;
import org.statnlp.hypergraph.StringIndex;
import org.statnlp.hypergraph.NetworkConfig.ModelType;
import org.statnlp.hypergraph.NetworkConfig.StoppingCriteria;
import org.statnlp.util.GenericPipeline;

import gnu.trove.map.hash.TIntIntHashMap;
import gnu.trove.map.hash.TIntObjectHashMap;
import gnu.trove.set.TIntSet;

public class LinearCRFMain {
	
	public static void main(String args[]) throws IOException, InterruptedException{
		runPipeline(args);
//		runOldPipeline(args);
	}
	
	public static void runPipeline(String[] args){
		GenericPipeline pipeline = new GenericPipeline()
				.withTrainPath("data/train.data")
				.withTestPath("data/test.data")
				.withModelPath("test.model")
				.withLogPath("test.log")
				.withL2(0.01)
				.withWeightInit(0.0)
				.withModelType(ModelType.CRF)
//				.withUseBatchTraining(true)
//				.withBatchSize(1)
				.withStoppingCriteria(StoppingCriteria.SMALL_RELATIVE_CHANGE)
				.withMaxIter(1000)
				.withEvaluateEvery(0)
				.withWriteModelAsText(true)
				.withEvaluateCallback(LinearCRFMain::evaluate)
				.withPredictTopK(10)
				.addTask("train")
				.addTasks("test", "evaluate")
//				.addTask("visualize")
				;
		pipeline.execute();
		return;
	}
	
	public static void evaluate(Instance[] instances){
		int countCorrect = 0;
		int countGold = 0;
		int count = 0;
		for(Instance instance: instances){
			LinearInstance<?> inst = (LinearInstance<?>)instance;
			countCorrect += inst.countNumCorrectlyPredicted();
			countGold += inst.size();
			if(inst.size() > 5 && count < 5){
				for(String[] input: inst.input){
					System.out.print(input[0]+" ");
				}
				System.out.println();
				System.out.println(inst.output);
				for(int i=0; i<10; i++){
					System.out.println(inst.getTopKPredictions().get(i));
				}
				count += 1;
			}
		}
		System.out.println("Correct/Gold: "+countCorrect+"/"+countGold);
		System.out.println(String.format("Accuracy: %.3f%%", 100.0*countCorrect/countGold));
	}
	
	public static void runOldPipeline(String[] args) throws IOException, InterruptedException{
		String trainPath = System.getProperty("trainPath", "data/train.data");
		String testPath = System.getProperty("testPath", "data/test.data");
		
		String resultPath = System.getProperty("resultPath", "lcrf.result");
		String modelPath = System.getProperty("modelPath", "lcrf.model");
		String logPath = System.getProperty("logPath", "lcrf.log");
		
		boolean writeModelText = Boolean.parseBoolean(System.getProperty("writeModelText", "false"));

		NetworkConfig.TRAIN_MODE_IS_GENERATIVE = Boolean.parseBoolean(System.getProperty("generativeTraining", "false"));
		NetworkConfig.PARALLEL_FEATURE_EXTRACTION = Boolean.parseBoolean(System.getProperty("parallelTouch", "true"));
		NetworkConfig.BUILD_FEATURES_FROM_LABELED_ONLY = Boolean.parseBoolean(System.getProperty("featuresFromLabeledOnly", "false"));
		NetworkConfig.CACHE_FEATURES_DURING_TRAINING = Boolean.parseBoolean(System.getProperty("cacheFeatures", "true"));
		NetworkConfig.L2_REGULARIZATION_CONSTANT = Double.parseDouble(System.getProperty("l2", "0.01")); //0.01
		NetworkConfig.NUM_THREADS = Integer.parseInt(System.getProperty("numThreads", "4"));
		
		NetworkConfig.MODEL_TYPE = ModelType.valueOf(System.getProperty("modelType", "CRF")); // The model to be used: CRF, SSVM, or SOFTMAX_MARGIN
		NetworkConfig.USE_BATCH_TRAINING = Boolean.parseBoolean(System.getProperty("useBatchTraining", "false")); // To use or not to use mini-batches in gradient descent optimizer
		NetworkConfig.BATCH_SIZE = Integer.parseInt(System.getProperty("batchSize", "1000"));  // The mini-batch size (if USE_BATCH_SGD = true)
		NetworkConfig.MARGIN = Double.parseDouble(System.getProperty("svmMargin", "1.0"));
		
		// Set weight to not random to make meaningful comparison between sequential and parallel touch
		NetworkConfig.RANDOM_INIT_WEIGHT = false;
		NetworkConfig.FEATURE_INIT_WEIGHT = 0.0;
		NetworkConfig.USE_NEURAL_FEATURES = false;
		NetworkConfig.REGULARIZE_NEURAL_FEATURES = true;
		NetworkConfig.OPTIMIZE_NEURAL = false;
		NetworkConfig.AVOID_DUPLICATE_FEATURES = false;
		String weightInitFile = null;
		
		int numIterations = Integer.parseInt(System.getProperty("numIter", "1000"));
		
		int argIndex = 0;
		boolean shouldStop = false;
		while(argIndex < args.length && !shouldStop){
			switch(args[argIndex].substring(1)){
			case "trainPath":
				trainPath = args[argIndex+1];
				argIndex += 2;
				break;
			case "testPath":
				testPath = args[argIndex+1];
				argIndex += 2;
				break;
			case "modelPath":
				modelPath = args[argIndex+1];
				argIndex += 2;
				break;
			case "resultPath":
				resultPath = args[argIndex+1];
				argIndex += 2;
				break;
			case "logPath":
				logPath = args[argIndex+1];
				argIndex += 2;
				break;
			case "writeModelText":
				writeModelText = true;
				argIndex += 1;
				break;
			case "parallelTouch":
				NetworkConfig.PARALLEL_FEATURE_EXTRACTION = true;
				argIndex += 1;
				break;
			case "featuresFromLabeledOnly":
				NetworkConfig.BUILD_FEATURES_FROM_LABELED_ONLY = true;
				argIndex += 1;
				break;
			case "noCacheFeatures":
				NetworkConfig.CACHE_FEATURES_DURING_TRAINING = false;
				argIndex += 1;
				break;
			case "trySaveMemory":
				NetworkConfig.AVOID_DUPLICATE_FEATURES = true;
				argIndex += 1;
				break;
			case "l2":
				NetworkConfig.L2_REGULARIZATION_CONSTANT = Double.parseDouble(args[argIndex+1]);
				argIndex += 2;
				break;
			case "numThreads":
				NetworkConfig.NUM_THREADS = Integer.parseInt(args[argIndex+1]);
				argIndex += 2;
				break;
			case "modelType":
				NetworkConfig.MODEL_TYPE = ModelType.valueOf(args[argIndex+1].toUpperCase());
				argIndex += 2;
				break;
			case "useBatchSGD":
				NetworkConfig.USE_BATCH_TRAINING = true;
				argIndex += 1;
				break;
			case "batchSize":
				NetworkConfig.BATCH_SIZE = Integer.parseInt(args[argIndex+1]);
				argIndex += 2;
				break;
			case "margin":
				NetworkConfig.MARGIN = Double.parseDouble(args[argIndex+1]);
				argIndex += 2;
				break;
			case "weightInit":
				if(args[argIndex+1].equals("random")){
					NetworkConfig.RANDOM_INIT_WEIGHT = true;
				} else if (args[argIndex+1].equals("file")){
					weightInitFile = args[argIndex+2];
					argIndex += 1;
				} else {
					NetworkConfig.RANDOM_INIT_WEIGHT = false;
					NetworkConfig.FEATURE_INIT_WEIGHT = Double.parseDouble(args[argIndex+1]);
				}
				argIndex += 2;
				break;
			case "numIter":
				numIterations = Integer.parseInt(args[argIndex+1]);
				argIndex += 2;
				break;
			case "-":
				shouldStop = true;
				argIndex += 1;
				break;
			case "h":
			case "help":
				printHelp();
				System.exit(0);
			default:
				throw new IllegalArgumentException("Unrecognized argument: "+args[argIndex]);
			}
		}
		
		PrintStream outstream = new PrintStream(logPath);
		
		OptimizerFactory optimizerFactory;
		if(NetworkConfig.MODEL_TYPE.USE_SOFTMAX && !(NetworkConfig.USE_NEURAL_FEATURES && !NetworkConfig.OPTIMIZE_NEURAL)){
			optimizerFactory = OptimizerFactory.getLBFGSFactory();
		} else {
			optimizerFactory = OptimizerFactory.getGradientDescentFactoryUsingAdaMThenStop();
		}
		if(weightInitFile != null){
			HashMap<String, HashMap<String, HashMap<String, Double>>> featureWeightMap = new HashMap<String, HashMap<String, HashMap<String, Double>>>();
			Scanner reader = new Scanner(new File(weightInitFile));
			HashMap<String, HashMap<String, Double>> outputToInput = null;
			HashMap<String, Double> inputToWeight = null;
			String input;
			double weight = 0.0;
			while(reader.hasNextLine()){
				String line = reader.nextLine();
				if(line.startsWith("\t\t")){
					line = line.substring(2);
					int lastSpace = line.lastIndexOf(" ");
					if(lastSpace == -1){
						input = "";
						weight = Double.parseDouble(line);
					} else {
						input = line.substring(0, lastSpace);
						weight = Double.parseDouble(line.substring(lastSpace+1));
					}
					inputToWeight.put(input, weight);
				} else if (line.startsWith("\t")){
					inputToWeight = new HashMap<String, Double>();
					outputToInput.put(line.trim(), inputToWeight);
				} else {
					outputToInput = new HashMap<String, HashMap<String, Double>>();
					featureWeightMap.put(line.trim(), outputToInput);
				}
			}
			reader.close();
			optimizerFactory = new InitWeightOptimizerFactory(featureWeightMap, optimizerFactory);
		}

		String[] argsToFeatureManager = new String[args.length-argIndex];
		for(int i=argIndex; i<args.length; i++){
			argsToFeatureManager[i-argIndex] = args[i];
		}
		GlobalNetworkParam param = new GlobalNetworkParam(optimizerFactory);

		int numTrain = -1;
		LinearInstance<Label>[] trainInstances = readCoNLLData(param, trainPath, true, true, numTrain);
		int size = trainInstances.length;
		System.err.println("Read.."+size+" instances from "+trainPath);
		
		LinearCRFNetworkCompiler compiler = new LinearCRFNetworkCompiler(LABELS.values());
		LinearCRFFeatureManager fm = new LinearCRFFeatureManager(param, argsToFeatureManager);
		
		NetworkModel model = DiscriminativeNetworkModel.create(fm, compiler, outstream);
		model.visualize(LinearCRFViewer.class, trainInstances);
		
		model.train(trainInstances, numIterations);
		
		writeModelText = true;
		if(writeModelText){
			PrintStream modelTextWriter = new PrintStream(modelPath+".txt");
			modelTextWriter.println("Model path: "+modelPath);
			modelTextWriter.println("Train path: "+trainPath);
			modelTextWriter.println("Test path: "+testPath);
			modelTextWriter.println("#Threads: "+NetworkConfig.NUM_THREADS);
			modelTextWriter.println("L2 param: "+NetworkConfig.L2_REGULARIZATION_CONSTANT);
			modelTextWriter.println("Weight init: "+0.0);
			modelTextWriter.println("objtol: "+NetworkConfig.OBJTOL);
			modelTextWriter.println("Max iter: "+numIterations);
			modelTextWriter.println();
			modelTextWriter.println("Labels:");
			List<Label> labelsUsed = new ArrayList<Label>(compiler._labels.values());
			Collections.sort(labelsUsed);
			modelTextWriter.println(labelsUsed);
			GlobalNetworkParam paramG = fm.getParam_G();
			modelTextWriter.println("Num features: "+paramG.countFeatures());
			modelTextWriter.println("Features:");
			TIntObjectHashMap<TIntObjectHashMap<TIntIntHashMap>> featureIntMap = paramG.getFeatureIntMap();
			StringIndex stringIndex = paramG.getStringIndex();
			stringIndex.buildReverseIndex();
			for(String featureType: sorted(stringIndex, featureIntMap.keySet())){
				modelTextWriter.println(featureType);
				TIntObjectHashMap<TIntIntHashMap> outputInputMap = featureIntMap.get(stringIndex.get(featureType));
				for(String output: sorted(stringIndex, outputInputMap.keySet())){
					modelTextWriter.println("\t"+output);
					TIntIntHashMap inputMap = outputInputMap.get(stringIndex.get(output));
					for(String input: sorted(stringIndex, inputMap.keySet())){
						int featureId = inputMap.get(stringIndex.get(input));
						modelTextWriter.printf("\t\t%s %d %.17f\n", input, featureId, fm.getParam_G().getWeight(featureId));
					}
				}
			}
			stringIndex.removeReverseIndex();
			modelTextWriter.close();
		}
		
		LinearInstance<Label>[] testInstances = readCoNLLData(param, testPath, true, false);
//		testInstances = Arrays.copyOf(testInstances, 1);
		int k = 8;
		Instance[] predictions = model.decode(testInstances, k);
		
		PrintStream[] outstreams = new PrintStream[]{outstream, System.out};
		PrintStream resultStream = new PrintStream(resultPath);
		
		int corr = 0;
		int total = 0;
		int count = 0;
		for(Instance ins: predictions){
			@SuppressWarnings("unchecked")
			LinearInstance<Label> instance = (LinearInstance<Label>)ins;
			List<Label> goldLabel = instance.getOutput();
			List<Label> actualLabel = instance.getPrediction();
			List<List<Label>> topKPredictions = instance.getTopKPredictions();
			ArrayList<String[]> words = (ArrayList<String[]>)instance.getInput();
			for(int i=0; i<goldLabel.size(); i++){
				if(goldLabel.get(i).equals(actualLabel.get(i))){
					corr++;
				}
				total++;
				if(count < 3){
					print(String.format("%15s %6s %6s %6s%s", words.get(i)[0], goldLabel.get(i), actualLabel.get(i), topKPredictions.get(k-1).get(i), actualLabel.get(i) == topKPredictions.get(k-1).get(i) ? "" : " DIFFERENT"), outstreams);
				}
				resultStream.println(words.get(i)[0]+" "+goldLabel.get(i)+" "+actualLabel.get(i));
			}
			count++;
			if(count < 3){
				print("", outstreams);
			}
		}
		resultStream.close();
		print(String.format("Correct/Total: %d/%d", corr, total), outstreams);
		print(String.format("Accuracy: %.2f%%", 100.0*corr/total), outstreams);
		outstream.close();
	}
	
	@SuppressWarnings("unchecked")
	private static LinearInstance<Label>[] readCoNLLData(GlobalNetworkParam param, String fileName, boolean withLabels, boolean isLabeled, int number) throws IOException{
		InputStreamReader isr = new InputStreamReader(new FileInputStream(fileName), "UTF-8");
		BufferedReader br = new BufferedReader(isr);
		ArrayList<LinearInstance<Label>> result = new ArrayList<LinearInstance<Label>>();
		ArrayList<String[]> words = null;
		ArrayList<Label> labels = null;
		int instanceId = 1;
		while(br.ready()){
			if(words == null){
				words = new ArrayList<String[]>();
			}
			if(withLabels && labels == null){
				labels = new ArrayList<Label>();
			}
			String line = br.readLine().trim();
			if(line.length() == 0){
				LinearInstance<Label> instance = new LinearInstance<Label>(instanceId, 1, words, labels);
				if(isLabeled){
					instance.setLabeled(); // Important!
				} else {
					instance.setUnlabeled();
				}
				instanceId++;
				result.add(instance);
				if(result.size()==number) break;
				words = null;
				labels = null;
			} else {
				int lastSpace = line.lastIndexOf(" ");
				String[] features = line.substring(0, lastSpace).split(" ");
				words.add(features);
				if(withLabels){
					Label label = getLabel(line.substring(lastSpace+1));
					labels.add(label);
				}
			}
		}
		br.close();
		return result.toArray(new LinearInstance[result.size()]);
	}
	
	private static LinearInstance<Label>[]  readCoNLLData(GlobalNetworkParam param, String fileName, boolean withLabels, boolean isLabeled) throws IOException{
		return readCoNLLData(param, fileName, withLabels, isLabeled, -1);
	}
	
	private static List<String> sorted(StringIndex stringIndex, TIntSet coll){
		List<String> result = new ArrayList<String>(coll.size());
		for(int key: coll.toArray()){
			result.add(stringIndex.get(key));
		}
		Collections.sort(result);
		return result;
	}

	public static final Map<String, Label> LABELS = new HashMap<String, Label>();
	public static final Map<Integer, Label> LABELS_INDEX = new HashMap<Integer, Label>();
	
	public static Label getLabel(String form){
		if(!LABELS.containsKey(form)){
			Label label = new Label(form, LABELS.size());
			LABELS.put(form, label);
			LABELS_INDEX.put(label.getId(), label);
		}
		return LABELS.get(form);
	}
	
	public static Label getLabel(int id){
		return LABELS_INDEX.get(id);
	}
	
	public static void reset(){
		LABELS.clear();
		LABELS_INDEX.clear();
	}

	private static void printHelp(){
		System.out.println("Options:\n"
				+ "-modelPath <modelPath>\n"
				+ "\tSerialize model to <modelPath>\n"
				
				+ "-writeModelText\n"
				+ "\tAlso write the model in text version for debugging purpose\n"
				
				+ "-trainPath <trainPath>\n"
				+ "\tTake training file from <trainPath>. If not specified, no training will be performed\n"
				
				+ "-testPath <testPath>\n"
				+ "\tTake test file from <testPath>. If not specified, no testing will be performed\n"
				
				+ "-logPath <logPath>\n"
				+ "\tPrint log information to the specified file\n"
				
				+ "-resultPath <resultPath>\n"
				+ "\tPrint the result to <resultPath>\n"

				+ "-parallelTouch\n"
				+ "\tWhether the feature extraction process should be parallel.\n"
				
				+ "-featuresFromLabeledOnly\n"
				+ "\tWhether to define features only from those appearing in training data.\n"
				
				+ "-noCacheFeatures\n"
				+ "\tWhether to disable feature caching. It is recommended to keep caching enabled,\n"
				+ "\tsince training will be quite slow otherwise."
				
				+ "-trySaveMemory\n"
				+ "\tWhether an attempt to reduce memory usage should be done.\n"
				+ "\tThe amount of saving depends on how the feature arrays were created.\n"
				
				+ "-l2 <l2_value>\n"
				+ "\tThe l2 regularization parameter.\n"
				
				+ "-numThreads <n>\n"
				+ "\tThe number of threads to train and test the model.\n"
				
				+ "-modelType (STRUCTURED_PERCEPTRON|CRF|SSVM|SOFTMAX_MARGIN)\n"
				+ "\tThe training algorithm, must be one of these:\n"
				+ "\t- Structured perceptron: mistake-driven, best combined with batch training.\n"
				+ "\t- CRF: Standard log-likelihood training.\n"
				+ "\t- SSVM: Max-margin based training.\n"
				+ "\t- Softmax Margin: A cost-augmented log-likelihood training.\n"
				
				+ "-useBatchSGD\n"
				+ "\tWhether to use batch training.\n"
				
				+ "-batchSize <n>\n"
				+ "\tThe batch size to be used if -useBatchSGD is used.\n"
				
				+ "-margin <value>\n"
				+ "\tThe margin hyperparameter if SSVM or Softmax-Margin training algorithm is used.\n"
				+ "\tThe default value is 1.0.\n"
				
				+ "-weightInit <path_to_weights>\n"
				+ "\tWhether to initialize the weights based on a file.\n"
				+ "\tThe file should specify the feature weights in the following format:\n"
				+ "\t-Lines with no prefix: Defines the feature types for the subsequent lines.\n"
				+ "\t-Lines with '\t' (tab) prefix: Defines the output feature for the subsequent lines.\n"
				+ "\t-Lines with '\t\t' (double tab) prefix: Defines the input feature and the feature weight.\n"
				
				+ "-numIter <n>\n"
				+ "\tDefines the maximum number of iteration the training algorithm should run.\n"
				+ "\tThe training might finish earlier than this number, but it will never exceed it.\n"
				);
	}
}
