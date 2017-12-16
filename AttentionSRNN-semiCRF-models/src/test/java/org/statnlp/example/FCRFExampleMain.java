package org.statnlp.example;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.List;

import org.statnlp.commons.ml.opt.GradientDescentOptimizerFactory;
import org.statnlp.commons.ml.opt.OptimizerFactory;
import org.statnlp.commons.types.Instance;
import org.statnlp.example.fcrf.Chunk;
import org.statnlp.example.fcrf.FCRFConfig;
import org.statnlp.example.fcrf.FCRFEval;
import org.statnlp.example.fcrf.FCRFFeatureManager;
import org.statnlp.example.fcrf.FCRFInstance;
import org.statnlp.example.fcrf.FCRFNetworkCompiler;
import org.statnlp.example.fcrf.FCRFReader;
import org.statnlp.example.fcrf.Tag;
import org.statnlp.example.fcrf.FCRFConfig.TASK;
import org.statnlp.hypergraph.DiscriminativeNetworkModel;
import org.statnlp.hypergraph.FeatureManager;
import org.statnlp.hypergraph.GlobalNetworkParam;
import org.statnlp.hypergraph.NetworkConfig;
import org.statnlp.hypergraph.NetworkModel;
import org.statnlp.hypergraph.NetworkConfig.InferenceType;

public class FCRFExampleMain {

	public static int trainNumber = -100;
	public static int testNumber = -100;
	public static int numIteration = -100;
	public static int numThreads = -100;
	public static String trainFile = "";
	public static String testFile = "";
	/**The output file of Chunking result **/
	public static String nerOut;
	/**The output file of POS tagging result **/
	public static String posOut;
	public static String neural_config = "config/fcrfneural.config";
	public static OptimizerFactory optimizer = OptimizerFactory.getLBFGSFactory();
	public static boolean useJointFeatures = false;
	public static TASK task = TASK.JOINT;
	public static boolean IOBESencoding = true;
	public static boolean npchunking = true;
	/** Cascaded CRF option. **/
	public static boolean cascade = false;
	public static int windowSize = 5;
	public static String modelFile = "data/conll2000/model";
	/** The option to save model **/
	public static boolean saveModel = false;
	/** The option to use existing model **/
	public static boolean useExistingModel = false;
	public static int randomSeed = 1234;
	public static boolean removeChunkNeural = false;
	public static boolean removePOSNeural = false;
	public static boolean removeJointNeural = false;
	public static boolean parallelFeatureExtraction = true;
	public static String dataset = "conll2000";
	public static double l2val = 0.01;
	public static boolean isWndowsOS = false;
	public static String optionalOutputSuffix = "";
	public static int hiddenSize = 10;
	
	public static void main(String[] args) throws IOException, InterruptedException, ClassNotFoundException{
		
		trainNumber = 80;
		testNumber = 2;
		numThreads = 5;
		numIteration = 200;
		
		
		
		processArgs(args);
		
		FCRFConfig config = new FCRFConfig(dataset, l2val, isWndowsOS);
		trainFile = config.train;
		testFile = config.test;
		nerOut = config.nerOut+optionalOutputSuffix;
		posOut = config.posOut+optionalOutputSuffix;
		
		List<FCRFInstance> trainInstances = null;
		List<FCRFInstance> testInstances = null;
		/***********DEBUG*****************/
		trainFile = "data/"+dataset+"/train.txt";
//		trainNumber = 2;
		testFile = "data/"+dataset+"/test.txt";;
//		testNumber = 2;
//		numIteration = 1000;   
//		testFile = trainFile;
//		NetworkConfig.MAX_MF_UPDATES = 6;
//		useJointFeatures = true;
//		task = TASK.JOINT;
		IOBESencoding = true;
		if (dataset.equals("conll2003") || dataset.equals("semeval10t1"))
			IOBESencoding = false;
//		saveModel = false;
		//modelFile = "data/conll2000/model";
//		useExistingModel = false;
		npchunking = true;
		if (dataset.equals("conll2003") || dataset.equals("semeval10t1"))
			npchunking = false;
		config.l2val = 0.01;
		NetworkConfig.AVOID_DUPLICATE_FEATURES = true;
		NetworkConfig.RANDOM_INIT_FEATURE_SEED = randomSeed;
//		cascade = true;
//		testFile = "data/conll2000/NP_chunk_final_prediction.txt";
//		npchunking = true;
//		testFile = "data/conll2000/POS_final_prediction.txt";
//		optimizer = OptimizerFactory.getGradientDescentFactoryUsingAdaM(0.0001, 0.9, 0.999, 10e-8);
		/***************************/
		
		if (cascade) {
			if (task == TASK.TAGGING) {
				testFile = config.nerOut;
				posOut = config.posPipeOut;
			}
			else if (task == TASK.CHUNKING) {
				testFile = config.posOut;
				nerOut = config.nerPipeOut;
			}
		}
		
		System.err.println("[Info] trainingFile: "+trainFile);
		System.err.println("[Info] testFile: "+testFile);
		System.err.println("[Info] nerOut: "+nerOut);
		System.err.println("[Info] posOut: "+posOut);
		System.err.println("[Info] task: "+task.toString());
		System.err.println("[Info] #max-mf: " + NetworkConfig.MAX_MF_UPDATES);
		
		trainInstances = FCRFReader.readCONLLData(trainFile, true, trainNumber, npchunking, IOBESencoding, task);
		boolean iobesOnTest = task == TASK.TAGGING && cascade ? true : false;
		testInstances = FCRFReader.readCONLLData(testFile, false, testNumber, npchunking, iobesOnTest, task, cascade);
		
//		trainInstances = FCRFReader.readGRMMData("data/conll2000/conll2000.train1k.txt", true, -1);
//		testInstances = FCRFReader.readGRMMData("data/conll2000/conll2000.test1k.txt", false, -1);
		
		
		Chunk.lock();
		Tag.lock();
		
		
		System.err.println("chunk size:"+Chunk.CHUNKS_INDEX.toString());
		System.err.println("tag size:"+Tag.TAGS.size());
		System.err.println("tag size:"+Tag.TAGS_INDEX.toString());
		
		NetworkConfig.TRAIN_MODE_IS_GENERATIVE = false;
		NetworkConfig.CACHE_FEATURES_DURING_TRAINING = true;
		NetworkConfig.L2_REGULARIZATION_CONSTANT = config.l2val;
		NetworkConfig.NUM_THREADS = numThreads;
		NetworkConfig.PARALLEL_FEATURE_EXTRACTION = parallelFeatureExtraction;
		NetworkConfig.BUILD_FEATURES_FROM_LABELED_ONLY = false;
		NetworkConfig.NUM_STRUCTS = 2;
		NetworkConfig.INFERENCE = task == TASK.JOINT ? InferenceType.MEAN_FIELD : InferenceType.FORWARD_BACKWARD;
		
		/***Neural network Configuration**/
//		NetworkConfig.USE_NEURAL_FEATURES = false; 
//		NeuralConfig.HIDDEN_SIZE = 200;
		/****/
		
		// TODO: Update the GlobalNetworkParam with the respective FeatureValueProvider from NeuralNetwork
		GlobalNetworkParam param_g = null; 
		if(useExistingModel){
			ObjectInputStream in = new ObjectInputStream(new FileInputStream(modelFile));
			param_g = (GlobalNetworkParam)in.readObject();
			in.close();
		}else{
			param_g = new GlobalNetworkParam(optimizer);
		}
		
		FeatureManager fa = null;
		fa = new FCRFFeatureManager(param_g, useJointFeatures, cascade, task, windowSize, IOBESencoding,
									removeChunkNeural, removePOSNeural, removeJointNeural);
//		fa = new GRMMFeatureManager(param_g, useJointFeatures);
		FCRFNetworkCompiler compiler = new FCRFNetworkCompiler(task, IOBESencoding);
		NetworkModel model = DiscriminativeNetworkModel.create(fa, compiler);
		FCRFInstance[] ecrfs = trainInstances.toArray(new FCRFInstance[trainInstances.size()]);
		
		if(!useExistingModel){
			System.out.println("Training Instance size: " + ecrfs.length);
			model.train(ecrfs, numIteration);
		}
		
		if(saveModel){
			ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(modelFile));
			out.writeObject(param_g);
			//out.writeObject(param_g.getNNCRFController());
			out.close();
		}
		
		Instance[] predictions = model.decode(testInstances.toArray(new FCRFInstance[testInstances.size()]));
		/**Evaluation part**/
		if (task == TASK.CHUNKING || task == TASK.JOINT) {
			FCRFEval.evalFscore(predictions, nerOut);
			FCRFEval.evalChunkAcc(predictions);
		}
		if (task == TASK.TAGGING || task == TASK.JOINT)
			FCRFEval.evalPOSAcc(predictions, posOut);
		if (task == TASK.JOINT)
			FCRFEval.evalJointAcc(predictions);
	}

	
	/**
	 * Process the input argument
	 * @param args: arguments
	 */
	public static void processArgs(String[] args){
		if(args.length>1 && (args[0].equals("-h") || args[0].equals("help") || args[0].equals("-help") )){
			System.err.println("Factorial CRFs for Joint NP Chunking and POS tagging ");
			System.err.println("\t usage: java -jar fcrf.jar -trainNum -1 -testNum -1 -thread 5 -iter 100 -pipe true");
			System.err.println("\t put numTrainInsts/numTestInsts = -1 if you want to use all the training/testing instances");
			System.exit(0);
		}else{
			for(int i=0;i<args.length;i=i+2){
				switch(args[i]){
					case "-trainNum": trainNumber = Integer.valueOf(args[i+1]); break;
					case "-testNum": testNumber = Integer.valueOf(args[i+1]); break;
					case "-iter": numIteration = Integer.valueOf(args[i+1]); break;
					case "-thread": numThreads = Integer.valueOf(args[i+1]); break;
					case "-seed": randomSeed = Integer.valueOf(args[i+1]); break;
					case "-testFile": testFile = args[i+1]; break;
					case "-reg": l2val = Double.valueOf(args[i+1]); break;
					case "-windows": isWndowsOS = args[i+1].equals("true")? true:false; break;
					case "-mfround":NetworkConfig.MAX_MF_UPDATES = Integer.valueOf(args[i+1]);
									useJointFeatures = true;
									if(NetworkConfig.MAX_MF_UPDATES == 0) useJointFeatures = false;
									break;
					case "-task": 
						if(args[i+1].equals("chunk"))  task = TASK.CHUNKING;
						else if (args[i+1].equals("tag")) task  = TASK.TAGGING;
						else if (args[i+1].equals("joint"))  task  = TASK.JOINT;
						else throw new RuntimeException("Unknown task:"+args[i+1]+"?"); break;
					case "-iobes": 		IOBESencoding = args[i+1].equals("true")? true:false; break;
					case "-npchunking": npchunking = args[i+1].equals("true")? true:false; break;
					case "-optim": 
						if(args[i+1].equals("lbfgs"))
							optimizer = GradientDescentOptimizerFactory.getLBFGSFactory();
						else if(args[i+1].equals("adagrad")) {
							//specify the learning rate also 
							if(args[i+2].startsWith("-")) {System.err.println("Please specify the learning rate for adagrad.");System.exit(0);}
							optimizer = GradientDescentOptimizerFactory.getGradientDescentFactoryUsingAdaGrad(Double.valueOf(args[i+2]));
							i=i+1;
						}else if(args[i+1].equals("adam")){
							if(args[i+2].startsWith("-")) {System.err.println("Please specify the learning rate for adam.");System.exit(0);}
							//default should be 1e-3
							optimizer = GradientDescentOptimizerFactory.getGradientDescentFactoryUsingAdaM(Double.valueOf(args[i+2]), 0.9, 0.999, 10e-8);
							i=i+1;
						}else{
							System.err.println("No optimizer named: "+args[i+1]+"found..");System.exit(0);
						}
						break;
					case "-cascade": cascade = args[i+1].equals("true")? true:false; break;
					case "-wsize": 	 windowSize = Integer.valueOf(args[i+1]); break; //default: 5. the window size of neural feature.
					case "-nerout":  nerOut = args[i+1]; break; //default: name is output/nerout
					case "-posout":  posOut = args[i+1]; break; //default: name is output/pos_out;
					case "-mode": 	if (args[i+1].equals("train")){
										//train also test the file
										useExistingModel = false;
										saveModel = true;
										modelFile = args[i+2];
								  	}else if (args[i+1].equals("test")){
								  		useExistingModel = true;
										saveModel = false;
										modelFile = args[i+2];
								  	}else if (args[i+1].equals("train-only")){
								  		useExistingModel = false;
										saveModel = false;
										modelFile = args[i+2];
								  	}else{
										System.err.println("Unknown mode: "+args[i+1]+" found..");System.exit(0);
									}
									i = i + 1;
								break;
					case "-neural": if(args[i+1].equals("true")){ 
						NetworkConfig.USE_NEURAL_FEATURES = true;
						NetworkConfig.OPTIMIZE_NEURAL = true;  //false: optimize in neural network
						NetworkConfig.IS_INDEXED_NEURAL_FEATURES = false; //only used when using the senna embedding.
						NetworkConfig.REGULARIZE_NEURAL_FEATURES = true; //true means regularize in the crf part
					}
					break;
					case "-rmchunk" : removeChunkNeural = args[i+1].equals("true")? true:false; break;
					case "-rmpos" : removePOSNeural = args[i+1].equals("true")? true:false; break;
					case "-rmjoint" : removeJointNeural = args[i+1].equals("true")? true:false; break;
					case "-multitouch": parallelFeatureExtraction = args[i+1].equals("true")? true:false; break;
					case "-dataset": dataset = args[i+1]; break;
					case "-optsuffix" : optionalOutputSuffix = args[i+1]; break;
					case "-hidden": hiddenSize = Integer.valueOf(args[i+1]); break;
					default: System.err.println("Invalid arguments :"+args[i]+", please check usage."); System.exit(0);
				}
			}
			System.err.println("[Info] trainNum: "+trainNumber);
			System.err.println("[Info] testNum: "+testNumber);
			System.err.println("[Info] numIter: "+numIteration);
			System.err.println("[Info] numThreads: "+numThreads);
			System.err.println("[Info] Regularization Parameter: "+ l2val);	
		}
	}
	
	
}
