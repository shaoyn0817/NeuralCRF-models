package semi_crf;


import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.function.Function;

import org.statnlp.commons.ml.opt.OptimizerFactory;
import org.statnlp.commons.types.Instance;
import org.statnlp.hypergraph.DiscriminativeNetworkModel;
import org.statnlp.hypergraph.GlobalNetworkParam;
import org.statnlp.hypergraph.NetworkConfig;
import org.statnlp.hypergraph.NetworkModel;
import org.statnlp.hypergraph.decoding.Metric;
import org.statnlp.hypergraph.neural.AttentionSRNN;
import org.statnlp.hypergraph.neural.BidirectionalLSTM;
import org.statnlp.hypergraph.neural.GlobalNeuralNetworkParam;
import org.statnlp.hypergraph.neural.NeuralNetworkCore;

import scala.reflect.internal.Trees.This;

 


 
public class semi_main {
    public static String neuralType = "AttentionSRNN";
    public static void main(String []args) throws FileNotFoundException, Exception{
    	//experiment/data/traindata.txt
    	//experiment/data/testdata.txt
    	///home/ubuntu/traindata.txt
    	///home/ubuntu/testdata.txt
    	String trainPath = "data/con03/tt";
    	String devPath = "data/con03/tt";
		String testPath = "data/con03/tt";
		//String trainPath = "experiment/data/testdata.txt";
		//String testPath = "experiment/data/testdata.txt";
		
        int validbatch = 300;
		NetworkConfig.TRAIN_MODE_IS_GENERATIVE = Boolean.parseBoolean(System.getProperty("generativeTraining", "false"));
		NetworkConfig.PARALLEL_FEATURE_EXTRACTION = Boolean.parseBoolean(System.getProperty("parallelTouch", "true"));
		NetworkConfig.CACHE_FEATURES_DURING_TRAINING = Boolean.parseBoolean(System.getProperty("cacheFeatures", "true"));
		NetworkConfig.L2_REGULARIZATION_CONSTANT = Double.parseDouble(System.getProperty("l2", "0.01"));
		NetworkConfig.NUM_THREADS = Integer.parseInt(System.getProperty("numThreads", "8"));
        
		
		NetworkConfig.USE_BATCH_TRAINING = false; // To use or not to use mini-batches in gradient descent optimizer
		NetworkConfig.BATCH_SIZE = Integer.parseInt(System.getProperty("batchSize", "4")); 
		NetworkConfig.MARGIN = Double.parseDouble(System.getProperty("svmMargin", "1.0"));

		// Set weight to not random to make meaningful comparison between sequential and parallel touch
		NetworkConfig.RANDOM_INIT_WEIGHT = false;
		NetworkConfig.FEATURE_INIT_WEIGHT = 0.0;
	    NetworkConfig.USE_NEURAL_FEATURES = false;
		NetworkConfig.OS = "linux";
	    NetworkConfig.PRINT_BATCH_OBJECTIVE = true;
	    NetworkConfig.RANDOM_BATCH = true;
	    NetworkConfig.PRINT_BATCH_OBJECTIVE = true;
		int numIterations = Integer.parseInt(System.getProperty("numIter", "100"));
	    PrintStream outstream = null;//new PrintStream("experiment/model.txt");
		semiCRFInstance[] trainInstances = readCoNLLData_BIO(trainPath, true);
		semiCRFInstance[] devInstances = readCoNLLData_BIO(devPath, false);
		int size = trainInstances.length;
		System.err.println("Read.."+size+" instances from ");
		
		//OptimizerFactory optimizer = OptimizerFactory.getGradientDescentFactoryUsingGradientClipping(0.001, 4);
		OptimizerFactory optimizer = OptimizerFactory.getLBFGSFactory();
			
		System.out.println(Label.LABELS.size());
		List<NeuralNetworkCore> nets = new ArrayList<NeuralNetworkCore>();
		if(NetworkConfig.USE_NEURAL_FEATURES){
			if (neuralType.equals("AttentionSRNN")) {
				int hiddenSize = 100;
				String opt = "sgd";
				boolean bidirection = false;
				nets.add(new AttentionSRNN(hiddenSize, bidirection, opt, 0.001, 5, Label.LABELS.size(), 0, "glove"));
			} //else if (neuralType.equals("continuous")) {
				//nets.add(new ECRFContinuousFeatureValueProvider(50, Label.LABELS.size()));
			//}		
		} 
		
		
		GlobalNetworkParam gnp = new GlobalNetworkParam(optimizer, new GlobalNeuralNetworkParam(nets));
		semiCRFFeatureManager fm = new semiCRFFeatureManager(gnp);
		
		semiCRFNetworkCompiler compiler = new semiCRFNetworkCompiler();
		

		NetworkModel model = DiscriminativeNetworkModel.create(fm, compiler);

		
		 String tmpOut = "data/tmp_out.txt";
			
			Function<Instance[], Metric> evalFunc = new Function<Instance[], Metric>() {

				@Override
				public Metric apply(Instance[] t) {
					return ECRFEval.evalNER(t, tmpOut);
				}
				
			};

	     //model.train(trainInstances, numIterations, devInstances, evalFunc, 3);
		
		
		
		model.train(trainInstances, numIterations);
  

		semiCRFInstance[] testInstances = readCoNLLData_BIO(testPath, false);
		Instance predictions[] = model.decode(testInstances);

	    //segmentEvaluation e = new segmentEvaluation(predictions);
        segmentEvaluation.eval2(predictions);
       semiCRFInstance si= (semiCRFInstance)predictions[5]; 
       ArrayList<Span> out = si.getOutput();
       ArrayList<Span> pr = si.getPrediction();
        for(int i = 0; i < out.size(); i++){
	    System.out.print(out.get(i)._end+"|"+out.get(i)._label+"    ");
        }
	    System.out.println();
	    for(int i = 0; i < pr.size(); i++){
		    System.out.print(pr.get(i)._end+"|"+pr.get(i)._label+"    ");
	    }
    }

	private static semiCRFInstance[] readCoNLLData_BIO(String Path, Boolean isTrain) throws IOException, FileNotFoundException {
		// TODO Auto-generated method stub]
		ArrayList<semiCRFInstance> result = new ArrayList<semiCRFInstance>();
		if(isTrain){
            File f = new File(Path);
            FileInputStream fin = new FileInputStream(f);
            Long len = f.length();
            byte []reader = new byte[len.intValue()];
            fin.read(reader);
            String []text = new String(reader).split("\n\n");
			for(int i = 0; i < text.length; i++){	 
				if(text[i].equals("-DOCSTART- -X- O O"))
				{
					continue;
				}
				String[] oneWords = text[i].split("\n");	
				ArrayList<String> input = new ArrayList<String>();
				ArrayList<String> output = new ArrayList<String>();
				for(int j = 0; j < oneWords.length; j++){
					String wordProperty[] = oneWords[j].split(" ");
					String inputWord = wordProperty[0];
					String label = wordProperty[wordProperty.length-1];
					input.add(inputWord);
					output.add(label);
				}
			    if(input.size()>60)
			    	continue;
				ArrayList<Span> reorganizedOutput = reorganized(output);
				semiCRFInstance instance = new semiCRFInstance(i+1, 1, input, reorganizedOutput);
				instance.setLabeled();
	            result.add(instance);
			}
			fin.close();
		} else {
			 File f = new File(Path);
	            FileInputStream fin = new FileInputStream(f);
	            Long len = f.length();
	            byte []reader = new byte[len.intValue()];
	            fin.read(reader);
	            String []text = new String(reader).split("\n\n");
				for(int i = 0; i < text.length; i++){
					if(text[i].equals("-DOCSTART- -X- O O"))
					{
						continue;
					}
					String[] oneWords = text[i].split("\n");
					
					ArrayList<String> input = new ArrayList<String>();
					ArrayList<String> output = new ArrayList<String>();
					for(int j = 0; j < oneWords.length; j++){
						String wordProperty[] = oneWords[j].split(" ");
						String inputWord = wordProperty[0];
						String label = wordProperty[wordProperty.length-1];
						input.add(inputWord);
						output.add(label);
					}
					if(input.size()>60)
				    	continue;
					ArrayList<Span> reorganizedOutput = reorganized(output);
					semiCRFInstance instance = new semiCRFInstance(i+1, 1, input, reorganizedOutput);
					instance.setUnlabeled();;
		            result.add(instance);
				}
				fin.close();
		}
		return result.toArray(new semiCRFInstance[result.size()]);
	}
	
	//用于把String类型的output整理成Span类型（相同的label合并成一个segment）
	//每个Span记录了对应segment的end position和label
	public static ArrayList<Span> reorganized(ArrayList<String> output){
		ArrayList<Span> result = new ArrayList<Span>();
		String currentLabel = "";
		for(int i = 0; i < output.size(); i++){
			if(output.get(i).equals("O")) {
				if(!currentLabel.equals("")){
					result.add(new Span(i-1, Label.get(currentLabel).getId()));
				}
				result.add(new Span(i, Label.get("O").getId()));
				currentLabel = "";
				continue;
			}
			if(currentLabel.equals("")){
				currentLabel = output.get(i).substring(2);
			} else {
				if(!currentLabel.equals(output.get(i).substring(2))){
				result.add(new Span(i-1, Label.get(currentLabel).getId()));
				currentLabel = output.get(i).substring(2);
			    } else {
			            if(output.get(i).startsWith("B-")){
			        	    result.add(new Span(i-1, Label.get(currentLabel).getId()));
						    continue;	
			            }  
			    }
			}
		}
		if(!currentLabel.equals(""))
		result.add(new Span(output.size()-1, Label.get(currentLabel).getId()));
		return result;
	}
 
}
    	
    	
    	
    	