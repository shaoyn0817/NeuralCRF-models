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
    	String trainPath = "data/con03/eng.train";
    	String devPath = "data/con03/eng.testa";
		String testPath = "data/con03/eng.testb";
		//String trainPath = "experiment/data/testdata.txt";
		//String testPath = "experiment/data/testdata.txt";
		
        int validbatch = 300;
		NetworkConfig.TRAIN_MODE_IS_GENERATIVE = Boolean.parseBoolean(System.getProperty("generativeTraining", "false"));
		NetworkConfig.PARALLEL_FEATURE_EXTRACTION = Boolean.parseBoolean(System.getProperty("parallelTouch", "true"));
		NetworkConfig.CACHE_FEATURES_DURING_TRAINING = Boolean.parseBoolean(System.getProperty("cacheFeatures", "true"));
		NetworkConfig.L2_REGULARIZATION_CONSTANT = Double.parseDouble(System.getProperty("l2", "0.01"));
		NetworkConfig.NUM_THREADS = Integer.parseInt(System.getProperty("numThreads", "40"));
        
		
		NetworkConfig.USE_BATCH_TRAINING = true; // To use or not to use mini-batches in gradient descent optimizer
		NetworkConfig.BATCH_SIZE = Integer.parseInt(System.getProperty("batchSize", "10"));  // The mini-batch size (if USE_BATCH_SGD = true)
		NetworkConfig.MARGIN = Double.parseDouble(System.getProperty("svmMargin", "1.0"));

		// Set weight to not random to make meaningful comparison between sequential and parallel touch
		NetworkConfig.RANDOM_INIT_WEIGHT = false;
		NetworkConfig.FEATURE_INIT_WEIGHT = 0.0;
	    NetworkConfig.USE_NEURAL_FEATURES = true;
		NetworkConfig.OS = "linux";
	    NetworkConfig.PRINT_BATCH_OBJECTIVE = true;
	    NetworkConfig.RANDOM_BATCH = true;
		int numIterations = Integer.parseInt(System.getProperty("numIter", "20000"));
	    PrintStream outstream = null;//new PrintStream("experiment/model.txt");
		semiCRFInstance[] trainInstances = readCoNLLData(trainPath, true);
		semiCRFInstance[] devInstances = readCoNLLData(devPath, false);
		int size = trainInstances.length;
		System.err.println("Read.."+size+" instances from ");
		
		OptimizerFactory optimizer = OptimizerFactory.getGradientDescentFactoryUsingGradientClipping(0.01);
		//OptimizerFactory optimizer = OptimizerFactory.getLBFGSFactory();
			
		System.out.println(Label.LABELS.size());
		List<NeuralNetworkCore> nets = new ArrayList<NeuralNetworkCore>();
		if(NetworkConfig.USE_NEURAL_FEATURES){
			if (neuralType.equals("AttentionSRNN")) {
				int hiddenSize = 100;
				String opt = "sgd";
				boolean bidirection = true;
				nets.add(new AttentionSRNN(hiddenSize, bidirection, opt, 0.01, 5, Label.LABELS.size(), 0, "glove"));
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

	     model.train(trainInstances, numIterations, devInstances, evalFunc, 5);
		
		
		
		//model.train(trainInstances, numIterations);
  

		semiCRFInstance[] testInstances = readCoNLLData(testPath, false);
		Instance[] predictions = new Instance[testInstances.length];
		int number = testInstances.length/validbatch+1;
		for(int i = 0; i < number; i++) {
		int start = i*validbatch;
		int end = (i+1)*validbatch;
		if(end > testInstances.length)
			end = testInstances.length;
		semiCRFInstance[] batch = new semiCRFInstance[validbatch];
		for(int j = start; j < end; j++) {
			batch[j] = testInstances[j];
		}
		Instance temp[] = model.decode(batch);
		for(int j = start; j < end; j++) {
			predictions[j] = temp[j];
		}
		}
	    //segmentEvaluation e = new segmentEvaluation(predictions);
        segmentEvaluation.eval(predictions);
	    
	    
	    /*
		int labelAll[] = new int[Label.LABELS.size()];
		int labelFind[] = new int[Label.LABELS.size()];
		int labelFindRight[] = new int[Label.LABELS.size()];
		int num = 0;
		
		System.out.println("共有 "+Label.LABELS.size()+" 个label");
		//xiru wenjian
		String outPath = "data/myResult.txt";
		FileOutputStream fout = new FileOutputStream(new File(outPath));
		
		for(Instance ins: predictions){
			num++;

			String gold = "";
			String pre = "";
			String words = "";
			semiCRFInstance instance = (semiCRFInstance)ins;
			ArrayList<String> word = instance.getInput();
			int err = 0;	
			Integer[] out = recover(instance.prediction);
			Integer[] result = recover(instance.output);
			

			////写入输出结果 进行查看
			
			String currentTargetLabel = "nnnnn";
	    	String currentOutLabel = "nnnnnn";
	    	for(int i = 0; i < out.length; i++){
	    	    String t = "";
	    		String targetlabel = Label.get(result[i]).getForm();
	    		String outlabel =  Label.get(out[i]).getForm();
	            if(!targetlabel.equals(currentTargetLabel))
	            	{
	            	t += "B-"+targetlabel+" ";
	            	currentTargetLabel = targetlabel;
	            	}
	            else 
	            	t += "I-"+targetlabel+" ";
	            
	            if(!outlabel.equals(currentOutLabel)){
	            	t += "B-"+outlabel+"\r\n";
	            	currentOutLabel = outlabel;
	            }
	            else
	            	t += "I-"+outlabel+"\r\n";
	  
	    		fout.write((t).getBytes());
	    	}
			fout.write("\r\n".getBytes());
			
			
			
			
			
			
			if(out.length != result.length) System.out.println("recover方法出错");
			for(int i = 0; i < out.length; i++){
				labelAll[result[i]]++;
				if(out[i] == result[i]){
					labelFindRight[result[i]]++;
				}
				labelFind[out[i]]++;
				words += word.get(i)+"  ";
				gold += Label.get(result[i]).getForm()+"  ";
				if(result[i] != out[i])
				    {
					    err++;
					    pre += "~~"+Label.get(out[i]).getForm()+"~~  ";
				    }
				else
					pre += Label.get(out[i]).getForm()+"  ";
			}
			
			
			if(err > 10000){
				System.out.println(words);
				System.out.println(gold);
				System.out.println(pre);
				System.out.println();
				System.out.println();
			}
		}
		double averagePrecise = 0;
		double averageRecall = 0;

						
				
		System.out.println("共有"+num+"个测试样例");
		for(int i = 0; i < labelAll.length; i++){
			System.out.println(Label.get(i).getForm()
					+" precision: "+(double)labelFindRight[i]/labelFind[i]+"  "+"   召回率:"+(double)labelFindRight[i]/labelAll[i]);
		    averagePrecise += (double)labelFindRight[i]/labelFind[i] / labelAll.length;
		    averageRecall += (double)labelFindRight[i]/labelAll[i] / labelAll.length;
		}
		System.out.println();
		System.out.println("average precision:"+averagePrecise);
		System.out.println("average recall:"+averageRecall);
		*/
    }

	private static semiCRFInstance[] readCoNLLData(String Path, Boolean isTrain) throws IOException, FileNotFoundException {
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
					if(!label.equals("O"))
						label = label.substring(2);

					input.add(inputWord);
					output.add(label);
				}
			    if(input.size()>60)
			    	continue;
				ArrayList<Span> reorganizedOutput = reorganized(output);
				//for(int m = 0; m < reorganizedOutput.size(); m++){
			    	//System.out.print(reorganizedOutput.get(m)._end+":"+reorganizedOutput.get(m)._label._form+"  ");
			    //}
			    //System.out.println();
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
						if(!label.equals("O"))
							label = label.substring(2);

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
		int flag = 0;
		String currentLabel = "";
		for(int i = 0; i < output.size(); i++){
			if(flag == 0 && output.get(i).equals("O")) {
				result.add(new Span(i, Label.get("O").getId()));
				continue;
			}
			if(flag == 0){
				flag = 1;
				currentLabel = output.get(i);
			} else {
				if(!currentLabel.equals(output.get(i))){
				result.add(new Span(i-1, Label.get(currentLabel).getId()));
				if(output.get(i).equals("O"))
				{
					result.add(new Span(i, Label.get("O").getId()));
					flag = 0;
					
					currentLabel = "";
					continue;
				} else
				    {
					currentLabel = output.get(i);
				    }
			    }
			}
		}
		if(!currentLabel.equals(""))
		result.add(new Span(output.size()-1, Label.get(currentLabel).getId()));
		return result;
	}

    public static Integer[] recover(ArrayList<Span> output){
    	ArrayList<Integer> result = new ArrayList<Integer>();
    	int curr = -1;
    	for(int i = 0; i < output.size(); i ++){
    		int len = output.get(i)._end - curr;
            for(int j = 0; j < len; j++){
            	result.add(output.get(i)._label._id);
            }
            curr = output.get(i)._end;
    	}
    	Integer a[] = new Integer[result.size()];
    	result.toArray(a);
    	return a;
    }
}
    	
    	
    	
    	