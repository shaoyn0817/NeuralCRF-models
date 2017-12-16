package org.statnlp.maxmargin;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;

import org.statnlp.commons.ml.opt.OptimizerFactory;
import org.statnlp.commons.types.Label;
import org.statnlp.commons.types.LinearInstance;
import org.statnlp.example.linear_crf.LinearCRFFeatureManager;
import org.statnlp.example.linear_crf.LinearCRFNetworkCompiler;
import org.statnlp.hypergraph.NetworkConfig.ModelType;
import org.statnlp.hypergraph.NetworkConfig.StoppingCriteria;
import org.statnlp.util.GenericPipeline;
import org.statnlp.util.instance_parser.DelimiterBasedInstanceParser;

public class LinearCRFMain {
	
	public static void main(String args[]) throws IOException, InterruptedException{
		GenericPipeline pipeline = new GenericPipeline();
		DelimiterBasedInstanceParser parser = new DelimiterBasedInstanceParser(pipeline){
			private static final long serialVersionUID = -6995904257432947531L;

			@Override
			public LinearInstance<Label>[] buildInstances(String... sources)
					throws FileNotFoundException {
				try {
					return readCoNLLData(this, sources[0], true, true);
				} catch (IOException e) {
					throw new RuntimeException(e);
				}
			}
			
		};
		parser.getLabel("O");
		parser.getLabel("I-MISC");
		parser.getLabel("I-LOC");
		parser.getLabel("I-PER");
		parser.getLabel("I-ORG");
		pipeline.withTrainPath("data/CoNLL2003/eng.train")
				.withNumTrain(1000)
//				.withDevPath("data/CoNLL2003/eng.testa")
//				.withNumDev(100)
				.withTestPath("data/CoNLL2003/eng.testb")
				.withNumTest(100)
				.withInstanceParser(parser)
				.withNetworkCompiler(LinearCRFNetworkCompiler.class)
				.withFeatureManager(LinearCRFFeatureManager.class)
				.withModelPath("test.model")
				.withLogPath("test.log")
				.withL2(0.0001)
				.withWeightInit(0.0)
				.withModelType(ModelType.SSVM)
				.withOptimizerFactory(OptimizerFactory.getGradientDescentFactoryUsingAdaGrad(0.2))
				.withMargin(1.0)
				.withNodeMismatchCost(1.0)
				.withEdgeMismatchCost(0.0)
				.withUseBatchTraining(true)
				.withBatchSize(1)
				.withStoppingCriteria(StoppingCriteria.MAX_ITERATION_REACHED)
				.withMaxIter(100000)
				.withEvaluateEvery(0)
				.withWriteModelAsText(true)
				.addTask("train")
				.addTasks("test", "evaluate")
				;
		pipeline.execute();
		return;
	}
	
	@SuppressWarnings("unchecked")
	private static LinearInstance<Label>[] readCoNLLData(DelimiterBasedInstanceParser parser, String fileName, boolean withLabels, boolean isLabeled) throws IOException{
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
			if(line.startsWith("-DOCSTART-")){
				continue;
			}
			if(line.length() == 0){
				if(words.size() == 0){
					continue;
				}
				LinearInstance<Label> instance = new LinearInstance<Label>(instanceId, 1, words, labels);
				if(isLabeled){
					instance.setLabeled(); // Important!
				} else {
					instance.setUnlabeled();
				}
				instanceId++;
				result.add(instance);
				words = null;
				labels = null;
			} else {
				int lastSpace = line.lastIndexOf(" ");
				String[] features = line.substring(0, lastSpace).split(" ");
				words.add(features);
				if(withLabels){
					String labelStr = line.substring(lastSpace+1);
					labelStr = labelStr.replace("B-", "I-");
					Label label = parser.getLabel(labelStr);
					labels.add(label);
				}
			}
		}
		br.close();
		return result.toArray(new LinearInstance[result.size()]);
	}
}
