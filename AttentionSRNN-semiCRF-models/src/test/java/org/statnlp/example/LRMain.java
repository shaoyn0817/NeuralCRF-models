package org.statnlp.example;

import java.io.IOException;

import org.statnlp.commons.ml.opt.OptimizerFactory;
import org.statnlp.commons.types.Instance;
import org.statnlp.example.logstic_regression.LREval;
import org.statnlp.example.logstic_regression.LRFeatureManager;
import org.statnlp.example.logstic_regression.LRInstance;
import org.statnlp.example.logstic_regression.LRNetworkCompiler;
import org.statnlp.example.logstic_regression.LRReader;
import org.statnlp.example.logstic_regression.RelationType;
import org.statnlp.hypergraph.DiscriminativeNetworkModel;
import org.statnlp.hypergraph.GlobalNetworkParam;
import org.statnlp.hypergraph.NetworkConfig;
import org.statnlp.hypergraph.NetworkModel;

import net.sourceforge.argparse4j.ArgumentParsers;
import net.sourceforge.argparse4j.inf.ArgumentParser;
import net.sourceforge.argparse4j.inf.ArgumentParserException;
import net.sourceforge.argparse4j.inf.Namespace;

public class LRMain {

	
	public static double l2val = 0.01;
	public static int numThreads = 8;
	public static int numIteration = 1000;
	public static String trainFile = "data/rel/sample_train.txt";
	public static String testFile = "data/rel/sample_test.txt";
	
	public static void main(String[] args) throws IOException, InterruptedException {
		
		setArgs(args);
		
		/***
		 * Parameter settings and model configuration
		 */
		NetworkConfig.L2_REGULARIZATION_CONSTANT = l2val;
		NetworkConfig.NUM_THREADS = numThreads;
		
		LRInstance[] trainData = LRReader.readInsts(trainFile, true);
		System.out.println("#Relations: " + RelationType.RELS.size());
		System.out.println("Relations: " + RelationType.RELS.toString());
		RelationType.lock();
		GlobalNetworkParam gnp = new GlobalNetworkParam(OptimizerFactory.getLBFGSFactory());
		LRFeatureManager tfm = new LRFeatureManager(gnp);
		LRNetworkCompiler tnc = new LRNetworkCompiler();
		NetworkModel model = DiscriminativeNetworkModel.create(tfm, tnc);
		model.train(trainData, numIteration);
		
		/**
		 * Testing Phase
		 */
		LRInstance[] testInsts = LRReader.readInsts(testFile, false);
		Instance[] results = model.decode(testInsts);
		
		/***
		 * Evaluation Code
		 */
		LREval.evaluate(results);
	}
	
	
	/***
	 * Process the arguments in command line.
	 * @param args
	 */
	private static void setArgs(String[] args) {
		ArgumentParser parser = ArgumentParsers.newArgumentParser("")
				.defaultHelp(true).description("Logistic Regression Model for Relation Extraction");
		parser.addArgument("-t", "--thread").setDefault(8).help("number of threads");
		parser.addArgument("--l2").setDefault(l2val).help("L2 Regularization");
		parser.addArgument("--iter").setDefault(numIteration).help("The number of iteration.");
		parser.addArgument("--trainFile").setDefault(trainFile).help("The path of the training file");
		parser.addArgument("--testFile").setDefault(trainFile).help("The path of the testing file");
		Namespace ns = null;
        try {
            ns = parser.parseArgs(args);
        } catch (ArgumentParserException e) {
            parser.handleError(e);
            System.exit(1);
        }
        numThreads = Integer.valueOf(ns.getString("thread"));
        l2val = Double.valueOf(ns.getString("l2"));
        numIteration = Integer.valueOf(ns.getString("iter"));
        trainFile = ns.getString("trainFile");
        testFile = ns.getString("testFile");
        System.err.println(ns.getAttrs().toString());
	}
}
