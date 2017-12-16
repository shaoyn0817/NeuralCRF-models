package org.statnlp;

import org.statnlp.example.LinearCRFMain;
import org.statnlp.hypergraph.NetworkConfig;

import ext.svm_struct.SVMStruct;

/**
 * The code to test the equivalence between our SSVM implementation with SVM struct
 */
public class TestSSVM {

	public static void main(String[] args) throws Exception {
		NetworkConfig.NODE_COST = 1.0;
		NetworkConfig.EDGE_COST = 0.0;
		NetworkConfig.NORMALIZE_COST = false;
		String trainPath = "data/train.data";
		String testPath = "data/test.data";
		int numExamples = 77; // SVM-struct divides the margin by the number of examples in training data
		double margin = 1.0;
		String posHalfWindowSize = "0";
		String wordHalfWindowSize = "0";
		String features = "tag";
		
		// Linear CRF training and testing
		LinearCRFMain.main(new String[]{
				"-trainPath", trainPath,
				"-testPath", testPath,
				"-modelPath", "experiments/test/lcrf.model",
				"-logPath", "experiments/test/lcrf.log",
				"-writeModelText",
				
				"-modelType", "CRF",
				"-l2", "0.125",
				"-numIter", "1500",
				
				"--",
				
				"-wordHalfWindowSize", wordHalfWindowSize,
				"-posHalfWindowSize", posHalfWindowSize,
				"-features", features+",transition",
		});

		// SVM Struct training and testing
		SVMStruct.main(new String[]{
				"-C", String.format("%.3f", margin*numExamples),
				"-trainPath", trainPath,
				"-testPath", testPath,
				"-modelPath", "experiments/test/svmstruct.model",
				"-resultPath", "experiments/test/svmstruct.result",
				"-logPath", "experiments/test/svmstruct.log",
				"--",
				"-wordHalfWindowSize", wordHalfWindowSize,
				"-posHalfWindowSize", posHalfWindowSize,
				"-features", features,
				"-productWithOutput", "false",
		});
		
		// Our SSVM training and testing
		LinearCRFMain.main(new String[]{
				"-trainPath", trainPath,
				"-testPath", testPath,
				"-modelPath", "experiments/test/ssvm.model",
				"-logPath", "experiments/test/ssvm.log",
				"-writeModelText",
				"-weightInit", "file", "experiments/test/svmstruct.model.features",
				
				"-modelType", "SSVM",
				"-l2", "0.5",
				"-numIter", "1",
				"-margin", String.format("%.3f", margin),
				
				"--",
				
				"-wordHalfWindowSize", wordHalfWindowSize,
				"-posHalfWindowSize", posHalfWindowSize,
				"-features", features+",transition",
		});
		
		// Softmax-margin training and testing
		LinearCRFMain.main(new String[]{
				"-trainPath", trainPath,
				"-testPath", testPath,
				"-modelPath", "experiments/test/softmax_margin.model",
				"-logPath", "experiments/test/softmax_margin.log",
				"-writeModelText",
				"-weightInit", "file", "experiments/test/svmstruct.model.features", 
				
				"-modelType", "SOFTMAX_MARGIN",
				"-l2", "0.125",
				"-numIter", "1500",
				"-margin", String.format("%.3f", margin),
				
				"--",
				
				"-wordHalfWindowSize", wordHalfWindowSize,
				"-posHalfWindowSize", posHalfWindowSize,
				"-features", features+",transition",
		});
	}
}
