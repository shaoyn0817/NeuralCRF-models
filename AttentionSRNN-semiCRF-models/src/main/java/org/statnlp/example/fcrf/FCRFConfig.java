package org.statnlp.example.fcrf;

public class FCRFConfig {

	public String train = "data/conll2000/train.txt";
	public String test = "data/conll2000/test.txt";
	public String dev = "data/conll2000/dev.txt";
	
	public String nerOut = "data/conll2000/output/nerOut.txt";
	public String posOut = "data/conll2000/output/posOut.txt";
	
	public String nerPipeOut = "data/conll2000/output/nerPipeOut.txt";
	public String posPipeOut = "data/conll2000/output/posPipeOut.txt";
	
	public String dataset = "conll2000";
	
	public double l2val = 0.01;
	
	public static boolean windows = false;
	
	public static enum TASK{
		CHUNKING,
		TAGGING,
		JOINT;
	}
	
	public FCRFConfig (String dataset, double l2, boolean isWindows) {
		this.dataset = dataset;
		this.train = "data/"+dataset+"/train.txt";
		this.dev = "data/"+dataset+"/dev.txt";
		this.test = "data/"+dataset+"/test.txt";
		this.nerOut = "data/"+dataset+"/output/nerOut.txt";
		this.posOut = "data/"+dataset+"/output/posOut.txt";
		this.nerPipeOut = "data/"+dataset+"/output/nerPipeOut.txt";
		this.posPipeOut ="data/"+dataset+"/output/posPipeOut.txt";
		this.l2val = l2;
		FCRFConfig.windows = isWindows;
		
	}
}
