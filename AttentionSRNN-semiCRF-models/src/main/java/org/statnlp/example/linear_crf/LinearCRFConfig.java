/**
 * 
 */
package org.statnlp.example.linear_crf;

/**
 * A configuration object for linear CRF models
 */
public class LinearCRFConfig {
	
	public int wordHalfWindowSize = 1;
	public int posHalfWindowSize = 0;
	
	public boolean productWithOutput = true;
	public boolean useBoundaryTag = true;
	
	public String[] features = new String[]{
			"word",
			"tag",
			"transition",
	};
	
	public LinearCRFConfig(){}
	
	public LinearCRFConfig(String[] args){
		int argIndex = 0;
		while(argIndex < args.length){
			switch(args[argIndex].substring(1)){
			case "wordHalfWindowSize":
				wordHalfWindowSize = Integer.parseInt(args[argIndex+1]);
				argIndex += 2;
				break;
			case "posHalfWindowSize":
				posHalfWindowSize = Integer.parseInt(args[argIndex+1]);
				argIndex += 2;
				break;
			case "features":
				features = args[argIndex+1].split(",");
				argIndex += 2;
				break;
			case "productWithOutput":
				productWithOutput = Boolean.parseBoolean(args[argIndex+1]);
				argIndex += 2;
				break;
			case "useBoundaryTag":
				useBoundaryTag = Boolean.parseBoolean(args[argIndex+1]);
				argIndex += 2;
				break;
			case "h":
			case "-help":
				printHelp();
				System.exit(0);
				break;
			default:
				throw new IllegalArgumentException("Unrecognized argument: "+args[argIndex]);
			}
		}
	}

	private static void printHelp(){
		System.out.println("Options:\n"
				+ "-wordHalfWindowSize <n>\n"
				+ "\tThe window size for word features, specified as half of the window size\n"
				+ "-posHalfWindowSize <n>\n"
				+ "\tThe window size for POS tag features, specified as half of the window size\n"
				+ "-features <comma-separated-feature-names>\n"
				+ "\tThe comma-separated list of features to be used, taken from:\n"
				+ "\t\t- WORD\n"
				+ "\t\t- WORD_BIGRAM\n"
				+ "\t\t- TAG\n"
				+ "\t\t- TAG_BIGRAM\n"
				+ "\t\t- TRANSITION\n"
				);
	}
	

}
