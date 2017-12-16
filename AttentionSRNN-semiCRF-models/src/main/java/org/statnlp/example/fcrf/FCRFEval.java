package org.statnlp.example.fcrf;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;

import org.statnlp.commons.io.RAWF;
import org.statnlp.commons.types.Instance;
import org.statnlp.commons.types.Sentence;

public class FCRFEval {

	
	/**
	 * 
	 * @param testInsts
	 * @param nerOut: word, true pos, true entity, pred entity
	 * @throws IOException
	 */
	public static void evalFscore(Instance[] testInsts, String nerOut) throws IOException{
		PrintWriter pw = RAWF.writer(nerOut);
		for(int index=0;index<testInsts.length;index++){
			FCRFInstance eInst = (FCRFInstance)testInsts[index];
			ArrayList<String> predEntities = eInst.getChunkPredictons();
			ArrayList<String> trueEntities = eInst.getOutput();
			Sentence sent = eInst.getInput();
			for(int i=0;i<sent.length();i++){
				pw.write(sent.get(i).getForm()+" "+sent.get(i).getTag()+" "+trueEntities.get(i)+" "+predEntities.get(i)+"\n");
			}
			pw.write("\n");
		}
		pw.close();
		evalFscore(nerOut);
	}
	
	
	private static void evalFscore(String outputFile) throws IOException{
		try{
			System.err.println("perl eval/conlleval.pl < "+outputFile);
			ProcessBuilder pb = null;
			if(FCRFConfig.windows){
				pb = new ProcessBuilder("D:/Perl64/bin/perl","eval/conlleval.pl"); 
			}else{
				pb = new ProcessBuilder("eval/conlleval.pl"); 
			}
			pb.redirectInput(new File(outputFile));
			pb.redirectOutput(ProcessBuilder.Redirect.INHERIT);
			pb.redirectError(ProcessBuilder.Redirect.INHERIT);
			pb.start();
		}catch(IOException ioe){
			ioe.printStackTrace();
		}
	}
	
	/**
	 * Evaluation of POS tagging result
	 * @param testInsts
	 * @param posOut: the output of the pos file: word, trueTag, predTag, trueChunk
	 * @throws IOException
	 */
	public static void evalPOSAcc(Instance[] testInsts, String posOut) throws IOException{
		PrintWriter pw = RAWF.writer(posOut);
		int corr = 0;
		int total = 0;
		for(int index=0;index<testInsts.length;index++){
			FCRFInstance eInst = (FCRFInstance)testInsts[index];
			ArrayList<String> tPred = eInst.getTagPredictons();
			Sentence sent = eInst.getInput();
			for(int i=0;i<sent.length();i++){
				if(sent.get(i).getTag().equals(tPred.get(i)))
					corr++;
				total++;
				pw.write(sent.get(i).getForm()+" "+sent.get(i).getTag()+" "+tPred.get(i)+" "+sent.get(i).getEntity()+"\n");
			}
			pw.write("\n");
		}
		System.out.printf("[POS Accuracy]: %.2f%%\n", corr*1.0/total*100);
		pw.close();
	}
	
	public static void evalChunkAcc(Instance[] testInsts) throws IOException{
		int corr = 0;
		int total = 0;
		for(int index=0;index<testInsts.length;index++){
			FCRFInstance eInst = (FCRFInstance)testInsts[index];
			ArrayList<String> tPred = eInst.getChunkPredictons();
			ArrayList<String> trueEntities = eInst.getOutput();
			for(int i=0;i<tPred.size();i++){
				if(tPred.get(i).equals(trueEntities.get(i)))
					corr++;
				total++;
			}
		}
		System.out.printf("[Chunking Accuracy]: %.2f%%\n", corr*1.0/total*100);
	}
	
	public static void evalJointAcc(Instance[] testInsts) throws IOException{
		int corr = 0;
		int total = 0;
		for(int index=0;index<testInsts.length;index++){
			FCRFInstance eInst = (FCRFInstance)testInsts[index];
			ArrayList<String> ePred = eInst.getChunkPredictons();
			ArrayList<String> trueEntities = eInst.getOutput();
			ArrayList<String> tPred = eInst.getTagPredictons();
			Sentence sent = eInst.getInput();
			for(int i=0;i<ePred.size();i++){
				if(ePred.get(i).equals(trueEntities.get(i)) && sent.get(i).getTag().equals(tPred.get(i)))
					corr++;
				total++;
			}
		}
		System.out.printf("[Joint Accuracy]: %.2f%%\n", corr*1.0/total*100);
	}
	
}
