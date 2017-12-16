package org.statnlp.example.linear_ne;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.statnlp.commons.io.RAWF;
import org.statnlp.commons.types.Sentence;
import org.statnlp.commons.types.WordToken;

public class EReader {

	
	public static ECRFInstance[] readData(String path, boolean setLabel, int number) throws IOException{
		return readData(path, setLabel, number, "IOB");
	}
	
	/**
	 * 
	 * @param path
	 * @param setLabel
	 * @param number
	 * @param encoding: IOB, IOBES, NONE (by default it's iob encoding)
	 * @return
	 * @throws IOException
	 */
	public static ECRFInstance[] readData(String path, boolean setLabel, int number, String encoding) throws IOException{
		BufferedReader br = RAWF.reader(path);
		String line = null;
		List<ECRFInstance> insts = new ArrayList<ECRFInstance>();
		int index =1;
		ArrayList<WordToken> words = new ArrayList<WordToken>();
		ArrayList<String> es = new ArrayList<String>();
		String prevLine = null;
		String prevEntity = "O";
		Entity.get("O");
		while((line = br.readLine())!=null){
			if(line.startsWith("-DOCSTART-")) { prevLine = "-DOCSTART-"; continue;}
			if(line.equals("") && !prevLine.equals("-DOCSTART-")){
				WordToken[] wordsArr = new WordToken[words.size()];
				words.toArray(wordsArr);
				Sentence sent = new Sentence(wordsArr);
				ECRFInstance inst = new ECRFInstance(index++,1.0,sent);
				inst.output = es;
//				System.err.println(es.toString());
				if(!encoding.equals("NONE")) setEncoding(inst, encoding);
//				System.err.println(inst.entities.toString());
				if(setLabel) inst.setLabeled(); else inst.setUnlabeled();
				insts.add(inst);
				words = new ArrayList<WordToken>();
				es = new ArrayList<String>();
				prevLine = "";
				prevEntity = "O";
				if(number!=-1 && insts.size()==number) break;
				continue;
			}
			if(line.equals("") && prevLine.equals("-DOCSTART-")){
				prevLine = ""; prevEntity = "O"; continue;
			}
			String[] values = line.split(" ");
			String rawCurrEntity = values[3];

			String currEntity = null;
			if(!encoding.equals("NONE")){
				if(rawCurrEntity.equals("O")) currEntity = "O";
				else{
					if(prevEntity.equals("O")) currEntity = "B-"+rawCurrEntity.substring(2);
					else if(prevEntity.substring(2).equals(rawCurrEntity.substring(2))){
						if(prevEntity.equals(rawCurrEntity)){
							currEntity = prevEntity;
						}else{
							assert !prevEntity.equals("O");
							if(prevEntity.startsWith("B-")) currEntity = "I-"+rawCurrEntity.substring(2);
							else currEntity = "B-"+rawCurrEntity.substring(2);
						}
						
					}else{
						currEntity = "B-"+rawCurrEntity.substring(2);
					}
				}
			}else
				currEntity = rawCurrEntity;
			words.add(new WordToken(values[0],values[1],-1, currEntity));
			es.add(currEntity);
			prevLine = line;
			prevEntity = currEntity;
		}
		br.close();
		List<ECRFInstance> myInsts = insts;
		String type = setLabel? "Training":"Testing";
		System.err.println(type+" instance, total:"+ myInsts.size()+" Instance. ");
		return myInsts.toArray(new ECRFInstance[myInsts.size()]);
	}

	private static void setEncoding(ECRFInstance inst, String encoding){
		String prevEntity = "O";
		if(encoding.equals("IOBES")){
			ArrayList<String> output = inst.getOutput();
			Sentence sent = inst.getInput();
			for(int pos=0; pos<inst.size(); pos++){
				String currEntity = output.get(pos);
				String nextEntity = pos<inst.size()-1? output.get(pos+1):"O";
				
				String newCurrEntity = currEntity;
				if(prevEntity.equals("O")){
					if(!currEntity.equals("O")) {
						if(nextEntity.startsWith("I-") && nextEntity.substring(2).equals(currEntity.substring(2))) newCurrEntity="B-"+currEntity.substring(2);
						else newCurrEntity = "S-"+currEntity.substring(2);
					}
				}else if(!currEntity.equals("O")){
					//previous Entity is not O. is something. and the current is not O
					if(prevEntity.substring(2).equals(currEntity.substring(2))){
						if(prevEntity.startsWith("E-") || prevEntity.startsWith("S-")){
							if(nextEntity.startsWith("I-") && nextEntity.substring(2).equals(currEntity.substring(2))) newCurrEntity = "B-"+currEntity.substring(2);
							else newCurrEntity = "S-"+currEntity.substring(2);
						}else if(prevEntity.startsWith("I-")){
							if(currEntity.startsWith("I-")){
								if(nextEntity.length()>2 && nextEntity.substring(2).equals(currEntity.substring(2)) && nextEntity.startsWith("I-")){
									newCurrEntity = currEntity;
								}else newCurrEntity = "E-"+currEntity.substring(2);
							}else {
								assert currEntity.equals("O");
								newCurrEntity = currEntity; //this one should be only O. check
							}
						}else if(prevEntity.startsWith("B-")){
							assert currEntity.startsWith("I-"); //can only be I-
							if(nextEntity.equals(currEntity)){
								newCurrEntity = currEntity;
							}else{
								newCurrEntity = "E-"+currEntity.substring(2);
							}
							
						}
					}else{
						//prev not o, current not o, and they are not equal. must be B- or S-
						if(!nextEntity.startsWith("I-") || !nextEntity.substring(2).equals(currEntity.substring(2))) newCurrEntity = "S-"+currEntity.substring(2);
						else  newCurrEntity = "B-"+currEntity.substring(2);
					}
				}
				
				
				output.set(pos, newCurrEntity); 
				sent.get(pos).setEntity(newCurrEntity);
				Entity.get(newCurrEntity);
				prevEntity = newCurrEntity;
			}
		}else if(encoding.equals("IOB")){
			//TODO: do nothing, since by default it's iob encoding.
		}
	}
 	
}
