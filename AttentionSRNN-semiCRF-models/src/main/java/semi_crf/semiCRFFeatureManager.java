package semi_crf;

//王者的基础上打开了orignpunc
import java.util.ArrayList;
import java.util.AbstractMap.SimpleImmutableEntry;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.statnlp.example.mention_hypergraphNoEmb.AttributedWord;
import org.statnlp.hypergraph.FeatureArray;
import org.statnlp.hypergraph.FeatureManager;
import org.statnlp.hypergraph.GlobalNetworkParam;
import org.statnlp.hypergraph.Network;
import org.statnlp.hypergraph.NetworkConfig;

import semi_crf.semiCRFNetworkCompiler.NODE_TYPES;


//~
public class semiCRFFeatureManager extends FeatureManager{

	private static final long serialVersionUID = 353522079325768165L;

	public enum FeatureType{
		OrignPunc(false), //final2 打开了这个选项
		WordBigram(true),
		StartEndWithPunctuation(false),//**
		EndEndWithPunctuation(false),//**.o
		EndPunctuationFeature(false),//**
		FirstChar(true),
		FirstTwoChar(true),
		FirstThreeChar(true),
		FirstFourChar(true),
		LastChar(true),
		LastTwoChar(true),
		LastThreeChar(true),
		LastFourChar(true),  //above should be all true
		LowerCase(true),//**
		StartCapitalFeature(true),//**
		EndCapitalFeature(true),//**
		CapitalFeature(true),//aaaaaaaaaaaaaaaaaaa
		BinaryCapitalFeature(true),//确认过
		StartNumFeature(false),//
		EndNumFeature(false),//
		NumFeature(false),
		//删除了binaryNumfeature
		BinaryNumFeature(false), ////感觉上要取消
		PunctuationFeature(false),//原来是false
		PossibleVol(false),

		
		IndexedWord(false),//daiding
		InvertedWord(false),//daiding
		RelativePosition(false),
		PossibleEditor(false),//daiding
		PossibleJournal(false),
		BeforeWord(true),//**
		AfterWord(true),//**
		BeforeLabel(false),//**
		BinaryLabel(true),

		
		StartWord(false),//**  lowercase and no punc   made 王者3
        EndWord(false),//** lowercase and no punc   made 王者3
        
        

        StartBoundaryWord(false),//_________________________duiying base best 3
        EndBoundaryWord(false),//__________________________duiying base best 3
        
	    ;
		
        private boolean isEnabled;
		
		private FeatureType(){
			this(true);
		}
		
		private FeatureType(boolean enabled){
			this.isEnabled = enabled;
		}
		
		public void enable(){
			this.isEnabled = true;
		}
		
		public void disable(){
			this.isEnabled = false;
		}
		
		public boolean enabled(){
			return isEnabled;
		}
		
		public boolean disabled(){
			return !isEnabled;
		}
	}
	
	public semiCRFFeatureManager(GlobalNetworkParam param_g) {
		super(param_g);
	}
	
	@Override
	protected FeatureArray extract_helper(Network network, int parent_k, int[] children_k, int children_k_index) {
		semiCRFInstance instance = (semiCRFInstance) network.getInstance();
		int[] array_k = network.getNodeArray(parent_k);
		
		int position = array_k[0];
		int tagId = array_k[1];
		int nodeType = array_k[4];
		
		if(nodeType == NODE_TYPES.LEAF.ordinal()
				|| nodeType == NODE_TYPES.ROOT.ordinal()){
			return FeatureArray.EMPTY;
		} 
		
		GlobalNetworkParam param_g = this._param_g;
		ArrayList<Integer> features = new ArrayList<Integer>();
		int[] childArray = network.getNodeArray(children_k[0]);
		int childPosition = childArray[0];
		int childTagId = childArray[1];
		int childNodeType = childArray[4];
		ArrayList<String> input = instance.getInput();
		int window = 2;
		
		
		if(position == input.size()){
			return FeatureArray.EMPTY;
		} 
		
		String segment = "";
		if(childNodeType == NODE_TYPES.LEAF.ordinal())
			childPosition = -1;
		for(int i = childPosition+1; i <= position; i++){
			
			segment += input.get(i)+" ";
		}
		segment = segment.trim();
		
		//compute neural feature row number
        int l = position - childPosition;
        int pre = 0;
        int padlen = 42;
        for(int m = 1; m < l;m++) {
        	pre += padlen-(m-1);
        }
        pre += childPosition+2-1;
		//int []toFind = new int[position-childPosition];
		//for(int i = 0; i < toFind.length; i++){
			//toFind[i] = childPosition+1+i;
		//}
		int []toFind = new int[]{childPosition, childPosition+1, position, position+1};
		
		String s = sen2str(input);
		if(NetworkConfig.USE_NEURAL_FEATURES){
			Object in = null;
			if(semi_main.neuralType.equals("AttentionSRNN")) {
				in = new SimpleImmutableEntry<String, Integer>(s, pre);
			}
			else if(semi_main.neuralType.equals("mlp")) {
				in = segment;
			}
			else if(semi_main.neuralType.equals("continuous")) {
				in = segment;
			}
			int id = tagId;
			this.addNeural(network, 0, parent_k, children_k_index, in, id);
		}
        
		
		
		//Indexed word
		if(FeatureType.IndexedWord.enabled()){
			for(int i = childPosition+1; i <= position; i++){
				features.add(param_g.toFeature(network, FeatureType.IndexedWord+":"+(i-childPosition-1), tagId+"", input.get(i).replaceAll("[\\pP\\p{Punct}]+", "").toLowerCase()));
			}
		}
		
		
		//StartBinaryNumFeature
			if(FeatureType.BinaryNumFeature.enabled()){
				for(int i = childPosition+1; i < position; i++){
				            String firstSymbol = "*";
				            String secondSymbol = "*";
				            if(i >= 0 && i <= input.size()-1)
				            	firstSymbol = this.getNumFeature(input, i);
				            if(i+1 >= 0 && i+1 <= input.size()-1)
				            	secondSymbol = this.getNumFeature(input, i+1); 
							features.add(param_g.toFeature(network, FeatureType.BinaryNumFeature+"", tagId+"", firstSymbol+" "+secondSymbol));
				}
	}
		
		
		
		
		//Inverted word
	    if(FeatureType.InvertedWord.enabled()){
			for(int i = position; i >= childPosition+1; i--){
				features.add(param_g.toFeature(network, FeatureType.InvertedWord+":"+(position - i), tagId+"", input.get(i)));
			}
		}		
		
		
		
		//StartWord
		if(FeatureType.StartWord.enabled()){
			features.add(param_g.toFeature(network, FeatureType.StartWord+"", tagId+"", input.get(childPosition+1).replaceAll("[\\pP\\p{Punct}]+", "").toLowerCase()));
		}
				
		//EndWord
		if(FeatureType.EndWord.enabled()){
			features.add(param_g.toFeature(network, FeatureType.EndWord+"", tagId+"", input.get(position).replaceAll("[\\pP\\p{Punct}]+", "").toLowerCase()));
		}
				
		//segment\u524d\u7684\u4e00\u4e2a\u5355\u8bcd
		if(FeatureType.BeforeWord.enabled()){
		    if(childPosition >= 0){
		    	features.add(param_g.toFeature(network, FeatureType.BeforeWord+"", tagId+"", input.get(childPosition)));
		    } else {
		    	features.add(param_g.toFeature(network, FeatureType.BeforeWord+"", tagId+"", "*"));
		    }
		}
		
		//segment\u540e\u7684\u4e00\u4e2a\u5355\u8bcd
		if(FeatureType.AfterWord.enabled()){
		    if(position < input.size() - 1){
		        features.add(param_g.toFeature(network, FeatureType.AfterWord+"", tagId+"", input.get(position+1)));
		    } else {
	            features.add(param_g.toFeature(network, FeatureType.AfterWord+"", tagId+"", "*"));
	        }
		}
		
	

		
		
	   //RealativePosition
	   if(FeatureType.RelativePosition.enabled()){
	      int length = input.size();
	  	  int start = childPosition+1;
	  	  int end = position;
	  	  int middle = (start+end)/2;
	  	  int relativePosition = (int)((double)middle/length * 12);
	  	  features.add(param_g.toFeature(network, FeatureType.RelativePosition+":"+relativePosition, tagId+"", ""));
	   }
		
		
	   //BeforeLabel
	   if(FeatureType.BeforeLabel.enabled()){
		  if(childPosition >= 0){
			features.add(param_g.toFeature(network, FeatureType.BeforeLabel+"", tagId+"", Label.get(childTagId).getForm()));
		  } else {
			features.add(param_g.toFeature(network, FeatureType.BeforeLabel+"", tagId+"", "0"));
		  }
	   }	
		
			
			
       //BinaryLabel
	   if(FeatureType.BinaryLabel.enabled()){
	       if(childPosition >= 0 && childPosition <= input.size()-1){
		       features.add(param_g.toFeature(network, FeatureType.BinaryLabel+"", tagId+"", Label.get(childTagId).getForm()+" "+Label.get(tagId).getForm()));
		   } else if(childPosition == -1){
		       features.add(param_g.toFeature(network, FeatureType.BinaryLabel+"", tagId+"", "start "+Label.get(tagId).getForm()));
		   } else {
			   System.out.println("\u51fa\u73b0\u602a\u5f02\u60c5\u51b5");
			   System.out.println(childPosition);
		   }
	   }
			
		
     
		
		//OriginPunc
		if(FeatureType.OrignPunc.enabled()){
			for(int i = childPosition+1; i<=position; i++){
				String word = "";
				if(i >= 0 && i <= input.size()-1)
					word = input.get(i).replaceAll("[a-zA-Z0-9]", "");
		            features.add(param_g.toFeature(network, FeatureType.OrignPunc+"", tagId + "", word));
		        }
		}		
		
		
		//WordBigram
		if(FeatureType.WordBigram.enabled()){
			for(int i = childPosition+1; i < position; i++){
				String beforword = "*";
				String word = "*";
				if(i >= 0 && i < input.size()){
					beforword = input.get(i);
				}
				if(i+1 >= 0 && i+1 < input.size()){
					word = input.get(i+1);
				}
		        features.add(param_g.toFeature(network, FeatureType.WordBigram+":"+(i-childPosition-1), tagId + "", beforword+" "+word));
			}
		}
		
		//StartWithPunctuation
		if(FeatureType.StartEndWithPunctuation.enabled()){
			for(int j = -window; j <= window; j++){
				String word = "*";
				if(childPosition+1+j >= 0 && childPosition+1+j <= input.size()-1)
					word = input.get(childPosition+1+j);
				char lastChar = word.charAt(word.length()-1);
				if(Character.isUpperCase(lastChar))
					word = "A";
				else if(Character.isLowerCase(lastChar))
			        word = "a";
				else word = String.valueOf(lastChar);
				features.add(param_g.toFeature(network, FeatureType.StartEndWithPunctuation+":"+j, tagId+"", word));
				}
		}
				
		
		
		//EndWithPunctuation
		if(FeatureType.EndEndWithPunctuation.enabled()){
			for(int j = -window; j <= window; j++){
			    String word = "*";
			    if(position+j >= 0 && position+j <= input.size()-1)
				    word = input.get(position+j);
			    char lastChar = word.charAt(word.length()-1);
			    if(Character.isUpperCase(lastChar))
				    word = "A";
			    else if(Character.isLowerCase(lastChar))
				    word = "a";
			    else word = String.valueOf(lastChar);
			        features.add(param_g.toFeature(network, FeatureType.EndEndWithPunctuation+":"+j, tagId+"", word));
			}
		}
		
		
		//EndPunctuationFeature
		if(FeatureType.EndPunctuationFeature.enabled()){
			for(int i = childPosition+1; i <= position; i++){
			    String word = "*";
			    if(i >= 0 && i <= input.size()-1)
				    word = input.get(i);
			    char lastChar = word.charAt(word.length()-1);
			    if(Character.isUpperCase(lastChar))
				    word = "A";
			    else if(Character.isLowerCase(lastChar))
				    word = "a";
			    else word = String.valueOf(lastChar);
			        features.add(param_g.toFeature(network, FeatureType.EndPunctuationFeature+":"+(i-childPosition-1), tagId+"", word));
			}
		}		
		
		
		//FirstChar
		if(FeatureType.FirstChar.enabled()){
			for(int i = childPosition+1; i <= position; i++){
				String word = "*";
				if(i >= 0 && i <= input.size()-1)
				    word = input.get(i);
				char firstChar = word.charAt(0);
		        features.add(param_g.toFeature(network, FeatureType.FirstChar+"", tagId+"", firstChar+""));
			}
		}
		
		//FirstTwoChar
		if(FeatureType.FirstTwoChar.enabled()){
			for(int i = childPosition+1; i <= position; i++){
				String word = "*";
				if(i >= 0 && i <= input.size()-1)
				    word = input.get(i);
				word = word.length() >= 2 ? word.substring(0,2) : word;
		        features.add(param_g.toFeature(network, FeatureType.FirstTwoChar+"", tagId+"", word));
			}
		}
		
		//FirstThreeChar
		if(FeatureType.FirstThreeChar.enabled()){
			for(int i = childPosition+1; i <= position; i++){
				String word = "*";
				if(i >= 0 && i <= input.size()-1)
				    word = input.get(i);
				word = word.length() >= 3 ? word.substring(0,3) : word;
		        features.add(param_g.toFeature(network, FeatureType.FirstThreeChar+"", tagId+"", word));
		        
			}
		}
		
		
		//FirstFourChar
		if(FeatureType.FirstFourChar.enabled()){
			for(int i = childPosition+1; i <= position; i++){
				String word = "*";
				if(i >= 0 && i <= input.size()-1)
				    word = input.get(i);
				word = word.length() >= 2 ? word.substring(0,2) : word;
		        features.add(param_g.toFeature(network, FeatureType.FirstFourChar+"", tagId+"", word));   
			}
		}
		
		
			
		//LastChar
		if(FeatureType.LastChar.enabled()){
			for(int i = childPosition+1; i <= position; i++){
				String word = "*";
				if(i >= 0 && i <= input.size()-1)
				    word = input.get(i);
		        features.add(param_g.toFeature(network, FeatureType.LastChar+"", tagId+"", word.charAt(word.length()-1)+""));      
			}
		}
		
		//LastTwoChar
		if(FeatureType.LastTwoChar.enabled()){
			for(int i = childPosition+1; i <= position; i++){
				String word = "*";
				if(i >= 0 && i <= input.size()-1)
				    word = input.get(i);
				if(word.length() >= 2)
					word = word.substring(word.length()- 2);
		        features.add(param_g.toFeature(network, FeatureType.LastTwoChar+"", tagId+"", word));
		       
			}
		}
		
		//LastThreeChar
		if(FeatureType.LastThreeChar.enabled()){
			for(int i = childPosition+1; i <= position; i++){
				String word = "*";
				if(i >= 0 && i <= input.size()-1)
				    word = input.get(i);
				word = word.length() >= 3 ? word.substring(word.length()-3) : word;
		        features.add(param_g.toFeature(network, FeatureType.LastThreeChar+"", tagId+"", word));
			}
		}
		
		//LastFourChar
		if(FeatureType.LastFourChar.enabled()){
			for(int i = childPosition+1; i <= position; i++){
				String word = "*";
				if(i >= 0 && i <= input.size()-1)
				    word = input.get(i);
				word = word.length() >= 4 ? word.substring(word.length()-4) : word;
		        features.add(param_g.toFeature(network, FeatureType.LastFourChar+"", tagId+"", word));     
			}
		}
       
		
		//LowerCase
		if(FeatureType.LowerCase.enabled()){
			for(int i = childPosition+1; i <= position; i++){
					String word = "*";
					if(i >=0 && i <= input.size()-1)
						word = input.get(i).replaceAll("[\\pP\\p{Punct}]", "").toLowerCase();
			        features.add(param_g.toFeature(network, FeatureType.LowerCase+":"+(i-childPosition-1), tagId+"", word));	
			}
		}
		
		
		//StartCapitalFeature
		if(FeatureType.StartCapitalFeature.enabled()){
			for(int j = -window; j <= window; j++){
				String word = "a";
				if(childPosition+1+j >= 0 && childPosition+1+j <= input.size()-1)
				    word = input.get(childPosition+1+j).replaceAll("[\\pP\\p{Punct}]", "");
				String symbol = "";
				boolean firstCapital = false;
				int capitalNum = 0;
				
				for(int m = 0; m < word.length(); m++){
					char character = word.charAt(m);
					if(m == 0 && Character.isUpperCase(character)){
						firstCapital = true;
						capitalNum++;
					} else if(Character.isUpperCase(character)){
						capitalNum++;
					}
				}
				if(capitalNum == word.length())
					symbol = "AllCap";
				else if(capitalNum == 1 && firstCapital == true && word.length() == 1)
					symbol = "SingleCap";
				else if(capitalNum == 1 && firstCapital == true && word.length() > 1)
					symbol = "InitCap";
				else 
					symbol = "others";
		        features.add(param_g.toFeature(network, FeatureType.StartCapitalFeature+":"+j, tagId+"", symbol));
		        
			}
	    }
		
		
		//EndCapitalFeature
		if(FeatureType.EndCapitalFeature.enabled()){
			for(int j = -window; j <= window; j++){
				String word = "a";
				if(position+j >= 0 && position+j <= input.size()-1)
				    word = input.get(position+j).replaceAll("[\\pP\\p{Punct}]", "");
				String symbol = "";
				boolean firstCapital = false;
				int capitalNum = 0;
				
				for(int m = 0; m < word.length(); m++){
					char character = word.charAt(m);
					if(m == 0 && Character.isUpperCase(character)){
						firstCapital = true;
						capitalNum++;
					} else if(Character.isUpperCase(character)){
						capitalNum++;
					}
				}
				if(capitalNum == word.length())
					symbol = "AllCap";
				else if(capitalNum == 1 && firstCapital == true && word.length() == 1)
					symbol = "SingleCap";
				else if(capitalNum == 1 && firstCapital == true && word.length() > 1)
					symbol = "InitCap";
				else 
					symbol = "others";
		        features.add(param_g.toFeature(network, FeatureType.EndCapitalFeature+":"+j, tagId+"", symbol));
		        
			}
	    }
				
		
		//CapitalFeature
		if(FeatureType.CapitalFeature.enabled()){
			for(int i = childPosition+1; i <= position; i++){
				String word = "a";
				if(i >= 0 && i <= input.size()-1)
				    word = input.get(i).replaceAll("[\\pP\\p{Punct}]", "");
				String symbol = "";
				boolean firstCapital = false;
				int capitalNum = 0;
				
				for(int m = 0; m < word.length(); m++){
					char character = word.charAt(m);
					if(m == 0 && Character.isUpperCase(character)){
						firstCapital = true;
						capitalNum++;
					} else if(Character.isUpperCase(character)){
						capitalNum++;
					}
				}
				if(capitalNum == word.length())
					symbol = "AllCap";
				else if(capitalNum == 1 && firstCapital == true && word.length() == 1)
					symbol = "SingleCap";
				else if(capitalNum == 1 && firstCapital == true && word.length() > 1)
					symbol = "InitCap";
				else 
					symbol = "others";
		        features.add(param_g.toFeature(network, FeatureType.CapitalFeature+":"+(i-childPosition-1), tagId+"", symbol));
		        
			}
	    }		
		
		
		//BinaryCapitalFeature
		//int Thiswindow = 1;
		if(FeatureType.BinaryCapitalFeature.enabled()){
			for(int i = childPosition+1; i < position; i++){
				for(int j = 0; j <= 0; j++){
					String firstWord = "a";
					String secondWord = "a";
					if(i+j >= 0 && i+j <= input.size()-1)
					    firstWord = input.get(i+j).replaceAll("[\\pP\\p{Punct}]", "");
					if(i+j+1 >= 0 && i+j+1 <= input.size()-1)
						secondWord = input.get(i+j+1).replaceAll("[\\pP\\p{Punct}]", "");
					
					//firstWord Feature
					String symbol1 = "";
					boolean firstCapital = false;
					int capitalNum = 0;
					
					for(int m = 0; m < firstWord.length(); m++){
						char character = firstWord.charAt(m);
						if(m == 0 && character >= 'A' && character <= 'Z'){
							firstCapital = true;
							capitalNum++;
						} else if(character >= 'A' && character <= 'Z'){
							capitalNum++;
						}
					}
					if(capitalNum == firstWord.length())
						symbol1 = "AllCap";
					else if(capitalNum == 1 && firstCapital == true && firstWord.length() == 1)
						symbol1 = "SingleCap";
					else if(capitalNum == 1 && firstCapital == true && firstWord.length() > 1)
						symbol1 = "InitCap";
					else 
						symbol1 = "others";
					
					//secondWord Feature
					String symbol2 = "";
					firstCapital = false;
					capitalNum = 0;
					
					for(int m = 0; m < secondWord.length(); m++){
						char character = secondWord.charAt(m);
						if(m == 0 && character >= 'A' && character <= 'Z'){
							firstCapital = true;
							capitalNum++;
						} else if(character >= 'A' && character <= 'Z'){
							capitalNum++;
						}
					}
					if(capitalNum == secondWord.length())
						symbol2 = "AllCap";
					else if(capitalNum == 1 && firstCapital == true && secondWord.length() == 1)
						symbol2 = "SingleCap";
					else if(capitalNum == 1 && firstCapital == true && secondWord.length() > 1)
						symbol2 = "InitCap";
					else 
						symbol2 = "others";
			        features.add(param_g.toFeature(network, FeatureType.BinaryCapitalFeature+"", tagId+"", symbol1+" "+symbol2));
				}
			}
		}
		
		
		//StartNumFeature
		if(FeatureType.StartNumFeature.enabled()){
				for(int j = -window; j <= window; j++){
		            String symbol = "*";
		            if(childPosition+1+j >= 0 && childPosition+1+j <= input.size()-1)
		            	symbol = this.getNumFeature(input, childPosition+1+j);
					features.add(param_g.toFeature(network, FeatureType.StartNumFeature+":"+j, tagId+"", symbol));
				}
			
		}
		
		
		//EndNumFeature
		if(FeatureType.EndNumFeature.enabled()){
				for(int j = -window; j <= window; j++){
		            String symbol = "*";
		            if(position+j >= 0 && position+j <= input.size()-1)
		            	symbol = this.getNumFeature(input, position+j);
					features.add(param_g.toFeature(network, FeatureType.EndNumFeature+":"+j, tagId+"", symbol));
				}
			
		}
		
		
		//NumFeature
		if(FeatureType.NumFeature.enabled()){
			for(int i = childPosition+1; i <= position; i++){
	            String symbol = "*";
	            if(i >= 0 && i <= input.size()-1)
	            	symbol = this.getNumFeature(input, i);
				features.add(param_g.toFeature(network, FeatureType.NumFeature+":"+(i-childPosition-1), tagId+"", symbol));
			}
		
	    }		

		
		//possibelEditor
		if(FeatureType.PossibleEditor.enabled()){
			for(int i = childPosition+1; i <= position; i++){
				String word = input.get(i).toLowerCase();
				if(word.contains("editor")
						|| word.contains("edit")
						|| input.get(i).contains("eds"))
					features.add(param_g.toFeature(network, FeatureType.PossibleEditor+":", tagId+"", ""));		
			}
		}
		
		
		//possibleJournal
		if(FeatureType.PossibleJournal.enabled()){
			for(int i = childPosition+1; i <= position; i++){
				String word = input.get(i).toLowerCase();
				if(word.contains("journal") || word.contains("proceeding"))
					features.add(param_g.toFeature(network, FeatureType.PossibleJournal+":", tagId+"", ""));		
			}
		}
		
		
		//Punctuation
		if(FeatureType.PunctuationFeature.enabled()){
			for(int i = childPosition+1; i <= position; i++){
				String word = input.get(i).replaceAll("[a-zA-Z0-9]+", "");
				if(word.length() == 0)
					word = "k";
				char c = word.charAt(word.length()-1);
				if(c == ',')
					features.add(param_g.toFeature(network, FeatureType.PunctuationFeature+":", tagId+"", "contPunc"));
				else if(c == '-')
					features.add(param_g.toFeature(network, FeatureType.PunctuationFeature+":", tagId+"", "hyphen"));
			    else if(c == '.')
					features.add(param_g.toFeature(network, FeatureType.PunctuationFeature+":", tagId+"", "stopPunc"));
				else if(c == '(' || c == ')' || c == '{' || c == '}' || c == '[' || c == ']')
					features.add(param_g.toFeature(network, FeatureType.PunctuationFeature+":", tagId+"", "brace"));
			    else if(c == ':')
					features.add(param_g.toFeature(network, FeatureType.PunctuationFeature+":", tagId+"", ":"));
			    else if(c =='&')
					features.add(param_g.toFeature(network, FeatureType.PunctuationFeature+":", tagId+"", "&"));
				else 
					features.add(param_g.toFeature(network, FeatureType.PunctuationFeature+":"+(i-childPosition-1), tagId+"", "others"));
			}
		}
		
	
		

		
		//StartBoundaryWord
		if(FeatureType.StartBoundaryWord.enabled()){
			if(childPosition != -1)
			    features.add(param_g.toFeature(network, FeatureType.StartBoundaryWord+"", tagId+"", input.get(childPosition).replaceAll("[\\pP\\p{Punct}]+", "")+" "+input.get(childPosition+1).replaceAll("[\\pP\\p{Punct}]+", "")));
			else
				features.add(param_g.toFeature(network, FeatureType.StartBoundaryWord+"", tagId+"", "* "+input.get(childPosition+1).replaceAll("[\\pP\\p{Punct}]+", "")));
		}
		
		
		//EndBoundaryWord
		if(FeatureType.EndBoundaryWord.enabled()){
			if(position+1 != input.size())
				features.add(param_g.toFeature(network, FeatureType.EndBoundaryWord+"", tagId+"", input.get(position).replaceAll("[\\pP\\p{Punct}]+", "")+" "+input.get(position+1).replaceAll("[\\pP\\p{Punct}]+", "")));
			else
				features.add(param_g.toFeature(network, FeatureType.EndBoundaryWord+"", tagId+"", input.get(position).replaceAll("[\\pP\\p{Punct}]+", "")+" *"));
		}
		
		
		//PossibleVol
		if(FeatureType.PossibleVol.enabled()){
			for(int i = childPosition+1; i<=position; i++){
				String word = input.get(i);
				if(word.toLowerCase().contains("vol"))
					features.add(param_g.toFeature(network, FeatureType.PossibleVol+":"+(i-childPosition-1), tagId+"", "y"));
				else
					features.add(param_g.toFeature(network, FeatureType.PossibleVol+":"+(i-childPosition-1), tagId+"", "n"));
			}
		}


		
		

		
		
		
		
		int[] featureArray = new int[features.size()];
		for(int i=0; i<featureArray.length; i++){
			featureArray[i] = features.get(i);
		}
		return new FeatureArray(featureArray);
	}

	
	//i is the position
	public String getNumFeature(ArrayList<String> input, int i){
		String word = input.get(i);
		String symbol = "";
		
		Boolean isFeatured = false;
		Pattern pattern = Pattern.compile("[0-9]+");
		Matcher m = pattern.matcher(word);
		if(!m.find()){
			isFeatured = true;
			symbol = "nonNum";
		}
		
		//map (123123)
		if(isFeatured == false){
			pattern = Pattern.compile("\\([0-9]+\\)");
			m = pattern.matcher(word);
			if(m.find()){
				String str = m.group().replaceAll("\\(", "").replaceAll("\\)", "").trim();
				int num = Integer.parseInt(str);
				if(num >= 1800 && num <= 2005){
					symbol = "year";
					isFeatured = true;
				} else {
					symbol = "possibleVol";
					isFeatured = true;
				}
			}
		}
		
        //map year
		if(isFeatured == false){
			pattern = Pattern.compile("[0-9]{4}");
			m = pattern.matcher(word);
			if(m.find()){
				pattern = Pattern.compile("[0-9]+");
    			Matcher m2 = pattern.matcher(word);
    			m2.find();
    			if(m2.group().length() == 4){
    				String str = m.group();
    				int num = Integer.parseInt(str);
    				if(num >= 1700 && num <= 2010){
    					symbol = "year";
    					isFeatured = true;
    				} 
    			}
			}
		}
		
		//map page an tech
		if(isFeatured == false){
			String temp[] = word.split("-");
			if(temp.length == 2){
				String tt[] = temp[0].split("-");
				boolean flag = true;
				pattern = Pattern.compile("[a-zA-Z]+");
    			m = pattern.matcher(tt[0]);
    			if(m.find()){
    				flag = false;
    			}
				if(flag == true){
					symbol = "possiblePage";
				} else {
					symbol = "possibleTech";
				}
				isFeatured = true;
			}  else if(temp.length >= 2){
				symbol = "possibleTech";
				isFeatured = true;
			}
		}
		
		if(isFeatured == false){
			Pattern p = Pattern.compile("[0-9]+th\\.*");// \u67e5\u627e\u89c4\u5219\u516c\u5f0f\u4e2d\u5927\u62ec\u53f7\u4ee5\u5185\u7684\u5b57\u7b26
		    m = p.matcher(word);
		    if(m.find()){
		    	symbol = "ordinal";
		    	isFeatured = true;
		    }
		    p = Pattern.compile("[0-9]+st\\.*");// \u67e5\u627e\u89c4\u5219\u516c\u5f0f\u4e2d\u5927\u62ec\u53f7\u4ee5\u5185\u7684\u5b57\u7b26
		    m = p.matcher(word);
		    if(m.find()){
		    	symbol = "ordinal";
		    	isFeatured = true;
		    }
		    p = Pattern.compile("[0-9]+nd\\.*");// \u67e5\u627e\u89c4\u5219\u516c\u5f0f\u4e2d\u5927\u62ec\u53f7\u4ee5\u5185\u7684\u5b57\u7b26
		    m = p.matcher(word);
		    if(m.find()){
		    	symbol = "ordinal";
		    	isFeatured = true;
		    }
		    p = Pattern.compile("[0-9]+rd\\.*");// \u67e5\u627e\u89c4\u5219\u516c\u5f0f\u4e2d\u5927\u62ec\u53f7\u4ee5\u5185\u7684\u5b57\u7b26
		    m = p.matcher(word);
		    if(m.find()){
		    	symbol = "ordinal";
		    	isFeatured = true;
		    }
		}
		
		if(isFeatured == false){
			pattern = Pattern.compile("[0-9]+");
			m = pattern.matcher(word);
			m.find();
			String tempWord = m.group();
			if(tempWord.length() <= word.length()){
				pattern = Pattern.compile("[a-zA-Z]+");
    			m = pattern.matcher(word);
    			if(m.find()){
    				symbol = "hasDig";
    			} else {
    				if(tempWord.length() == 1){
    					symbol = "1dig";
	    			} else if(tempWord.length() == 2){
	    				symbol = "2dig";
	    			} else if(tempWord.length() == 3){
	    				symbol = "3dig";
	    			} else symbol = "4+dig";
    			}
			} 
		}
		return symbol;
}
	
	public String sen2str(ArrayList<String> k){
		String tmp = "";
		for(int i = 0; i < k.size(); i++){
			tmp+= k.get(i)+" ";
		}
		tmp = tmp.trim();
		return tmp;
	}

}