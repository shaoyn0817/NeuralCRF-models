package org.statnlp.example.mention_hypergraphNoEmb;

import java.util.ArrayList;
import java.util.List;

import org.statnlp.example.mention_hypergraphNoEmb.MentionHypergraphNetworkCompiler.NodeType;
import org.statnlp.hypergraph.FeatureArray;
import org.statnlp.hypergraph.FeatureManager;
import org.statnlp.hypergraph.GlobalNetworkParam;
import org.statnlp.hypergraph.Network;
import org.statnlp.hypergraph.NetworkConfig;

import java.util.AbstractMap.SimpleImmutableEntry;

 

public class MentionHypergraphFeatureManager extends FeatureManager {

	private static final long serialVersionUID = 1359679442997276807L;
	private int wordWindowSize = 3;
	private int wordHalfWindowSize = 1;
	private int postagWindowSize = 3;
	private int postagHalfWindowSize = 1;
	
	private int wordNgramMinSize = 2;
	private int wordNgramMaxSize = 4;
	private int postagNgramMinSize = 2;
	private int postagNgramMaxSize = 4;
	
	private int bowWindowSize = 5;
	private int bowHalfWindowSize = 2;
 
	
	public enum FeatureType {
		WORD,
		WORD_NGRAM,
		
		POS_TAG,
		POS_TAG_NGRAM,
		
		BOW,
		
		ALL_CAPS,
		ALL_DIGITS,
		ALL_ALPHANUMERIC,
		ALL_LOWERCASE,
		CONTAINS_DIGITS,
		CONTAINS_DOTS,
		CONTAINS_HYPHEN,
		INITIAL_CAPS,
		LONELY_INITIAL,
		PUNCTUATION_MARK,
		ROMAN_NUMBER,
		SINGLE_CHARACTER,
		URL,
		
		MENTION_PENALTY,
	}

	public MentionHypergraphFeatureManager(GlobalNetworkParam param_g) {
		super(param_g);
	}
	
	@Override
	protected FeatureArray extract_helper(Network network, int parent_k, int[] children_k, int children_k_index) {
		//children_k_index is the hyper edge of the node
		MentionHypergraphInstance instance = (MentionHypergraphInstance)network.getInstance();
		AttributedWord[] words = instance.input.words;
		String[] posTags = instance.input.posTags;
		int size = instance.size();
		
		int[] parent_arr = network.getNodeArray(parent_k);
		int pos = size-parent_arr[0]-1;
		NodeType nodeType = NodeType.values()[parent_arr[1]];
		String labelId = String.valueOf(parent_arr[2]);
		int lid = Integer.valueOf(labelId);
		if(nodeType == NodeType.X_NODE || nodeType == NodeType.A_NODE || nodeType == NodeType.E_NODE ){
			return FeatureArray.EMPTY;
		}
		
		AttributedWord curWord = words[pos]; 
		List<Integer> features = new ArrayList<Integer>();
		GlobalNetworkParam param_g = this._param_g;
		
		
		
		String s = sen2str(words);

		
        boolean flag = true;
       
        
        if(pos != instance.size()){
		if(NetworkConfig.USE_NEURAL_FEATURES && flag){
			 
			Object input = null;
			if(MentionHypergraphMainBIO.neuralType.equals("lstm")) {
				input = new SimpleImmutableEntry<String, Integer>(s, pos);
			}
			else if(MentionHypergraphMainBIO.neuralType.equals("mlp")) {
				input = curWord.form;
			}
			else if(MentionHypergraphMainBIO.neuralType.equals("continuous")) {
				input = curWord.form;
			}
			int id = lid;
			NodeType childNodeType = NodeType.values()[network.getNodeArray(children_k[0])[1]];
			//input = new SimpleImmutableEntry<String, Integer>(s, pos);

			
			if(nodeType == NodeType.T_NODE && childNodeType == NodeType.X_NODE){
				//this.addNeural(network, 0, parent_k, children_k_index, input, id+Label.LABELS.size());
			}
			/*
			else if(nodeType == NodeType.T_NODE && childNodeType == NodeType.I_NODE){
				//this.addNeural(network, 0, parent_k, children_k_index, input, id);
			}
			else if(nodeType == NodeType.I_NODE && childNodeType == NodeType.X_NODE){
				if(children_k.length == 1)
				    {
					//this.addNeural(network, 0, parent_k, children_k_index, input, id+1*Label.LABELS.size());
				    }
				else{
					//this.addNeural(network, 0, parent_k, children_k_index, input, id+2*Label.LABELS.size());
				}
			}
			*/
			else
			   this.addNeural(network, 0, parent_k, children_k_index, input, id);
		}
        } 
        
	    boolean moreFeatures = true;
	   if(moreFeatures){
		for(int child_k: children_k){
			NodeType childNodeType = NodeType.values()[network.getNodeArray(child_k)[1]];
			
			if(nodeType == NodeType.T_NODE && childNodeType == NodeType.I_NODE){ 
				features.add(param_g.toFeature(network, FeatureType.MENTION_PENALTY.name(), "MP", "MP"));
			}
			
			String indicator = nodeType+" "+childNodeType;
			
			for(int idx=pos-wordHalfWindowSize; idx<=pos+wordHalfWindowSize; idx++){
				String word = "";
				if(idx >= 0 && idx < size){
					word = words[idx].form;
				}
				features.add(param_g.toFeature(network, indicator+FeatureType.WORD.name()+(idx-pos), labelId, word));
			}
			
			for(int idx=pos-postagHalfWindowSize; idx<=pos+postagHalfWindowSize; idx++){
				String postag = "";
				if(idx >= 0 && idx < size){
					postag = posTags[idx];
				}
				features.add(param_g.toFeature(network, indicator+FeatureType.POS_TAG.name()+(idx-pos), labelId, postag));
			}
			
			for(int ngramSize=wordNgramMinSize; ngramSize<=wordNgramMaxSize; ngramSize++){
				for(int relPos=0; relPos<ngramSize; relPos++){
					String ngram = "";
					for(int idx=pos-ngramSize+relPos+1; idx<pos+relPos+1; idx++){
						if(ngram.length() > 0) ngram += " ";
						if(idx >= 0 && idx < size){
							ngram += words[idx];
						}
					}
					features.add(param_g.toFeature(network, indicator+FeatureType.WORD_NGRAM+" "+ngramSize+" "+relPos, labelId, ngram));
				}
			}
			for(int ngramSize=postagNgramMinSize; ngramSize<=postagNgramMaxSize; ngramSize++){
				for(int relPos=0; relPos<ngramSize; relPos++){
					String ngram = "";
					for(int idx=pos-ngramSize+relPos+1; idx<pos+relPos+1; idx++){
						if(idx > pos-ngramSize+relPos+1) ngram += " ";
						if(idx >= 0 && idx < size){
							ngram += posTags[idx];
						}
					}
					features.add(param_g.toFeature(network, indicator+FeatureType.POS_TAG_NGRAM+" "+ngramSize+" "+relPos, labelId, ngram));
				}
			}
			
			for(int i = pos- bowHalfWindowSize; i  <= pos + bowHalfWindowSize;i ++){
			    if(i < 0)
			    	features.add(param_g.toFeature(network, indicator+FeatureType.BOW.name(), labelId, "@be@"));
			    else if(i >= words.length)
			    	features.add(param_g.toFeature(network, indicator+FeatureType.BOW.name(), labelId, "@en@"));
			    else
				    features.add(param_g.toFeature(network, indicator+FeatureType.BOW.name(), labelId, words[i].form));
			}
			
			for(FeatureType featureType: FeatureType.values()){
				switch(featureType){
				case ALL_CAPS:
				case ALL_DIGITS:
				case ALL_ALPHANUMERIC:
				case ALL_LOWERCASE:
				case CONTAINS_DIGITS:
				case CONTAINS_DOTS:
				case CONTAINS_HYPHEN:
				case INITIAL_CAPS:
				case LONELY_INITIAL:
				case PUNCTUATION_MARK:
				case ROMAN_NUMBER:
				case SINGLE_CHARACTER:
				case URL:
					features.add(param_g.toFeature(network, indicator+featureType.name(), labelId, curWord.getAttribute(featureType.name())));
				default:
					break;
				}
			}
			
		}
	}
		
		
		int[] featuresInt = new int[features.size()];
		for(int i=0; i<featuresInt.length; i++){
			featuresInt[i] = features.get(i);
		}
		
		FeatureArray result = new FeatureArray(featuresInt);
		return result;
	}
	
	public void setWordWindowSize(int windowSize){
		this.wordWindowSize = windowSize;
		this.wordHalfWindowSize = (windowSize-1)/2;
	}
	
	public void setBowWindowSize(int windowSize){
		this.bowWindowSize = windowSize;
		this.bowHalfWindowSize = (windowSize-1)/2;
	}
	
	public void setWordNgramSize(int minSize, int maxSize){
		this.wordNgramMinSize = minSize;
		this.wordNgramMaxSize = maxSize;
	}
	
	public void setPostagNgramSize(int minSize, int maxSize){
		this.postagNgramMinSize = minSize;
		this.postagNgramMaxSize = maxSize;
	}
	
	public String getConfig(){
		StringBuilder builder = new StringBuilder();
		builder.append(String.format("Word window size: %d\n", wordWindowSize));
		builder.append(String.format("Word Ngram size: %d-%d\n", wordNgramMinSize, wordNgramMaxSize));
		builder.append(String.format("POS Tag window size: %d\n", postagWindowSize));
		builder.append(String.format("POS Tag Ngram size: %d-%d\n", postagNgramMinSize, postagNgramMaxSize));
		builder.append(String.format("BOW window size: %d\n", bowWindowSize));
		return builder.toString();
	}

	
	public String sen2str(AttributedWord[] k){
		String tmp = "";
		for(int i = 0; i < k.length; i++){
			tmp+= k[i].form+" ";
		}
		tmp = tmp.trim();
		return tmp;
	}
}
