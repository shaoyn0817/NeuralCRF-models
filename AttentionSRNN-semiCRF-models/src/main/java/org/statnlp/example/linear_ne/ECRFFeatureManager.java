package org.statnlp.example.linear_ne;

import java.util.AbstractMap.SimpleImmutableEntry;

import org.statnlp.commons.types.Sentence;
import org.statnlp.example.linear_ne.ECRFNetworkCompiler.NodeType;
import org.statnlp.hypergraph.FeatureArray;
import org.statnlp.hypergraph.FeatureManager;
import org.statnlp.hypergraph.GlobalNetworkParam;
import org.statnlp.hypergraph.Network;
import org.statnlp.hypergraph.NetworkConfig;
import org.statnlp.hypergraph.NetworkIDMapper;
import org.statnlp.hypergraph.neural.MultiLayerPerceptron;

import java.util.ArrayList;

public class ECRFFeatureManager extends FeatureManager {

	private static final long serialVersionUID = 376931974939202432L;

	public enum FeaType{
		word, tag, lw, lt, ltt, rw, rt, prefix, suffix,
		transition};
	private String OUT_SEP = MultiLayerPerceptron.OUT_SEP; 
	private String IN_SEP = MultiLayerPerceptron.IN_SEP;
	private final String START = "STR";
	private final String END = "END";
	
	private String neuralType;
	private boolean moreBinaryFeatures = false;
	private boolean lowercase = true;
	private int maxmimumPrefixSUffixLength = 6;
	private Entity[] labels;
	
	public ECRFFeatureManager(GlobalNetworkParam param_g, Entity[] labels, String neuralType, boolean moreBinaryFeatures, boolean lowercase) {
		super(param_g);
		this.neuralType = neuralType;
		this.labels = labels;
		this.moreBinaryFeatures = moreBinaryFeatures;
		this.lowercase = lowercase;
	}
	
	@Override
	protected FeatureArray extract_helper(Network network, int parent_k, int[] children_k, int children_k_index) {
		
		ECRFInstance inst = ((ECRFInstance)network.getInstance());
		Sentence sent = inst.getInput();
		long node = network.getNode(parent_k);
		int[] nodeArr = NetworkIDMapper.toHybridNodeArray(node);
		
		FeatureArray fa = null;
		ArrayList<Integer> featureList = new ArrayList<Integer>();
		
		int pos = nodeArr[0];
		int eId = nodeArr[1];
		ECRFNetworkCompiler.NodeType nodeType = NodeType.values()[nodeArr[2]];
		if (nodeType == NodeType.Leaf) return FeatureArray.EMPTY;
		if (nodeType == NodeType.Root && pos != inst.size() - 1) return FeatureArray.EMPTY;
		String entity = eId + ""; 
		int[] child = NetworkIDMapper.toHybridNodeArray(network.getNode(children_k[0]));
		int childEId = child[1];
		
		String prevEntity =  childEId + "";
		featureList.add(this._param_g.toFeature(network, FeaType.transition.name(), entity,  prevEntity));
		
		if (nodeType == NodeType.Root) {
			return this.createFeatureArray(network, featureList);
		}
		
		String word = pos != inst.size() ? inst.getInput().get(pos).getForm() : END;
		String tag = pos != inst.size() ?  inst.getInput().get(pos).getTag() : END;
		String lw = pos > 0 ? sent.get(pos-1).getForm() : START;
		String llw = pos > 1 ? sent.get(pos-2).getForm(): START;
		String llt = pos > 1 ? sent.get(pos-2).getTag(): START;
		String lt = pos > 0 ? sent.get(pos-1).getTag():START;
		String rw = pos < sent.length()-1? sent.get(pos + 1).getForm() : END;
		String rt = pos < sent.length()-1? sent.get(pos + 1).getTag() :END;
		String rrw = pos < sent.length()-2? sent.get(pos + 2).getForm() :END;
		String rrt = pos < sent.length()-2? sent.get(pos + 2).getTag() :END;
		
		if(NetworkConfig.USE_NEURAL_FEATURES){
			if (eId != 0 && eId != this.labels.length - 1) {
				Object input = null;
				if(neuralType.equals("lstm")) {
					String sentenceInput = this.lowercase ? sent.toString().toLowerCase() : sent.toString();
					input = new SimpleImmutableEntry<String, Integer>(sentenceInput, pos);
				} else if(neuralType.equals("mlp")){
					input = llw+IN_SEP+lw+IN_SEP+word+IN_SEP+rw+IN_SEP+rrw+OUT_SEP+llt+IN_SEP+lt+IN_SEP+tag+IN_SEP+rt+IN_SEP+rrt;
					//for simplicity, just use the current word.
					input = word;
				} else {
					input = word;
				}
				//since we don't want to use start tag. later need to add one.
				this.addNeural(network, 0, parent_k, children_k_index, input, eId - 1);
			}
		} else {
			featureList.add(this._param_g.toFeature(network,FeaType.word.name(), entity, word));
		}
		
		if (moreBinaryFeatures) {
			featureList.add(this._param_g.toFeature(network,FeaType.tag.name(), entity,	tag));
			featureList.add(this._param_g.toFeature(network,FeaType.lw.name(), 	entity,	lw));
			featureList.add(this._param_g.toFeature(network,FeaType.lt.name(), 	entity,	lt));
			featureList.add(this._param_g.toFeature(network,FeaType.rw.name(), 	entity,	rw));
			featureList.add(this._param_g.toFeature(network,FeaType.rt.name(), 	entity,	rt));
			featureList.add(this._param_g.toFeature(network,FeaType.ltt.name(), entity,	lt+","+tag));
			/****Add some prefix features******/
			for(int plen = 1;plen <= maxmimumPrefixSUffixLength; plen++){
				if(word.length()>=plen){
					String suff = word.substring(word.length()-plen, word.length());
					featureList.add(this._param_g.toFeature(network,FeaType.suffix.name()+plen, entity, suff));
					String pref = word.substring(0,plen);
					featureList.add(this._param_g.toFeature(network,FeaType.prefix.name()+plen, entity, pref));
				}
			}
			featureList.add(this._param_g.toFeature(network,FeaType.transition.name()+"-wle", entity, word+":"+prevEntity));
			featureList.add(this._param_g.toFeature(network,FeaType.transition.name()+"lwle", entity,	lw+":"+prevEntity));
			featureList.add(this._param_g.toFeature(network,FeaType.transition.name()+"rwre", entity,	rw+":"+prevEntity));
			
			featureList.add(this._param_g.toFeature(network,FeaType.transition.name()+"tle", 	entity,tag + ":" + prevEntity));
			featureList.add(this._param_g.toFeature(network,FeaType.transition.name()+"ltle", entity,lt + ":" + prevEntity));
			featureList.add(this._param_g.toFeature(network,FeaType.transition.name()+"rtle", entity,rt + ":" + prevEntity));
			featureList.add(this._param_g.toFeature(network,FeaType.transition.name()+"lttle",entity,lt + ":" + tag + ":" + prevEntity));
		}
		fa = this.createFeatureArray(network, featureList);
		return fa;
	}
}
