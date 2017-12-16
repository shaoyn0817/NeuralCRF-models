/** Statistical Natural Language Processing System
    Copyright (C) 2014-2016  Lu, Wei

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
/**
 * 
 */
package org.statnlp.example.linear_crf;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import org.statnlp.commons.types.Label;
import org.statnlp.commons.types.LinearInstance;
import org.statnlp.example.base.BaseNetwork;
import org.statnlp.example.linear_crf.LinearCRFNetworkCompiler.NODE_TYPES;
import org.statnlp.hypergraph.FeatureArray;
import org.statnlp.hypergraph.FeatureManager;
import org.statnlp.hypergraph.GlobalNetworkParam;
import org.statnlp.hypergraph.Network;
import org.statnlp.hypergraph.NetworkConfig;
import org.statnlp.hypergraph.NetworkIDMapper;
import org.statnlp.hypergraph.neural.MultiLayerPerceptron;
import org.statnlp.util.Pipeline;
import org.statnlp.util.instance_parser.DelimiterBasedInstanceParser;
import org.statnlp.util.instance_parser.InstanceParser;

/**
 * @author Aldrian Obaja (aldrianobaja.m@gmail.com)
 */
public class LinearCRFFeatureManager extends FeatureManager{

	private static final long serialVersionUID = -4880581521293400351L;
	
	private String IN_SEP = MultiLayerPerceptron.IN_SEP;
	private String OUT_SEP = MultiLayerPerceptron.OUT_SEP;
	// TODO: Update the extract_helper with the net
	
	private static final boolean CHEAT = false;
	
	public int wordHalfWindowSize = 1;
	public int posHalfWindowSize = -1;
	public boolean productWithOutput = true;
	public Map<Integer, Label> labels;
	public Map<FeatureType, Boolean> featureTypes;
	
	public enum FeatureType {
		WORD(true),
		WORD_BIGRAM(false),
		TAG(false),
		TAG_BIGRAM(false),
		TRANSITION(true),
		LABEL(false),
		neural(false),
		;
		
		private boolean enabledByDefault;
		
		private FeatureType(){
			this(true);
		}
		
		private FeatureType(boolean enabledByDefault){
			this.enabledByDefault = enabledByDefault;
		}
		
		public boolean enabledByDefault(){
			return enabledByDefault;
		}
		
		public boolean disabledByDefault(){
			return !enabledByDefault;
		}
		
	}
	
	public LinearCRFFeatureManager(GlobalNetworkParam param_g) {
		this(param_g, (InstanceParser)null, new LinearCRFConfig());
	}
	
	public LinearCRFFeatureManager(GlobalNetworkParam param_g, String[] args) {
		this(param_g, (InstanceParser)null, new LinearCRFConfig(args));
	}
	
	public LinearCRFFeatureManager(GlobalNetworkParam param_g, LinearCRFConfig config) {
		this(param_g, (InstanceParser)null, config);
	}
	
	public LinearCRFFeatureManager(GlobalNetworkParam param_g, Map<Integer, Label> labels) {
		this(param_g, labels, new LinearCRFConfig());
	}
	
	public LinearCRFFeatureManager(GlobalNetworkParam param_g, Map<Integer, Label> labels, String[] args){
		this(param_g, labels, new LinearCRFConfig(args));
	}
	
	public LinearCRFFeatureManager(GlobalNetworkParam param_g, Map<Integer, Label> labels, LinearCRFConfig config) {
		this(param_g, (InstanceParser)null, config);
		this.labels = labels;
	}
	
	public LinearCRFFeatureManager(GlobalNetworkParam param_g, InstanceParser instanceParser) {
		this(param_g, instanceParser, new LinearCRFConfig());
	}
	
	public LinearCRFFeatureManager(GlobalNetworkParam param_g, InstanceParser instanceParser, String[] args){
		this(param_g, instanceParser, new LinearCRFConfig(args));
	}
	
	public LinearCRFFeatureManager(GlobalNetworkParam param_g, InstanceParser instanceParser, LinearCRFConfig config) {
		super(param_g, instanceParser);
		if(instanceParser != null){
			this.labels = ((DelimiterBasedInstanceParser)instanceParser).LABELS_INDEX;
		}
		wordHalfWindowSize = config.wordHalfWindowSize;
		posHalfWindowSize = config.posHalfWindowSize;
		productWithOutput = config.productWithOutput;
		featureTypes = new HashMap<FeatureType, Boolean>();
		if(config.features != null){
			for(FeatureType featureType: FeatureType.values()){
				disable(featureType);
			}
			for(String feat: config.features){
				enable(FeatureType.valueOf(feat.toUpperCase()));
			}
		}
	}

	public LinearCRFFeatureManager(Pipeline<?> pipeline){
		this(pipeline.param, pipeline.instanceParser);
	}

	/**
	 * Enables the specified feature type.
	 * @param featureType
	 */
	public void enable(FeatureType featureType){
		featureTypes.put(featureType, true);
	}
	
	/**
	 * Disables the specified feature type.
	 * @param featureType
	 */
	public void disable(FeatureType featureType){
		featureTypes.put(featureType, false);
	}
	
	/**
	 * Returns whether the specified feature type is enabled.
	 * @param featureType
	 * @return
	 */
	public boolean isEnabled(FeatureType featureType){
		return featureTypes.get(featureType);
	}
	
	@Override
	protected FeatureArray extract_helper(Network network, int parent_k, int[] children_k, int children_k_index) {
		GlobalNetworkParam param_g = this._param_g;
		
		BaseNetwork net = (BaseNetwork)network;
		
		@SuppressWarnings("unchecked")
		LinearInstance<String> instance = (LinearInstance<String>)net.getInstance();
		int size = instance.size();
		
		ArrayList<String[]> input = (ArrayList<String[]>)instance.getInput();
		
		long curNode = net.getNode(parent_k);
		int[] arr = NetworkIDMapper.toHybridNodeArray(curNode);
		
		int pos = arr[0]-1;
		int tag_id = arr[1];
		int nodeType = arr[4];
		if(!productWithOutput){
			tag_id = -1;
		}
		
		if(nodeType == NODE_TYPES.LEAF.ordinal()){
			return FeatureArray.EMPTY;
		}
		
		//long childNode = network.getNode(children_k[0]);
		int child_tag_id = network.getNodeArray(children_k[0])[1];
		int childNodeType = network.getNodeArray(children_k[0])[4];
		
		int labelSize = labels.size();

		if(childNodeType == NODE_TYPES.LEAF.ordinal()){
			child_tag_id = labelSize;
		}
		
		if(CHEAT){
			return new FeatureArray(new int[]{param_g.toFeature(net, "CHEAT", tag_id+"", Math.abs(instance.getInstanceId())+" "+pos+" "+child_tag_id)});
		}

		ArrayList<Integer> features = new ArrayList<Integer>();
		int prevIdx = pos - 1;
		int nextIdx = pos + 1;
		String prevWord = "STR";
		String nextWord ="END";
		String prevPos = "STR";
		if(nextIdx<input.size()-1) nextWord = input.get(nextIdx)[0];
		if(prevIdx>=0) {
			prevWord = input.get(prevIdx)[0]; 
			prevPos = input.get(prevIdx)[1];
		}
		
		if(NetworkConfig.USE_NEURAL_FEATURES){
			String postag = input.get(pos)[1];
//			features.add(param_g.toFeature(network, FeatureType.neural.name(), tag_id+"",input.get(pos)[0]));
			features.add(param_g.toFeature(network, FeatureType.neural.name(), tag_id+"", prevWord+IN_SEP+input.get(pos)[0]+IN_SEP+nextWord+OUT_SEP+prevPos+IN_SEP+postag));
		} else {
			// Word window features
			if(isEnabled(FeatureType.WORD) && tag_id != labelSize){
				int wordWindowSize = wordHalfWindowSize*2+1;
				if(wordWindowSize < 0){
					wordWindowSize = 0;
				}
				for(int i=0; i<wordWindowSize; i++){
					String word = "***";
					int relIdx = i-wordHalfWindowSize;
					int idx = pos + relIdx;
					if(idx >= 0 && idx < size){
						word = input.get(idx)[0];
					}
					if(idx > pos) continue; // Only consider the left window
					features.add(param_g.toFeature(network, FeatureType.WORD+":"+relIdx, tag_id+"", word));
				}
			}
		}
		
		// POS tag window features
		if(isEnabled(FeatureType.TAG) && tag_id != labelSize){
			int posWindowSize = posHalfWindowSize*2+1;
			if(posWindowSize < 0){
				posWindowSize = 0;
			}
			for(int i=0; i<posWindowSize; i++){
				String postag = "***";
				int relIdx = i-posHalfWindowSize;
				int idx = pos + relIdx;
				if(idx >= 0 && idx < size){
					postag = input.get(idx)[1];
				}
				features.add(param_g.toFeature(network, FeatureType.TAG+":"+relIdx, tag_id+"", postag));
			}
		}
		
		// Word bigram features
		if(isEnabled(FeatureType.WORD_BIGRAM)){
			for(int i=0; i<2; i++){
				String bigram = "";
				for(int j=0; j<2; j++){
					int idx = pos+i+j-1;
					if(idx >=0 && idx < size){
						bigram += input.get(idx)[0];
					} else {
						bigram += "***";
					}
					if(j==0){
						bigram += " ";
					}
				}
				features.add(param_g.toFeature(network, FeatureType.WORD_BIGRAM+":"+i, tag_id+"", bigram));
			}
		}
		
		// POS tag bigram features
		if(isEnabled(FeatureType.TAG_BIGRAM)){
			for(int i=0; i<2; i++){
				String bigram = "";
				for(int j=0; j<2; j++){
					int idx = pos+i+j-1;
					if(idx >=0 && idx < size){
						bigram += input.get(idx)[1];
					} else {
						bigram += "***";
					}
					if(j==0){
						bigram += " ";
					}
				}
				features.add(param_g.toFeature(network, FeatureType.TAG_BIGRAM+":"+i, tag_id+"", bigram));
			}
		}
		
		// Label feature
		if(isEnabled(FeatureType.LABEL)){
			int labelFeature = param_g.toFeature(network, FeatureType.LABEL.name(), tag_id+"", "");
			features.add(labelFeature);
		}
		FeatureArray featureArray = createFeatureArray(network, features);

		// Edge-based features
		features.clear();
		// Label transition feature
		if(isEnabled(FeatureType.TRANSITION)){
			if(tag_id != labelSize && child_tag_id != labelSize){
				int transitionFeature = param_g.toFeature(network, FeatureType.TRANSITION.name(), child_tag_id+" "+tag_id, "");
				features.add(transitionFeature);
			}
		}
		
		if(features.size() > 0){
			return createFeatureArray(network, features, featureArray);
		} else {
			return featureArray;
		}
	}

}
