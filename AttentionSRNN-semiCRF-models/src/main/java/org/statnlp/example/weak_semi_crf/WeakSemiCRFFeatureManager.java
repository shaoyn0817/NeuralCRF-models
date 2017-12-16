package org.statnlp.example.weak_semi_crf;

import java.util.ArrayList;
import java.util.List;

import org.statnlp.commons.types.LinearInstance;
import org.statnlp.commons.types.Span;
import org.statnlp.example.base.BaseNetwork;
import org.statnlp.example.weak_semi_crf.WeakSemiCRFNetworkCompiler.NodeType;
import org.statnlp.hypergraph.FeatureArray;
import org.statnlp.hypergraph.FeatureManager;
import org.statnlp.hypergraph.GlobalNetworkParam;
import org.statnlp.hypergraph.Network;
import org.statnlp.util.Pipeline;

public class WeakSemiCRFFeatureManager extends FeatureManager {
	
	private static final long serialVersionUID = 6510131496948610905L;
	
	private static final boolean CHEAT = false;

	public enum FeatureType{
		CHEAT,
		
		SEGMENT,
		START_CHAR,
		END_CHAR,
		
		UNIGRAM,
		SUBSTRING,

		ENDS_WITH_SPACE,
		NUM_SPACES,
		
		PREV_WORD,
		START_BOUNDARY_WORD,
		WORDS,
		END_BOUNDARY_WORD,
		NEXT_WORD,
		
		BIGRAM,
	}
	
	public int unigramWindowSize = 5;
	public int substringWindowSize = 5;

	public WeakSemiCRFFeatureManager(GlobalNetworkParam param_g) {
		super(param_g);
	}
	
	public WeakSemiCRFFeatureManager(Pipeline<?> pipeline){
		this(pipeline.param);
	}
	
	@Override
	protected FeatureArray extract_helper(Network net, int parent_k, int[] children_k, int children_k_index) {
		BaseNetwork network = (BaseNetwork)net;
		@SuppressWarnings("unchecked")
		LinearInstance<Span> instance = (LinearInstance<Span>)network.getInstance();
		
		int[] parent_arr = network.getNodeArray(parent_k);
		int parentPos = parent_arr[0];
		NodeType parentType = NodeType.values()[parent_arr[1]];
		int parentLabelId = parent_arr[2];
		
		if(parentType == NodeType.LEAF || children_k.length == 0){
			return FeatureArray.EMPTY;
		}
		
		int[] child_arr = network.getNodeArray(children_k[0]);
		int childPos = child_arr[0];
		NodeType childType = NodeType.values()[child_arr[1]];
		int childLabelId = child_arr[2];

		GlobalNetworkParam param_g = this._param_g;
		int bigramFeature = param_g.toFeature(network, FeatureType.BIGRAM.name(), parentLabelId+"", parentLabelId+" "+childLabelId);
		if(parentType == NodeType.ROOT || childType == NodeType.LEAF){
			return createFeatureArray(network, new int[]{bigramFeature});
		}
		
		if(CHEAT){
			int instanceId = Math.abs(instance.getInstanceId());
			int cheatFeature = param_g.toFeature(network, FeatureType.CHEAT.name(), parentLabelId+"", instanceId+" "+parentPos+" "+childPos+" "+parentLabelId+" "+childLabelId);
			return createFeatureArray(network, new int[]{cheatFeature});
		}
		
		List<String[]> inputTokenized = instance.input;
		String input = "";
		String[] inputArr = new String[inputTokenized.size()];
		for(int i=0; i<inputTokenized.size(); i++){
			String[] inputToken = inputTokenized.get(i);
			input += inputToken[0];
			inputArr[i] = inputToken[0];
		}
		String segment = input.substring(childPos, parentPos);
		int length = input.length();
		int isSpaceFeature = param_g.toFeature(network, FeatureType.ENDS_WITH_SPACE.name(), parentLabelId+"", (inputArr[parentPos].equals(" "))+"");
		int startCharFeature = param_g.toFeature(network, FeatureType.START_CHAR.name(), parentLabelId+"", inputArr[childPos]);
		int endCharFeature = param_g.toFeature(network, FeatureType.END_CHAR.name(), parentLabelId+"", inputArr[parentPos]);
		int segmentFeature = param_g.toFeature(network, FeatureType.SEGMENT.name(), parentLabelId+"", segment);
		
		String[] words = segment.split(" ");
		int numSpaces = words.length-1;
		int numSpacesFeature = param_g.toFeature(network, FeatureType.NUM_SPACES.name(), parentLabelId+"", numSpaces+"");
		
		int prevSpaceIdx = input.lastIndexOf(' ', childPos-1);
		if(prevSpaceIdx == -1){
			prevSpaceIdx = 0;
		}
		int firstSpaceIdx = input.indexOf(' ', childPos);
		if(firstSpaceIdx == -1){
			firstSpaceIdx = prevSpaceIdx;
		}
		int prevWordFeature = param_g.toFeature(network, FeatureType.PREV_WORD.name(), parentLabelId+"", input.substring(prevSpaceIdx, childPos));
		int startBoundaryWordFeature = param_g.toFeature(network, FeatureType.START_BOUNDARY_WORD.name(), parentLabelId+"", input.substring(prevSpaceIdx, firstSpaceIdx));
		
		int nextSpaceIdx = input.indexOf(' ', parentPos+1);
		if(nextSpaceIdx == -1){
			nextSpaceIdx = length;
		}
		int lastSpaceIdx = input.lastIndexOf(' ', parentPos);
		if(lastSpaceIdx == -1){
			lastSpaceIdx = nextSpaceIdx;
		}
		int nextWordFeature = param_g.toFeature(network, FeatureType.NEXT_WORD.name(), parentLabelId+"", input.substring(parentPos+1, nextSpaceIdx));
		int endBoundaryWordFeature = param_g.toFeature(network, FeatureType.END_BOUNDARY_WORD.name(), parentLabelId+"", input.substring(lastSpaceIdx, nextSpaceIdx));
		
		ArrayList<Integer> edgeFeatures = new ArrayList<Integer>();
		ArrayList<Integer> nodeStartFeatures = new ArrayList<Integer>();
		ArrayList<Integer> nodeEndFeatures = new ArrayList<Integer>();
		edgeFeatures.add(bigramFeature);
		nodeEndFeatures.add(isSpaceFeature);
		nodeStartFeatures.add(startCharFeature);
		nodeEndFeatures.add(endCharFeature);
		edgeFeatures.add(segmentFeature);
		edgeFeatures.add(numSpacesFeature);
		nodeStartFeatures.add(prevWordFeature);
		nodeEndFeatures.add(nextWordFeature);
		nodeStartFeatures.add(startBoundaryWordFeature);
		nodeEndFeatures.add(endBoundaryWordFeature);
		
		
		int[] wordFeatures = new int[2*words.length];
		for(int i=0; i<words.length; i++){
			wordFeatures[i] = param_g.toFeature(network, FeatureType.WORDS.name()+i, parentLabelId+"", words[i]);
			wordFeatures[2*words.length-i-1] = param_g.toFeature(network, FeatureType.WORDS.name()+"-"+i, parentLabelId+"", words[i]);
		}
		for(int feature: wordFeatures){
			edgeFeatures.add(feature);
		}
		
		int unigramFeatureSize = 2*unigramWindowSize;
		int[] unigramFeatures = new int[unigramFeatureSize];
		for(int i=0; i<unigramWindowSize; i++){
			String curInput = "";
			if(parentPos+i+1 < length){
				curInput = inputArr[parentPos+i+1];
			}
			unigramFeatures[i] = param_g.toFeature(network, FeatureType.UNIGRAM+":"+i, parentLabelId+"", curInput);
			curInput = "";
			if(childPos-i-1 >= 0){
				curInput = inputArr[childPos-i-1];
			}
			unigramFeatures[unigramFeatureSize-i-1] = param_g.toFeature(network, FeatureType.UNIGRAM+":-"+i, parentLabelId+"", curInput);
		}
		for(int feature: unigramFeatures){
			edgeFeatures.add(feature);
		}
		
		int substringFeatureSize = 2*substringWindowSize;
		int[] substringFeatures = new int[substringFeatureSize];
		for(int i=0; i<substringWindowSize; i++){
			String curInput = "";
			if(parentPos+i+1< length){
				curInput = input.substring(parentPos, parentPos+i+1);
			}
			substringFeatures[i] = param_g.toFeature(network, FeatureType.SUBSTRING+":"+i, parentLabelId+"", curInput);
			curInput = "";
			if(childPos-i-1 >= 0){
				curInput = input.substring(childPos-i-1, childPos);
			}
			substringFeatures[unigramFeatureSize-i-1] = param_g.toFeature(network, FeatureType.SUBSTRING+":-"+i, parentLabelId+"", curInput);
		}
		for(int feature: substringFeatures){
			edgeFeatures.add(feature);
		}
		
		FeatureArray featureArray = createFeatureArray(network, edgeFeatures);
		featureArray = createFeatureArray(network, nodeStartFeatures, featureArray);
		featureArray = createFeatureArray(network, nodeEndFeatures, featureArray);
		return featureArray;
	}

}
