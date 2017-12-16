package org.statnlp.example.tree_crf;

import org.statnlp.example.base.BaseNetwork;
import org.statnlp.example.tree_crf.TreeCRFNetworkCompiler.NodeType;
import org.statnlp.hypergraph.FeatureArray;
import org.statnlp.hypergraph.FeatureManager;
import org.statnlp.hypergraph.GlobalNetworkParam;
import org.statnlp.hypergraph.Network;
import org.statnlp.hypergraph.NetworkIDMapper;

public class TreeCRFFeatureManager extends FeatureManager {

	private static final long serialVersionUID = -4612307270890724830L;
	
	public enum FeatureType{
		LEFT_RIGHT,
//		LOWER_FIRST_WORD,
		FIRST_WORD,
		SPLIT_WORD,
		LAST_WORD,
		LAST_WORD_ENDING_1,
		LAST_WORD_ENDING_2,
		LAST_WORD_ENDING_3,
		FIRST_CAPITAL,
	}

	public TreeCRFFeatureManager(GlobalNetworkParam param_g) {
		super(param_g);
	}

	@Override
	protected FeatureArray extract_helper(Network net, int parent_k, int[] children_k, int children_k_index) {
		BaseNetwork network = (BaseNetwork)net;
		TreeCRFInstance instance = (TreeCRFInstance)network.getInstance();
		String[] words = instance.input;
		long parent = network.getNode(parent_k);
		int[] parentArr = NetworkIDMapper.toHybridNodeArray(parent);
		int height = parentArr[1];
		int index = parentArr[0]-height;
		int labelId = parentArr[2];
		int nodeType = parentArr[4];
		int start = index;
		int end = index+height;
		
		if(children_k.length == 0 || nodeType == NodeType.ROOT.ordinal()){
			return FeatureArray.EMPTY;
		}
		
		String startWord = normalize(words[start]);
		String endWord = normalize(words[end]);
		
		int firstWordFeature = this._param_g.toFeature(network, FeatureType.FIRST_WORD.name(), String.valueOf(labelId), startWord);
		int endWordLen = endWord.length();
		int lastWordFeature = this._param_g.toFeature(network, FeatureType.LAST_WORD.name(), String.valueOf(labelId), endWord);
		String last1 = endWordLen >= 1 ? endWord.substring(endWordLen-1) : endWord;
		String last2 = endWordLen >= 2 ? endWord.substring(endWordLen-2) : endWord;
		String last3 = endWordLen >= 3 ? endWord.substring(endWordLen-3) : endWord;
		int lastWordEnding1Feature = this._param_g.toFeature(network, FeatureType.LAST_WORD_ENDING_1.name(), String.valueOf(labelId), last1);
		int lastWordEnding2Feature = this._param_g.toFeature(network, FeatureType.LAST_WORD_ENDING_2.name(), String.valueOf(labelId), last2);
		int lastWordEnding3Feature = this._param_g.toFeature(network, FeatureType.LAST_WORD_ENDING_3.name(), String.valueOf(labelId), last3);
		boolean isFirstCapital = words[start].matches("[A-Z].*");
		int firstCapitalFeature = this._param_g.toFeature(network, FeatureType.FIRST_CAPITAL.name(), String.valueOf(labelId), String.valueOf(isFirstCapital));
//		int lowerFirstWordFeature = this._param_g.toFeature(FeatureType.LOWER_FIRST_WORD.name(), String.valueOf(labelId), words[start].toLowerCase());
		
		if(children_k.length == 1){ // Pre-Terminal nodes
			return createFeatureArray(network, new int[]{
											firstWordFeature,
											lastWordFeature,
											lastWordEnding1Feature,
											lastWordEnding2Feature,
											lastWordEnding3Feature,
											firstCapitalFeature,
//											lowerFirstWordFeature,
										});
		}
		
		long leftNode = network.getNode(children_k[0]);
		int[] leftArr = NetworkIDMapper.toHybridNodeArray(leftNode);
//		int leftHeight = leftArr[1];
//		int leftIndex = leftArr[0]-leftHeight;
		int leftLabelId = leftArr[2];
		
		long rightNode = network.getNode(children_k[1]);
		int[] rightArr = NetworkIDMapper.toHybridNodeArray(rightNode);
		int rightHeight = rightArr[1];
		int rightIndex = rightArr[0]-rightHeight;
		int rightLabelId = rightArr[2];
		int rightStart = rightIndex;
//		int rightEnd = rightIndex+rightHeight;
		
		String splitWord = normalize(words[rightStart]);

		int splitWordFeature = this._param_g.toFeature(network, FeatureType.SPLIT_WORD.name(), String.valueOf(labelId), splitWord);
		int leftRightFeature = this._param_g.toFeature(network, FeatureType.LEFT_RIGHT.name(), String.valueOf(labelId), leftLabelId+" "+rightLabelId);
		
		FeatureArray fa = createFeatureArray(network, new int[]{
													leftRightFeature,
													firstWordFeature,
													splitWordFeature,
													lastWordFeature,
													lastWordEnding1Feature,
													lastWordEnding2Feature,
													lastWordEnding3Feature,
													firstCapitalFeature,
//													lowerFirstWordFeature,
												});
		return fa;
	}
	
	private String normalize(String word){
		return word;
	}

}
