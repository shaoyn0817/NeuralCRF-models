package org.statnlp.example.tagging;

import java.util.AbstractMap.SimpleImmutableEntry;
import java.util.ArrayList;
import java.util.List;

import org.statnlp.commons.types.Sentence;
import org.statnlp.example.tagging.TagNetworkCompiler.NodeType;
import org.statnlp.hypergraph.FeatureArray;
import org.statnlp.hypergraph.FeatureManager;
import org.statnlp.hypergraph.GlobalNetworkParam;
import org.statnlp.hypergraph.Network;
import org.statnlp.hypergraph.NetworkConfig;
import org.statnlp.util.instance_parser.InstanceParser;

public class TagFeatureManager extends FeatureManager {

	private static final long serialVersionUID = -6059629463406022487L;

	private enum FeaType {
		unigram, bigram, transition
	}
	
	public TagFeatureManager(GlobalNetworkParam param_g) {
		super(param_g);
	}

	public TagFeatureManager(GlobalNetworkParam param_g, InstanceParser instanceParser) {
		super(param_g, instanceParser);
	}

	@Override
	protected FeatureArray extract_helper(Network network, int parent_k, int[] children_k, int children_k_index) {
		
		int[] paArr = network.getNodeArray(parent_k);
		if (NodeType.values()[paArr[2]] == NodeType.leaf || NodeType.values()[paArr[2]] == NodeType.root)
			return FeatureArray.EMPTY;
		List<Integer> fs = new ArrayList<>();
		TagInstance inst = (TagInstance) network.getInstance();
		Sentence sent = inst.getInput();
		
		int pos = paArr[0];
		int labelId = paArr[1];
		String output = labelId + "";
		String word = sent.get(pos).getForm();
		String lw = pos - 1 >= 0 ? sent.get(pos - 1).getForm() : "START";
		String rw = pos + 1 < sent.length() ? sent.get(pos + 1).getForm() : "END";
		
		fs.add(this._param_g.toFeature(network, FeaType.unigram.name(), output, word));
		fs.add(this._param_g.toFeature(network, FeaType.unigram.name() + "-left", output, lw));
		fs.add(this._param_g.toFeature(network, FeaType.unigram.name() + "-right", output, rw));
		
		fs.add(this._param_g.toFeature(network, FeaType.bigram.name() + "-1", output, lw + " " + word));
		fs.add(this._param_g.toFeature(network, FeaType.bigram.name() + "0", output, word + " " + rw));
		
		if (NetworkConfig.USE_NEURAL_FEATURES) {
			//edgeInput:
			// MLP: word.
			//BiLSTM: position, sentence
			SimpleImmutableEntry<String, Integer> edgeInput = 
					new SimpleImmutableEntry<String, Integer>(sent.toString(), pos);
			this.addNeural(network, 0, parent_k, children_k_index, edgeInput, labelId);
		}
		
		int[] childArr = network.getNodeArray(children_k[0]);
		NodeType childNodeType = NodeType.values()[childArr[2]];
		int childLabelId = childArr[1];
		String childLabel = childNodeType == NodeType.leaf ? "START" : childLabelId + "";
		fs.add(this._param_g.toFeature(network, FeaType.transition.name(), output, childLabel));
		return this.createFeatureArray(network, fs);
	}

}
