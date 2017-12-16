package org.statnlp.example.logstic_regression;

import java.util.ArrayList;
import java.util.List;

import org.statnlp.commons.types.Sentence;
import org.statnlp.example.logstic_regression.LRNetworkCompiler.NodeTypes;
import org.statnlp.hypergraph.FeatureArray;
import org.statnlp.hypergraph.FeatureManager;
import org.statnlp.hypergraph.GlobalNetworkParam;
import org.statnlp.hypergraph.Network;

public class LRFeatureManager extends FeatureManager {

	private static final long serialVersionUID = -7169027921164193697L;

	private enum FeaType {hm1, hm2, hm12}
	
	public LRFeatureManager(GlobalNetworkParam param_g) {
		super(param_g);
	}

	@Override
	protected FeatureArray extract_helper(Network network, int parent_k, int[] children_k, int children_k_index) {
		
		int[] paArr = network.getNodeArray(parent_k);
		int nodeType = paArr[2];
		if (nodeType != NodeTypes.NODE.ordinal())
			return FeatureArray.EMPTY;
		LRInstance inst = (LRInstance)network.getInstance();
		LGInput input = inst.getInput();
		List<Span> spans = input.spans;
		Sentence sent = input.sent;
		int leftIdx = input.leftSpanIdx;
		int rightIdx = input.rightSpanIdx;
		int relId = paArr[1];
		String rel = relId + "";
		String relForm = RelationType.get(relId).form;
		int arg1Idx = leftIdx;
		int arg2Idx = rightIdx;
		if (relForm.endsWith(Config.REV_SUFF)) {
			arg1Idx = rightIdx;
			arg2Idx = leftIdx;
		}
		Span arg1Span = spans.get(arg1Idx);
		Span arg2Span = spans.get(arg2Idx);
		List<Integer> fs = new ArrayList<>();
		
		int hm1Idx = this.getHeadIdx(sent, arg1Span);
		int hm2Idx = this.getHeadIdx(sent, arg2Span); 
		String hm1 = sent.get(hm1Idx).getForm();
		String hm2 = sent.get(hm2Idx).getForm();
		
		fs.add(this._param_g.toFeature(network, FeaType.hm1.name(), rel, hm1));
		fs.add(this._param_g.toFeature(network, FeaType.hm2.name(), rel, hm2));
		fs.add(this._param_g.toFeature(network, FeaType.hm12.name(), rel, hm1 + " " + hm2));
		
		FeatureArray fa = this.createFeatureArray(network, fs);
		return fa;
	}
	
	/**
	 * Find the head word of a phrase according to the paper Zhou et al., 2005. 
	 * @param sent
	 * @param span
	 * @return
	 */
	private int getHeadIdx (Sentence sent, Span span) {
		//a mention is from the start to the head end according to Zhou 2005
		for (int i = span.start; i <= span.end; i++) {
			if (i > span.start && sent.get(i).getTag().equals("IN")) {
				return i -1;
			}
		}
		return span.end;
	}

}
