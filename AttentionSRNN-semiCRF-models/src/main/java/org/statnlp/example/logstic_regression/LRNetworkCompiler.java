package org.statnlp.example.logstic_regression;

import java.util.Arrays;

import org.statnlp.commons.types.Instance;
import org.statnlp.example.base.BaseNetwork;
import org.statnlp.example.base.BaseNetwork.NetworkBuilder;
import org.statnlp.hypergraph.LocalNetworkParam;
import org.statnlp.hypergraph.Network;
import org.statnlp.hypergraph.NetworkCompiler;
import org.statnlp.hypergraph.NetworkIDMapper;

public class LRNetworkCompiler extends NetworkCompiler {

	private static final long serialVersionUID = -3523483926592577270L;

	public enum NodeTypes {LEAF,NODE,ROOT};
	public int _size;
	private boolean DEBUG = false;
	
	private long[] _allNodes;
	private int[][][] _allChildren;
	
	public LRNetworkCompiler() {
		NetworkIDMapper.setCapacity(new int[]{3, RelationType.RELS.size(), 3});
		this.buildGenericNetwork();
	}

	private long toNode_leaf() {
		return NetworkIDMapper.toHybridNodeID(new int[]{0, 0, NodeTypes.LEAF.ordinal()});
	}
	
	private long toNode(int label) {
		return NetworkIDMapper.toHybridNodeID(new int[]{1, label, NodeTypes.NODE.ordinal()});
	}
	
	private long toNode_root() {
		return NetworkIDMapper.toHybridNodeID(new int[]{2, 0, NodeTypes.ROOT.ordinal()});
	}

	@Override
	public BaseNetwork compileLabeled(int networkId, Instance inst, LocalNetworkParam param) {
		NetworkBuilder<BaseNetwork> networkBuilder = NetworkBuilder.builder(BaseNetwork.class);
		LRInstance lgInst = (LRInstance)inst;
		long leaf = toNode_leaf();
		long[] children = new long[]{leaf};
		networkBuilder.addNode(leaf);
		long node = toNode(lgInst.getOutput().id);
		networkBuilder.addNode(node);
		networkBuilder.addEdge(node, children);
		children = new long[]{node};
		long root = toNode_root();
		networkBuilder.addNode(root);
		networkBuilder.addEdge(root, children);
		BaseNetwork network = networkBuilder.build(networkId, inst, param, this);
		if (DEBUG) {
			BaseNetwork unlabeled = this.compileUnlabeled(networkId, inst, param);
			if(!unlabeled.contains(network))
				System.err.println("not contains");
		}
		return network;
	}

	@Override
	public BaseNetwork compileUnlabeled(int networkId, Instance inst, LocalNetworkParam param) {
		long root = this.toNode_root();
		int pos = Arrays.binarySearch(this._allNodes, root);
		int numNodes = pos+1; 
		BaseNetwork result = NetworkBuilder.quickBuild(BaseNetwork.class, networkId, inst, this._allNodes, this._allChildren, numNodes, param, this);
		return result;
	}
	
	private void buildGenericNetwork() {
		NetworkBuilder<BaseNetwork> networkBuilder = NetworkBuilder.builder(BaseNetwork.class);
		long leaf = toNode_leaf();
		long[] leaves = new long[]{leaf};
		networkBuilder.addNode(leaf);
		long root = toNode_root();
		networkBuilder.addNode(root);
		for (int l = 0; l < RelationType.RELS.size(); l++) {
			long node = this.toNode(l);
			networkBuilder.addNode(node);
			networkBuilder.addEdge(node, leaves);
			networkBuilder.addEdge(root, new long[]{node});
		}
		BaseNetwork network = networkBuilder.buildRudimentaryNetwork();
		this._allNodes = network.getAllNodes();
		this._allChildren = network.getAllChildren();
	}
	
	@Override
	public Instance decompile(Network network) {
		BaseNetwork baseNetwork = (BaseNetwork)network;
		LRInstance inst = (LRInstance)network.getInstance();
		long node = this.toNode_root();
		int nodeIdx = Arrays.binarySearch(baseNetwork.getAllNodes(), node);
		int labeledNodeIdx = baseNetwork.getMaxPath(nodeIdx)[0];
		int[] arr = baseNetwork.getNodeArray(labeledNodeIdx);
		int labelId = arr[1];
		inst.setPrediction(RelationType.get(labelId));
		return inst;
	}


}
