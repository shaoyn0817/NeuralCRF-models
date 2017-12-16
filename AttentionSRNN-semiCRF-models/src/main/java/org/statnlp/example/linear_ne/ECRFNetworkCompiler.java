package org.statnlp.example.linear_ne;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import org.statnlp.commons.types.Instance;
import org.statnlp.example.base.BaseNetwork;
import org.statnlp.example.base.BaseNetwork.NetworkBuilder;
import org.statnlp.hypergraph.LocalNetworkParam;
import org.statnlp.hypergraph.Network;
import org.statnlp.hypergraph.NetworkCompiler;
import org.statnlp.hypergraph.NetworkIDMapper;

public class ECRFNetworkCompiler extends NetworkCompiler{

	private static final long serialVersionUID = -2388666010977956073L;

	public enum NodeType {Leaf, Node, Root};
	public int _size;
	public BaseNetwork genericUnlabeledNetwork;
	private boolean useIOBESConstraintBuildNetwork = true;
	//0: should be start tag
	//length -1: should be end tag;
	private Entity[] labels;
	private Map<Entity, Integer> labelIndex;
	
	static {
		NetworkIDMapper.setCapacity(new int[]{150, 50, 3});
	}
	
	public ECRFNetworkCompiler(boolean useIOBESConstraint, Entity[] labels){
		this._size = 150;
		this.useIOBESConstraintBuildNetwork = useIOBESConstraint;
		this.labels = labels;
		this.labelIndex = new HashMap<>(this.labels.length);
		for (int l = 0; l < this.labels.length; l++) {
			this.labelIndex.put(this.labels[l], l);
		}
		this.compileUnlabeledInstancesGeneric();
	}
	
	public long toNode_leaf(){
		//since 0 is the start_tag index;
		int[] arr = new int[]{0, 0, NodeType.Leaf.ordinal()};
		return NetworkIDMapper.toHybridNodeID(arr);
	}
	
	public long toNode(int pos, int tag_id){
		int[] arr = new int[]{pos, tag_id, NodeType.Node.ordinal()};
		return NetworkIDMapper.toHybridNodeID(arr);
	}
	
	public long toNode_root(int size){
		int[] arr = new int[]{size - 1, labels.length - 1, NodeType.Root.ordinal()};
		return NetworkIDMapper.toHybridNodeID(arr);
	}

	public BaseNetwork compileLabeled(int networkId, Instance instance, LocalNetworkParam param){
		ECRFInstance inst = (ECRFInstance)instance;
		NetworkBuilder<BaseNetwork> lcrfNetwork = NetworkBuilder.builder();
		long leaf = toNode_leaf();
		long[] children = new long[]{leaf};
		lcrfNetwork.addNode(leaf);
		for(int i = 0; i < inst.size(); i++){
			long node = toNode(i, this.labelIndex.get( Entity.get(inst.getOutput().get(i))) );
			lcrfNetwork.addNode(node);
			long[] currentNodes = new long[]{node};
			lcrfNetwork.addEdge(node, children);
			children = currentNodes;
		}
		long root = toNode_root(inst.size());
		lcrfNetwork.addNode(root);
		lcrfNetwork.addEdge(root, children);
		BaseNetwork network = lcrfNetwork.build(networkId, inst, param, this);
		if(!genericUnlabeledNetwork.contains(network)){
			System.err.println("not contains");
		}
		return network;
	}
	
	public BaseNetwork compileUnlabeled(int networkId, Instance inst, LocalNetworkParam param){
		long[] allNodes = genericUnlabeledNetwork.getAllNodes();
		long root = toNode_root(inst.size());
		int rootIdx = Arrays.binarySearch(allNodes, root);
		BaseNetwork lcrfNetwork = NetworkBuilder.quickBuild(networkId, inst, allNodes, genericUnlabeledNetwork.getAllChildren(), rootIdx+1, param, this);
		return lcrfNetwork;
	}
	
	public void compileUnlabeledInstancesGeneric(){
		NetworkBuilder<BaseNetwork> lcrfNetwork = NetworkBuilder.builder();
		long leaf = toNode_leaf();
		long[] children = new long[]{leaf};
		lcrfNetwork.addNode(leaf);
		for(int i = 0; i < _size; i++){
			long[] currentNodes = new long[this.labels.length];
			for(int l = 0; l < this.labels.length; l++){
				long node = toNode(i,l);
				String currEntity =  this.labels[l].getForm();
				for(long child: children){
					if(child==-1) continue;
					if (useIOBESConstraintBuildNetwork) {
						int[] childArr = NetworkIDMapper.toHybridNodeArray(child);
						String childEntity = i == 0 ? "O" : this.labels[childArr[1]].getForm();
						if (l == 0 || l== this.labels.length -1) continue;
						if( (childEntity.startsWith("B-") || childEntity.startsWith("I-")  ) 
								&& (currEntity.startsWith("I-") || currEntity.startsWith("E-"))
								&& childEntity.substring(2).equals(currEntity.substring(2)) ) {
							if(lcrfNetwork.contains(child)){
								lcrfNetwork.addNode(node);
								lcrfNetwork.addEdge(node, new long[]{child});
							}
						}else if(   (childEntity.startsWith("S-") || childEntity.startsWith("E-") || childEntity.equals("O") ) 
								&& (currEntity.startsWith("B-") ||currEntity.startsWith("S-") || currEntity.startsWith("O") ) ) {
							if(lcrfNetwork.contains(child)){
								lcrfNetwork.addNode(node);
								lcrfNetwork.addEdge(node, new long[]{child});
							}
						}
					} else {
						if(lcrfNetwork.contains(child)){
							lcrfNetwork.addNode(node);
							lcrfNetwork.addEdge(node, new long[]{child});
						}
					}
				}
				if(lcrfNetwork.contains(node))
					currentNodes[l] = node;
				else currentNodes[l] = -1;
			}
			long root = toNode_root(i + 1);
			lcrfNetwork.addNode(root);
			for(long child : currentNodes){
				if(child==-1) continue;
				int[] childArr = NetworkIDMapper.toHybridNodeArray(child);
				String childEntity =  this.labels[childArr[1]].getForm();
				if (useIOBESConstraintBuildNetwork) {
					if(!childEntity.startsWith("B-")&&  !childEntity.startsWith("I-")
						&& childArr[1] != 0 && childArr[1] != this.labels.length - 1) {
						lcrfNetwork.addEdge(root, new long[]{child});
					}
				} else {
					lcrfNetwork.addEdge(root, new long[]{child});
				}
			}
			children = currentNodes;
		}
		BaseNetwork network = lcrfNetwork.buildRudimentaryNetwork();
		genericUnlabeledNetwork =  network;
		System.out.println("nodes:" + genericUnlabeledNetwork.getAllNodes().length);
	}
	
	@Override
	public ECRFInstance decompile(Network network) {
		BaseNetwork lcrfNetwork = (BaseNetwork)network;
		ECRFInstance lcrfInstance = (ECRFInstance)lcrfNetwork.getInstance();
		ECRFInstance result = lcrfInstance.duplicate();
		ArrayList<String> prediction = new ArrayList<String>();
		long root = toNode_root(lcrfInstance.size());
		int rootIdx = Arrays.binarySearch(lcrfNetwork.getAllNodes(),root);
		//System.err.println(rootIdx+" final score:"+network.getMax(rootIdx));
		for(int i=0;i<lcrfInstance.size();i++){
			int child_k = lcrfNetwork.getMaxPath(rootIdx)[0];
			long child = lcrfNetwork.getNode(child_k);
			rootIdx = child_k;
			int tagID = NetworkIDMapper.toHybridNodeArray(child)[1];
			String resEntity = this.labels[tagID].getForm();
			if(resEntity.startsWith("S-")) resEntity = "B-"+resEntity.substring(2);
			if(resEntity.startsWith("E-")) resEntity = "I-"+resEntity.substring(2);
			if (tagID == 0 || tagID == this.labels.length -1)
				resEntity = "O";
			prediction.add(0, resEntity);
		}
		result.setPrediction(prediction);
		return result;
	}
	
	public double costAt(Network network, int parent_k, int[] child_k){
		return super.costAt(network, parent_k, child_k);
	}
	
}
