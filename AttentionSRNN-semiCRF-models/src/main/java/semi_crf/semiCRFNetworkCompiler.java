
package semi_crf; 


import java.util.ArrayList; 
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.statnlp.commons.types.Instance;
import org.statnlp.example.base.BaseNetwork;
import org.statnlp.example.base.BaseNetwork.NetworkBuilder;
import org.statnlp.hypergraph.LocalNetworkParam;
import org.statnlp.hypergraph.Network;
import org.statnlp.hypergraph.NetworkCompiler;
import org.statnlp.hypergraph.NetworkException;
import org.statnlp.hypergraph.NetworkIDMapper;





public class semiCRFNetworkCompiler extends NetworkCompiler{ 
	 
	private static final long serialVersionUID = -3829680998638818730L; 
	public BaseNetwork unlabeledNetwork;
	public List<Label> _labels; 
	public static enum NODE_TYPES {LEAF, NODE, ROOT}; 
	private static int MAX_LENGTH =60; 
	//private static int maxSegmentLength = 20; 
	 
	private long[] _allNodes; 
	private int[][][] _allChildren; 
	 
	public semiCRFNetworkCompiler(){ 
		this._labels = new ArrayList<Label>();//初始化label集合 
		for(Label label: Label.LABELS.values()){ 
			this._labels.add(new Label(label)); 
		} 
		this.compile_unlabled_generic(); 
	} 
	 
	@Override 
	public BaseNetwork compile(int networkId, Instance instance, LocalNetworkParam param) { 
		semiCRFInstance inst = (semiCRFInstance) instance; 
		if(inst.isLabeled()){ 
			return this.compileLabeled(networkId, inst, param); 
		} else { 
			return this.compileUnlabeled(networkId, inst, param); 
		}	 
	} 

	@Override  
	public BaseNetwork compileUnlabeled(int networkId, Instance inst, LocalNetworkParam param){ 
        int size = inst.size();
        //System.out.println(size);
		long root = toNode_root(size);
		long[] allNodes = unlabeledNetwork.getAllNodes();
		int[][][] allChildren = unlabeledNetwork.getAllChildren();
		int root_k  = unlabeledNetwork.getNodeIndex(root);
		int numNodes = root_k+1;
		if(numNodes<=0) {
			System.out.println(inst.getInput());
		}
		BaseNetwork network = NetworkBuilder.quickBuild(networkId, inst, allNodes, allChildren, numNodes, param, this);
		return network;
	} 
	 
	private void compile_unlabled_generic(){ 
		NetworkBuilder<BaseNetwork> network = NetworkBuilder.builder();
		ArrayList<Long> currNodes = new ArrayList<Long>(); 
		 
        long leaf = this.toNode_leaf(); 
		network.addNode(leaf); 

		 
		for(int pos = 0; pos <MAX_LENGTH; pos++){ 
			for(int index = 0; index < this._labels.size(); index++){
				int tag_id = this._labels.get(index)._id;
				long node = this.toNode(pos, tag_id); 
				currNodes.add(node); 
				network.addNode(node); 
				 
				/*
				if(Label.get(tag_id).getForm().equals("date")) 
					this.connect(4, tag_id, pos, network); 
				else if(Label.get(tag_id).getForm().equals("tech")) 
					this.connect(10, tag_id, pos, network); 
				else if(Label.get(tag_id).getForm().equals("note")) 
					this.connect(11, tag_id, pos, network); 
				else if(Label.get(tag_id).getForm().equals("editor")) 
					this.connect(13, tag_id, pos, network); 
				else if(Label.get(tag_id).getForm().equals("author")) //40
					this.connect(40, tag_id, pos, network); 
				else if(Label.get(tag_id).getForm().equals("title")) //27
					this.connect(27, tag_id, pos, network); 
				else if(Label.get(tag_id).getForm().equals("volume")) 
					this.connect(4, tag_id, pos, network); 
				else if(Label.get(tag_id).getForm().equals("institution")) 
					this.connect(15, tag_id, pos, network); 
				else if(Label.get(tag_id).getForm().equals("journal")) 
					this.connect(14, tag_id, pos, network); 
				else if(Label.get(tag_id).getForm().equals("pages")) 
					this.connect(4, tag_id, pos, network); 
				else if(Label.get(tag_id).getForm().equals("publisher")) 
					this.connect(5, tag_id, pos, network); 
				else if(Label.get(tag_id).getForm().equals("location")) 
					this.connect(6, tag_id, pos, network); 
				else if(Label.get(tag_id).getForm().equals("booktitle")) 
					this.connect(16, tag_id, pos, network); 
				else System.err.println("出现不一样的title名字！"); 
				*/
				if(Label.get(tag_id).getForm().equals("ORG")) 
					this.connect(10, tag_id, pos, network); 
				else if(Label.get(tag_id).getForm().equals("LOC")) 
					this.connect(10, tag_id, pos, network); 
				else if(Label.get(tag_id).getForm().equals("MISC")) 
					this.connect(7, tag_id, pos, network); 
				else if(Label.get(tag_id).getForm().equals("PER")) 
					this.connect(6, tag_id, pos, network); 
				else if(Label.get(tag_id).getForm().equals("O")) 
					this.Oconnect(1, tag_id, pos, network); 
				else System.err.println("出现不一样的title名字！"+ Label.get(tag_id).getForm()+"**"); 
			} 
			 
			 
			long root = this.toNode_root(pos+1); 
			network.addNode(root); 
			for(long currNode : currNodes){ 
				network.addEdge(root, new long[]{currNode}); 
			} 
		 
			currNodes = new ArrayList<Long>();		 
		} 
        BaseNetwork net = network.buildRudimentaryNetwork();
		
		this.unlabeledNetwork = net;
		System.out.println("目前处于稀疏状态");  
		 
	} 
	 
	public long toNode_leaf(){ 
		int[] arr = new int[]{0, 0, 0, 0, NODE_TYPES.LEAF.ordinal()}; 
		return NetworkIDMapper.toHybridNodeID(arr); 
	} 
	 
	public long toNode(int position, int label){ 
		int[] arr = new int[]{position, label, 0, 0, NODE_TYPES.NODE.ordinal()}; 
		return NetworkIDMapper.toHybridNodeID(arr); 
	} 
	 
	public long toNode_root(int size){ 
		int[] arr = new int[]{size, this._labels.size(), 0, 0, NODE_TYPES.ROOT.ordinal()}; 
		return NetworkIDMapper.toHybridNodeID(arr); 
	} 

	@Override
	public BaseNetwork compileLabeled(int networkId, Instance instance, LocalNetworkParam param){ 
		NetworkBuilder<BaseNetwork> network = NetworkBuilder.builder();
		semiCRFInstance inst = (semiCRFInstance) instance;
		ArrayList<Span> outputs = inst.getOutput(); 
		ArrayList<String> inputs = inst.getInput(); 
		 
		// Add leaf 
		long leaf = toNode_leaf(); 
		network.addNode(leaf); 
		 
		long prevNode = leaf; 
		for(int i = 0; i < outputs.size(); i++){ 
			Label label = outputs.get(i)._label; 
			int endPosition = outputs.get(i)._end; 
			long node = toNode(endPosition, label.getId()); 
			network.addNode(node); 
			network.addEdge(node, new long[]{prevNode}); 
			prevNode = node; 
		} 
		 
		// Add root 
		long root = toNode_root(inputs.size()); 
		network.addNode(root); 
		network.addEdge(root, new long[]{prevNode}); 
		 
		BaseNetwork net = network.build(networkId, inst, param, this);
		 
		return net; 
	} 
	 
	@Override 
	public semiCRFInstance decompile(Network network) { 
		BaseNetwork semiNetwork = (BaseNetwork)network;
		semiCRFInstance instance = (semiCRFInstance)semiNetwork.getInstance(); 
		int size = instance.input.size(); 
		semiCRFInstance result = instance.duplicate(); 
		ArrayList<Span> predictions = new ArrayList<Span>(); 
		//long root = toNode_root(size); 
		long root = semiNetwork.getRoot();
		long allnodes[] = semiNetwork.getAllNodes();
		int node_k = Arrays.binarySearch(allnodes, root); 
		

		
		
	    while(true){ 
	    	int[] children_k = semiNetwork.getMaxPath(node_k); 
			if(children_k.length != 1){ 
				System.out.println("Child length not 1!   " ); 
				break; 
			} 
			int child_k = children_k[0]; 
			int []child_arr = network.getNodeArray(child_k); 
			int tag_id = child_arr[1]; 
			int endPosition = child_arr[0]; 
			 
			//遇到叶子节点break 
			if(child_arr[4] == NODE_TYPES.LEAF.ordinal()) 
				break; 
			predictions.add(0 ,new Span(endPosition, tag_id)); 
			node_k = child_k; 
		} 
		 
		result.setPrediction(predictions); 
		 
		return result; 
	} 
	 
	/*
	public double costAt(Network network, int parent_k, int[] child_k){ 
		return super.costAt(network, parent_k, child_k); 
	} 
	*/ 
	 
	public void connect(int windowSize, int tag_id, int pos, NetworkBuilder<BaseNetwork> network){ 
		long node = this.toNode(pos, tag_id); 
		for(int i = pos - 1; i >= pos-windowSize; i--){ 
			if(i == -1){ 
				long prenode = this.toNode_leaf(); 
				network.addEdge(node, new long[]{prenode}); 
				break;
			} else { 
				for(int j = 0; j < this._labels.size(); j++){ 
					int tag = this._labels.get(j)._id;
					if(tag == tag_id){						
						//long prenode = this.toNode(i, tag); 
						//network.addEdge(node, new long[]{prenode});
						continue; 
					}
					else{ 
						long prenode = this.toNode(i, tag); 
						network.addEdge(node, new long[]{prenode});
					} 
				} 
			} 
		}
	}

	public void Oconnect(int windowSize, int tag_id, int pos, NetworkBuilder<BaseNetwork> network){ 
		long node = this.toNode(pos, tag_id); 
		for(int i = pos - 1; i >= pos-windowSize; i--){ 
			if(i == -1){ 
				long prenode = this.toNode_leaf(); 
				network.addEdge(node, new long[]{prenode}); 
				break;
			} else { 
				for(int j = 0; j < this._labels.size(); j++){ 
					int tag = this._labels.get(j)._id;
					if(tag == tag_id){						
						long prenode = this.toNode(i, tag); 
						network.addEdge(node, new long[]{prenode});
						//continue; 
					}
					else{ 
						long prenode = this.toNode(i, tag); 
						network.addEdge(node, new long[]{prenode});
					} 
				} 
			} 
		}
	}





	 
	/*
	public void specialConnect(int windowSize, int tag_id, int pos, semiCRFNetwork network){ 
		long node = this.toNode(pos, tag_id); 
		for(int i = pos - 1; i >= pos - windowSize-20; i--){ 
			if(i == -1){ 
				long prenode = this.toNode_leaf(); 
				network.addEdge(node, new long[]{prenode}); 
				break; 
			} else { 
				for(int j = 0; j < this._labels.size(); j++){ 
					long prenode = this.toNode(i, j); 
					network.addEdge(node, new long[]{prenode}); 
				} 
			} 
		} 
	} 
	
	public void crfConnect(int tag_id,int pos, semiCRFNetwork network){
		long node = this.toNode(pos, tag_id);
		if(pos == 0){
			long leaf = toNode_leaf();
			network.addEdge(node, new long[]{leaf});
		} else {
			for(int j = 0; j < this._labels.size(); j++){ 
				long prenode = this.toNode(pos-1, j); 
				network.addEdge(node, new long[]{prenode}); 
			} 
		}
	}
*/
} 











