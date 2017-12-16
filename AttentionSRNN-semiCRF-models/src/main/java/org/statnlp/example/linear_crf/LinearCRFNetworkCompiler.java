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
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.NoSuchElementException;

import org.statnlp.commons.types.Instance;
import org.statnlp.commons.types.Label;
import org.statnlp.commons.types.LinearInstance;
import org.statnlp.example.base.BaseNetwork;
import org.statnlp.example.base.BaseNetwork.NetworkBuilder;
import org.statnlp.hypergraph.LocalNetworkParam;
import org.statnlp.hypergraph.Network;
import org.statnlp.hypergraph.NetworkCompiler;
import org.statnlp.hypergraph.NetworkIDMapper;
import org.statnlp.hypergraph.decoding.NodeHypothesis;
import org.statnlp.hypergraph.decoding.ScoredIndex;
import org.statnlp.util.Pipeline;
import org.statnlp.util.instance_parser.DelimiterBasedInstanceParser;

/**
 * @author wei_lu
 *
 */
public class LinearCRFNetworkCompiler extends NetworkCompiler{
	
	private static final long serialVersionUID = -3829680998638818730L;
	
	public Map<Integer, Label> _labels;
	public enum NODE_TYPES {
		LEAF,
		NODE,
		ROOT
		};
	private static int MAX_LENGTH = 300;
	
	private long[] _allNodes;
	private int[][][] _allChildren;

	public static HashMap<Long, HashMap<Long, Integer>> edge2idx;
	private int edgeId;
	
	public LinearCRFNetworkCompiler(Collection<Label> labels){
		this._labels = new HashMap<Integer, Label>();
		for(Label label: labels){
			this._labels.put(label.getId(), new Label(label));
		}
		edge2idx = new HashMap<Long, HashMap<Long, Integer>>();
		edgeId = 0;
		this.compile_unlabled_generic();
	}
	
	public LinearCRFNetworkCompiler(Pipeline<?> pipeline){
		this._labels = new HashMap<Integer, Label>();
		for(Label label: ((DelimiterBasedInstanceParser)pipeline.instanceParser).LABELS.values()){
			this._labels.put(label.getId(), new Label(label));
		}
		edge2idx = new HashMap<Long, HashMap<Long, Integer>>();
		edgeId = 0;
		this.compile_unlabled_generic();
	}
	
	public BaseNetwork compileLabeled(int networkId, Instance instance, LocalNetworkParam param){
		@SuppressWarnings("unchecked")
		LinearInstance<Label> inst = (LinearInstance<Label>)instance;
		NetworkBuilder<BaseNetwork> networkBuilder = NetworkBuilder.builder(BaseNetwork.class);
		
		ArrayList<Label> outputs = (ArrayList<Label>)inst.getOutput();
		int size = outputs.size();
		
		// Add leaf
		long leaf = toNode_leaf();
		networkBuilder.addNode(leaf);
		
		long prevNode = leaf;
		
		for(int i=0; i<size; i++){
			Label label = outputs.get(i);
			long node = toNode(i, label.getId());
			
			networkBuilder.addNode(toNode(i, label.getId()));
			
//			for(Label alllabel: Label.LABELS.values()){
//				network.addNode(toNode(i, alllabel.getId()));
//			}
			
			networkBuilder.addEdge(node, new long[]{prevNode});
			
			prevNode = node;
		}
		
		// Add root
		long root = toNode_root(outputs.size());
		networkBuilder.addNode(root);
		networkBuilder.addEdge(root, new long[]{prevNode});
		
		BaseNetwork network = networkBuilder.build(networkId, inst, param, this);

//		viewer.visualizeNetwork(network, null, "Labeled network for network "+networkId);
		
		return network;
	}

	public BaseNetwork compileUnlabeled(int networkId, Instance inst, LocalNetworkParam param){
		int size = inst.size();
		long root = this.toNode_root(size);
		
		int pos = Arrays.binarySearch(this._allNodes, root);
		int numNodes = pos+1; // Num nodes should equals to (instanceSize * (numLabels+1)) + 1
//		System.out.println(String.format("Instance size: %d, Labels size: %d, numNodes: %d", size, _labels.size(), numNodes));
		
		BaseNetwork result = NetworkBuilder.quickBuild(BaseNetwork.class, networkId, inst, this._allNodes, this._allChildren, numNodes, param, this);
		
//		viewer.visualizeNetwork(result, null, "Unlabeled network for network "+networkId);
		
		return result;
	}
	
	private void compile_unlabled_generic(){
		NetworkBuilder<BaseNetwork> networkBuilder = NetworkBuilder.builder(BaseNetwork.class);
		
		long leaf = this.toNode_leaf();
		networkBuilder.addNode(leaf);
		
		ArrayList<Long> prevNodes = new ArrayList<Long>();
		ArrayList<Long> currNodes = new ArrayList<Long>();
		prevNodes.add(leaf);
		
		for(int k = 0; k <MAX_LENGTH; k++){
			for(int tag_id = 0; tag_id < this._labels.size(); tag_id++){
				long node = this.toNode(k, tag_id);
				currNodes.add(node);
				networkBuilder.addNode(node);
				for(long prevNode : prevNodes){
					networkBuilder.addEdge(node, new long[]{prevNode});
					if(!edge2idx.containsKey(node)){
						edge2idx.put(node,  new HashMap<Long, Integer>());
					}
					if(!edge2idx.get(node).containsKey(prevNodes)) edge2idx.get(node).put(prevNode, edgeId++);
				}
			}
			prevNodes = currNodes;
			currNodes = new ArrayList<Long>();
			
			long root = this.toNode_root(k+1);
			networkBuilder.addNode(root);
			for(long prevNode : prevNodes){
				networkBuilder.addEdge(root, new long[]{prevNode});
				if(!edge2idx.containsKey(root)){
					edge2idx.put(root,  new HashMap<Long, Integer>());
				}
				if(!edge2idx.get(root).containsKey(prevNodes)) edge2idx.get(root).put(prevNode, edgeId++);
			}
			
		}
		
		BaseNetwork network = networkBuilder.buildRudimentaryNetwork();
		
		this._allNodes = network.getAllNodes();
		this._allChildren = network.getAllChildren();
		
	}
	
	public long toNode_leaf(){
		int[] arr = new int[]{0, 0, 0, 0, NODE_TYPES.LEAF.ordinal()};
		return NetworkIDMapper.toHybridNodeID(arr);
	}
	
	public long toNode(int pos, int tag_id){
		int[] arr = new int[]{pos+1, tag_id, 0, 0, NODE_TYPES.NODE.ordinal()};
		return NetworkIDMapper.toHybridNodeID(arr);
	}
	
	public long toNode_root(int size){
		int[] arr = new int[]{size, this._labels.size(), 0, 0, NODE_TYPES.ROOT.ordinal()};
		return NetworkIDMapper.toHybridNodeID(arr);
	}

	
	@Override
	public LinearInstance<Label> decompile(Network network) {
		return decompile(network, 1);
	}
	
	public LinearInstance<Label> decompile(Network network, int numPredictionsGenerated){

		BaseNetwork lcrfNetwork = (BaseNetwork)network;
		@SuppressWarnings("unchecked")
		LinearInstance<Label> instance = (LinearInstance<Label>)lcrfNetwork.getInstance();
		
		ArrayList<ArrayList<Label>> topKPredictions = new ArrayList<ArrayList<Label>>();
		for(int k=0; k<numPredictionsGenerated; k++){
			try{
				topKPredictions.add(getKthBestPrediction(instance, lcrfNetwork, k));
			} catch (NoSuchElementException e){
				break;
			}
		}
		
		LinearInstance<Label> result = instance.duplicate();
		
		result.setPrediction(topKPredictions.get(0));
		result.setTopKPredictions(topKPredictions);
		
		return result;
	}
	
	private ArrayList<Label> getKthBestPrediction(LinearInstance<Label> instance, BaseNetwork lcrfNetwork, int k){
		int size = instance.size();
		ArrayList<Label> predictions = new ArrayList<Label>();
		long root = toNode_root(size);
		int node_k = Arrays.binarySearch(_allNodes, root);
		NodeHypothesis nodeHypothesis = lcrfNetwork.getNodeHypothesis(node_k);
		ScoredIndex bestPath = nodeHypothesis.getKthBestHypothesis(k);

		ScoredIndex[] children_k;
		for(int i=size-1; i>=0; i--){
			try{
				children_k = lcrfNetwork.getMaxPath(nodeHypothesis, bestPath);
			} catch (NoSuchElementException e){
				throw new NoSuchElementException("There is no "+k+"-best result!");
			}
			if(children_k.length != 1){
				System.err.println("Child length not 1!");
			}
			int child_k = children_k[0].node_k;
			long child = lcrfNetwork.getNode(child_k);
			nodeHypothesis = lcrfNetwork.getNodeHypothesis(child_k);
			int[] child_arr = NetworkIDMapper.toHybridNodeArray(child);
			int pos = child_arr[0]-1;
			int tag_id = child_arr[1];
			if(pos != i){
				System.err.println("Position encoded in the node array not the same as the interpretation!");
			}
			predictions.add(0, _labels.get(tag_id));
//			node_k = child_k;
			bestPath = children_k[0];
		}
		return predictions;
	}
	
	public double costAt(Network network, int parent_k, int[] child_k){
		return super.costAt(network, parent_k, child_k);
	}

	/**
	 * Returns the position in the input represented by the given node.
	 * @return
	 */
	public int getPosForNode(int[] nodeArray){
		return nodeArray[0]-1;
	}
	
	public Integer getOutputForNode(int[] nodeArray){
		Label label =_labels.get(nodeArray[1]);
		if(label == null){
			return null;
		}
		if(nodeArray[4] == NODE_TYPES.LEAF.ordinal()){
			return null;
		}
		return nodeArray[1];
	}

}
