package org.statnlp.example.fcrf;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;

import org.statnlp.commons.types.Sentence;
import org.statnlp.example.fcrf.FCRFConfig.TASK;
import org.statnlp.example.fcrf.FCRFNetworkCompiler.NODE_TYPES;
import org.statnlp.hypergraph.FeatureArray;
import org.statnlp.hypergraph.FeatureBox;
import org.statnlp.hypergraph.FeatureManager;
import org.statnlp.hypergraph.GlobalNetworkParam;
import org.statnlp.hypergraph.Network;
import org.statnlp.hypergraph.NetworkConfig;
import org.statnlp.hypergraph.NetworkIDMapper;
import org.statnlp.hypergraph.neural.MultiLayerPerceptron;

public class FCRFFeatureManager extends FeatureManager {

	private static final long serialVersionUID = 376931974939202432L;

	private boolean useJointFeatures;
	//private String OUT_SEP = NeuralConfig.OUT_SEP; 
	private String IN_SEP = MultiLayerPerceptron.IN_SEP;
	// TODO: Update the extract_helper to use net
	
	private static HashSet<String> others = new HashSet<>(Arrays.asList("#STR#", "#END#", "#STR1#", "#END1#", "#str#", "#end#", "#str1#", "#end1#"));
	
	private int windowSize;
	private boolean cascade;
	private TASK task;
	private boolean iobes;
	private boolean removeChunkNeural = false;
	private boolean removePOSNeural = false;
	private boolean removeJointNeural = false;
	public enum FEATYPE {
		chunk_currWord,
		chunk_leftWord1,
		chunk_leftWord2,
		chunk_rightWord1,
		chunk_rightWord2,
		chunk_cap, 
		chunk_cap_l, 
		chunk_cap_ll, 
		chunk_cap_r, 
		chunk_cap_rr,
		chunk_transition,
		tag_currWord,
		tag_leftWord1,
		tag_leftWord2,
		tag_rightWord1,
		tag_rightWord2,
		tag_cap, 
		tag_cap_l, 
		tag_cap_ll, 
		tag_cap_r, 
		tag_cap_rr,
		tag_transition,
		joint1,
		joint2,
		joint3,
		joint4,
		joint5,
		neural
		};
	
		
	public FCRFFeatureManager(GlobalNetworkParam param_g, boolean useJointFeatures, boolean cascade, TASK task, int windowSize, boolean iobes,
			boolean removeChunkNeural, boolean removePOSNeural, boolean removeJointNeural) {
		super(param_g);
		this.useJointFeatures = useJointFeatures; 
		this.cascade = cascade;
		this.task = task;
		this.windowSize = windowSize;
		this.iobes = iobes;
		this.removeChunkNeural = removeChunkNeural;
		this.removePOSNeural = removePOSNeural;
		this.removeJointNeural = removeJointNeural;
	}

	@Override
	protected FeatureArray extract_helper(Network network, int parent_k, int[] children_k, int children_k_index) {
		FCRFInstance inst = ((FCRFInstance)network.getInstance());
		Sentence sent = inst.getInput();
		long node = network.getNode(parent_k);
		int[] nodeArr = NetworkIDMapper.toHybridNodeArray(node);
		FeatureArray fa = null;
		ArrayList<Integer> jointFeatureList = new ArrayList<>();
		ArrayList<Integer> dstNodes = new ArrayList<>();
		int threadId = network.getThreadId();
		int pos = nodeArr[0] - 1;
		int eId = nodeArr[1];
		if(pos<0 || pos > inst.size())
			return FeatureArray.EMPTY;
		ArrayList<Integer> c_wordList = new ArrayList<Integer>();
		ArrayList<Integer> c_capList = new ArrayList<Integer>();
		ArrayList<Integer> c_transitionList = new ArrayList<Integer>();
		ArrayList<Integer> c_cascadeList = new ArrayList<Integer>();
		ArrayList<Integer> c_neuralList = new ArrayList<Integer>();
		ArrayList<Integer> t_wordList = new ArrayList<Integer>();
		ArrayList<Integer> t_capList = new ArrayList<Integer>();
		ArrayList<Integer> t_transitionList = new ArrayList<Integer>();
		ArrayList<Integer> t_cascadeList = new ArrayList<Integer>();
		ArrayList<Integer> t_neuralList = new ArrayList<Integer>();
		
		
		
		ArrayList<ArrayList<Integer>> bigList = new ArrayList<ArrayList<Integer>>();
		if (nodeArr[2] == NODE_TYPES.ENODE.ordinal()){
			if(pos==inst.size() || eId==(Chunk.CHUNKS.size()+Tag.TAGS.size())) return FeatureArray.EMPTY;
			int childLabelId = network.getNodeArray(children_k[0])[1];
			addChunkFeatures(network, sent, pos, eId, childLabelId, c_wordList, c_capList, c_transitionList, c_cascadeList, c_neuralList);
			if(useJointFeatures)
				addJointFeatures(jointFeatureList, network, sent, pos, eId, parent_k, children_k, false, dstNodes);
			bigList.add(c_wordList);
			bigList.add(c_capList);
			bigList.add(c_transitionList);
			if (cascade) bigList.add(c_cascadeList);
			if (NetworkConfig.USE_NEURAL_FEATURES) bigList.add(c_neuralList); 
			if (useJointFeatures) bigList.add(jointFeatureList);
																
		} else if (nodeArr[2] == NODE_TYPES.TNODE.ordinal() ){//|| nodeArr[1] == NODE_TYPES.ROOT.ordinal()){
			if(pos==inst.size() || eId==(Chunk.CHUNKS.size()+Tag.TAGS.size())) return FeatureArray.EMPTY;
			int childLabelId = network.getNodeArray(children_k[0])[1];
			addPOSFeatures(network, sent, pos, eId, childLabelId, t_wordList, t_capList, t_transitionList, t_cascadeList, t_neuralList);
			if(useJointFeatures)
				addJointFeatures(jointFeatureList, network, sent, pos, eId, parent_k, children_k, true, dstNodes);
			bigList.add(t_wordList); bigList.add(t_capList);
			bigList.add(t_transitionList); 
			if (cascade) bigList.add(t_cascadeList);
			if (NetworkConfig.USE_NEURAL_FEATURES) bigList.add(t_neuralList);
			if (useJointFeatures) bigList.add(jointFeatureList);
		}
		int[] dstNodesArr = new int[dstNodes.size()];
		for (int idx = 0; idx < dstNodes.size(); idx++) dstNodesArr[idx] = dstNodes.get(idx);
		FeatureArray orgFa = new FeatureArray(FeatureBox.getFeatureBox(new int[]{}, this.getParams_L()[threadId]));
		fa = orgFa;
		for (int i = 0; i < bigList.size(); i++) {
			boolean setAlwaysChange = false;
			if (useJointFeatures && i == bigList.size() - 1) setAlwaysChange = true;
			FeatureArray curr = addNext(fa, bigList.get(i), threadId, setAlwaysChange, dstNodesArr);
			fa = curr;
		}
		
		return orgFa;
	}
	
	private FeatureArray addNext(FeatureArray fa, ArrayList<Integer> featureList, int threadId, boolean setAlwaysChange, int[] dstNodes)  {
		ArrayList<Integer> finalList = new ArrayList<Integer>();
		for(int i=0;i<featureList.size();i++){
			if(featureList.get(i)!=-1)
				finalList.add( featureList.get(i) );
		}
		if(finalList.size()==0) return fa;
		else {
			int[] fs = new int[finalList.size()];
			for(int i = 0; i < fs.length; i++) fs[i] = finalList.get(i);
			FeatureArray curr = new FeatureArray(FeatureBox.getFeatureBox(fs, this.getParams_L()[threadId]));
			if (setAlwaysChange) { //means this is the joint feature
				curr.setJoint();
				curr.setDstNodes(dstNodes);
			}
			fa.next(curr);
			return curr;
		}
	}
	
	private void addChunkFeatures(Network network, Sentence sent, int pos, int eId, int childLabelId,
			ArrayList<Integer> wordList, ArrayList<Integer> capList, ArrayList<Integer> transitionList, ArrayList<Integer> cascadeList,
			ArrayList<Integer> neuralList){
		String lw = pos > 0? sent.get(pos-1).getForm(): "#STR#";
		String lcaps = capsF(lw);
		String llw = pos == 0? "#STR1#": pos==1? "#STR#" : sent.get(pos-2).getForm();
		String llcaps = capsF(llw);
		String rw = pos<sent.length()-1? sent.get(pos+1).getForm():"#END#";
		String rcaps = capsF(rw);
		String rrw = pos == sent.length()-1? "#END1#": pos==sent.length()-2? "#END#":sent.get(pos+2).getForm();
		String rrcaps = capsF(rrw);
		String currWord = sent.get(pos).getForm();
		String currEn = Chunk.get(eId).getForm();
		String prevEn = childLabelId == Chunk.CHUNKS.size()+Tag.TAGS.size()? "O" : Chunk.get(eId).getForm();
		String currCaps = capsF(currWord);
		
		wordList.add(this._param_g.toFeature(network,FEATYPE.chunk_currWord.name(), 	currEn,	currWord.toLowerCase()));
		wordList.add(this._param_g.toFeature(network,FEATYPE.chunk_leftWord1.name(), 	currEn,	lw.toLowerCase()));
		wordList.add(this._param_g.toFeature(network,FEATYPE.chunk_leftWord2.name(), 	currEn,	llw.toLowerCase()));
		wordList.add(this._param_g.toFeature(network,FEATYPE.chunk_rightWord1.name(), 	currEn,	rw.toLowerCase()));
		wordList.add(this._param_g.toFeature(network,FEATYPE.chunk_rightWord2.name(), 	currEn,	rrw.toLowerCase()));
		
		capList.add(this._param_g.toFeature(network, FEATYPE.chunk_cap.name(), 		currEn,  currCaps));
		capList.add(this._param_g.toFeature(network, FEATYPE.chunk_cap_l.name(), 	currEn,  lcaps));
		capList.add(this._param_g.toFeature(network, FEATYPE.chunk_cap_ll.name(), 	currEn,  llcaps));
		capList.add(this._param_g.toFeature(network, FEATYPE.chunk_cap_r.name(), 	currEn,  rcaps));
		capList.add(this._param_g.toFeature(network, FEATYPE.chunk_cap_rr.name(),	currEn,  rrcaps));
		
		transitionList.add(this._param_g.toFeature(network, FEATYPE.chunk_transition.name(),	currEn,  prevEn));
		
		if(task == TASK.CHUNKING && cascade){
			String currTag = sent.get(pos).getTag();
			cascadeList.add(this._param_g.toFeature(network, FEATYPE.tag_currWord.name(), 	currEn,  currTag));
		}
		
		if(NetworkConfig.USE_NEURAL_FEATURES && !removeChunkNeural){
			if(windowSize == 5)
				neuralList.add(this._param_g.toFeature(network, FEATYPE.neural.name(), currEn, llw.toLowerCase()+IN_SEP+
																						lw.toLowerCase()+IN_SEP+
																						currWord.toLowerCase()+IN_SEP+
																						rw.toLowerCase()+IN_SEP+
																						rrw.toLowerCase()));
			else if(windowSize == 3)
				neuralList.add(this._param_g.toFeature(network, FEATYPE.neural.name(), currEn, lw.toLowerCase()+IN_SEP+
						currWord.toLowerCase()+IN_SEP+
						rw.toLowerCase()));
			else if(windowSize == 1)
				neuralList.add(this._param_g.toFeature(network, FEATYPE.neural.name(), currEn, currWord.toLowerCase()));
			else throw new RuntimeException("Unknown window size: "+windowSize);
		}
	}

	private void addPOSFeatures(Network network, Sentence sent, int pos, int tId, int childLabelId,
			ArrayList<Integer> wordList, ArrayList<Integer> capList, ArrayList<Integer> transitionList, ArrayList<Integer> cascadeList,
			ArrayList<Integer> neuralList){
		String currTag = Tag.get(tId).getForm();
		String prevTag = childLabelId == Chunk.CHUNKS.size()+Tag.TAGS.size()? "#STR#" : Tag.get(childLabelId).getForm();
		String lw = pos > 0? sent.get(pos-1).getForm():"#STR#";
		String llw = pos==0? "#STR1#": pos==1? "#STR#":sent.get(pos-2).getForm();
		String rw = pos<sent.length()-1? sent.get(pos+1).getForm():"#END#";
		String rrw = pos==sent.length()-1? "#END1#": pos==sent.length()-2? "#END#":sent.get(pos+2).getForm();
		String w = sent.get(pos).getForm();
		
		String caps = capsF(w);
		String lcaps = capsF(lw);
		String llcaps = capsF(llw);
		String rcaps = capsF(rw);
		String rrcaps = capsF(rrw);
		
		
		wordList.add(this._param_g.toFeature(network,FEATYPE.tag_currWord.name(), 	currTag,	w.toLowerCase()));
		wordList.add(this._param_g.toFeature(network,FEATYPE.tag_leftWord1.name(), 	currTag,	lw.toLowerCase()));
		wordList.add(this._param_g.toFeature(network,FEATYPE.tag_leftWord2.name(), 	currTag,	llw.toLowerCase()));
		wordList.add(this._param_g.toFeature(network,FEATYPE.tag_rightWord1.name(), currTag,	rw.toLowerCase()));
		wordList.add(this._param_g.toFeature(network,FEATYPE.tag_rightWord2.name(), currTag,	rrw.toLowerCase()));
		
		capList.add(this._param_g.toFeature(network, FEATYPE.tag_cap.name(), 	currTag,  caps));
		capList.add(this._param_g.toFeature(network, FEATYPE.tag_cap_l.name(), 	currTag,  lcaps));
		capList.add(this._param_g.toFeature(network, FEATYPE.tag_cap_ll.name(), currTag,  llcaps));
		capList.add(this._param_g.toFeature(network, FEATYPE.tag_cap_r.name(), 	currTag,  rcaps));
		capList.add(this._param_g.toFeature(network, FEATYPE.tag_cap_rr.name(),	currTag,  rrcaps));
		
		transitionList.add(this._param_g.toFeature(network, FEATYPE.tag_transition.name(),	currTag,  prevTag));
		
		if(task == TASK.TAGGING && cascade){
			String chunk = sent.get(pos).getEntity();
			cascadeList.add(this._param_g.toFeature(network, FEATYPE.chunk_currWord.name(),	currTag,  chunk));
		}
		
		
		if(NetworkConfig.USE_NEURAL_FEATURES && !removePOSNeural){
			if(windowSize==1)
				neuralList.add(this._param_g.toFeature(network,FEATYPE.neural.name(), currTag,  w.toLowerCase()));
			else if(windowSize==3)
				neuralList.add(this._param_g.toFeature(network,FEATYPE.neural.name(), currTag,  lw.toLowerCase()+IN_SEP+w.toLowerCase()
																							+IN_SEP+rw.toLowerCase()));
			else if(windowSize==5)
				neuralList.add(this._param_g.toFeature(network,FEATYPE.neural.name(), currTag,  llw.toLowerCase()+IN_SEP+
																							lw.toLowerCase()+IN_SEP+w.toLowerCase()
																							+IN_SEP+rw.toLowerCase()+IN_SEP+
																							rrw.toLowerCase()));
			else throw new RuntimeException("Unknown window size: "+windowSize);
		}
	}
	
	/**
	 * 
	 * @param featureList
	 * @param network
	 * @param sent
	 * @param pos
	 * @param paId
	 * @param parent_k
	 * @param children_k
	 * @param paTchildE: false means the current structure is NE structure.
	 */
	private void addJointFeatures(ArrayList<Integer> featureList, Network network, Sentence sent, int pos, int paId, 
			int parent_k, int[] children_k, boolean paTchildE, ArrayList<Integer> dstNodes){
		if(children_k.length!=1)
			throw new RuntimeException("The joint features should only have one children also");
		String currLabel = paTchildE? Tag.get(paId).getForm():Chunk.get(paId).getForm();
		int jf1, jf2, jf3, jf4, jf5;  
		int[] arr = null;
		int nodeType = -1;
		String lw = pos > 0? sent.get(pos-1).getForm():"#STR#";
		String llw = pos == 0? "#STR1#": pos==1? "#STR#":sent.get(pos-2).getForm();
		String rw = pos < sent.length()-1? sent.get(pos+1).getForm():"#END#";
		String rrw = pos==sent.length()-1? "#END1#": pos==sent.length()-2? "#END#":sent.get(pos+2).getForm();
		String w = sent.get(pos).getForm();
		if(!paTchildE){
			//current it's NE structure, need to refer to Tag node.
			nodeType = NODE_TYPES.TNODE.ordinal();
			for (int t = 0; t < Tag.TAGS_INDEX.size(); t++) {
				String tag = Tag.get(t).getForm();
				arr = new int[] { pos + 1, t, nodeType};
				long unlabeledDstNode = NetworkIDMapper.toHybridNodeID(arr);
				FCRFNetwork unlabeledNetwork = (FCRFNetwork) network.getUnlabeledNetwork();
				int unlabeledDstNodeIdx = Arrays.binarySearch(unlabeledNetwork.getAllNodes(), unlabeledDstNode);
				if (unlabeledDstNodeIdx >= 0) {
					if(windowSize >= 1) {
						jf1 = this._param_g.toFeature(network, FEATYPE.joint1.name(), currLabel + "&" + tag, w);
						if(jf1 != -1){
							featureList.add(jf1);
							dstNodes.add(unlabeledDstNodeIdx);
						}
					} 
					if (windowSize >=3) {
						jf2 = this._param_g.toFeature(network, FEATYPE.joint2.name(), currLabel + "&" + tag, lw);
						jf3 = this._param_g.toFeature(network, FEATYPE.joint3.name(), currLabel + "&" + tag, rw);
						if(jf2 != -1){
							featureList.add(jf2);
							dstNodes.add(unlabeledDstNodeIdx);
						}
						if(jf3 != -1){
							featureList.add(jf3);
							dstNodes.add(unlabeledDstNodeIdx);
						}
					}
					if (windowSize >= 5) {
						jf4 = this._param_g.toFeature(network, FEATYPE.joint4.name(), currLabel + "&" + tag, llw);
						jf5 = this._param_g.toFeature(network, FEATYPE.joint5.name(), currLabel + "&" + tag, rrw);
						if(jf4 != -1){
							featureList.add(jf4); 
							dstNodes.add(unlabeledDstNodeIdx);
						}
						if(jf5 != -1){
							featureList.add(jf5); 
							dstNodes.add(unlabeledDstNodeIdx);
						}
					} 
					if(NetworkConfig.USE_NEURAL_FEATURES && !removeJointNeural){
						int njf = -1;
						if(windowSize==1) 
							njf = this._param_g.toFeature(network,FEATYPE.neural.name(), currLabel + "&" + tag,  w.toLowerCase());
						else if(windowSize==3)
							njf =  this._param_g.toFeature(network,FEATYPE.neural.name(), currLabel + "&" + tag,  lw.toLowerCase()+IN_SEP+w.toLowerCase()
																										+IN_SEP+rw.toLowerCase());
						else if(windowSize==5)
							njf = this._param_g.toFeature(network,FEATYPE.neural.name(), currLabel + "&" + tag,  llw.toLowerCase()+IN_SEP+
																										lw.toLowerCase()+IN_SEP+w.toLowerCase()
																										+IN_SEP+rw.toLowerCase()+IN_SEP+
																										rrw.toLowerCase());
						else throw new RuntimeException("Unknown window size: "+windowSize);
						if (njf != -1) {
							featureList.add(njf);
							dstNodes.add(unlabeledDstNodeIdx);
						}
					}
				}
			}
			
		}else{
			//current it's POS structure, need to refer to chunk node
			nodeType = NODE_TYPES.ENODE.ordinal();
			for (int e = 0; e < Chunk.CHUNKS_INDEX.size(); e++) {
				String chunk = Chunk.get(e).getForm();
				if (this.iobes && pos == sent.length()-1 && (chunk.startsWith("B") || chunk.startsWith("I")) ) 
					continue;
				arr = new int[]{pos+1, e, nodeType};
				long unlabeledDstNode = NetworkIDMapper.toHybridNodeID(arr);
				FCRFNetwork unlabeledNetwork = (FCRFNetwork)network.getUnlabeledNetwork();
				int unlabeledDstNodeIdx = Arrays.binarySearch(unlabeledNetwork.getAllNodes(), unlabeledDstNode);
				if (unlabeledDstNodeIdx >= 0) {
					if(windowSize >= 1) {
						jf1 = this._param_g.toFeature(network, FEATYPE.joint1.name(), chunk + "&" + currLabel, w);
						if(jf1 != -1){
							featureList.add(jf1);
							dstNodes.add(unlabeledDstNodeIdx);
						}
					}
					if (windowSize >= 3) {
						jf2 = this._param_g.toFeature(network, FEATYPE.joint2.name(), chunk + "&" + currLabel, lw);
						jf3 = this._param_g.toFeature(network, FEATYPE.joint3.name(), chunk + "&" + currLabel, rw);
						if(jf2 != -1){
							featureList.add(jf2);
							dstNodes.add(unlabeledDstNodeIdx);
						}
						if(jf3 != -1){
							featureList.add(jf3);
							dstNodes.add(unlabeledDstNodeIdx);
						}
					} 
					if (windowSize >= 5) {
						jf4 = this._param_g.toFeature(network, FEATYPE.joint4.name(), chunk + "&" + currLabel, llw);
						jf5 = this._param_g.toFeature(network, FEATYPE.joint5.name(), chunk + "&" + currLabel, rrw);
						if(jf4 != -1){
							featureList.add(jf4); 
							dstNodes.add(unlabeledDstNodeIdx);
						}
						if(jf5 != -1){
							featureList.add(jf5); 
							dstNodes.add(unlabeledDstNodeIdx);
						}
					}
					if(NetworkConfig.USE_NEURAL_FEATURES && !removeJointNeural){
						int njf = -1;
						if(windowSize==1) 
							njf = this._param_g.toFeature(network,FEATYPE.neural.name(), chunk + "&" + currLabel,  w.toLowerCase());
						else if(windowSize==3)
							njf =  this._param_g.toFeature(network,FEATYPE.neural.name(), chunk + "&" + currLabel,  lw.toLowerCase()+IN_SEP+w.toLowerCase()
																										+IN_SEP+rw.toLowerCase());
						else if(windowSize==5)
							njf = this._param_g.toFeature(network,FEATYPE.neural.name(), chunk + "&" + currLabel,  llw.toLowerCase()+IN_SEP+
																										lw.toLowerCase()+IN_SEP+w.toLowerCase()
																										+IN_SEP+rw.toLowerCase()+IN_SEP+
																										rrw.toLowerCase());
						else throw new RuntimeException("Unknown window size: "+windowSize);
						if (njf != -1) {
							featureList.add(njf);
							dstNodes.add(unlabeledDstNodeIdx);
						}
					}
				}
			}
			
		}
			
		
	}
	
	
	private String capsF(String word){
		String cap = null;
		if(others.contains(word)) return "others";
		if(word.equals(word.toLowerCase())) cap = "all_lowercases";
		else if(word.equals(word.toUpperCase())) cap = "all_uppercases";
		else if(word.matches("[A-Z][a-z0-9]*")) cap = "first_upper";
		else if(word.matches("[a-z0-9]+[A-Z]+.*")) cap = "at_least_one";
		else cap = "others";
		return cap;
	}
}
