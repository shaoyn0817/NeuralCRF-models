package org.statnlp.example.tree_crf;

import java.util.Arrays;

import org.statnlp.example.base.BaseInstance;

public class TreeCRFInstance extends BaseInstance<TreeCRFInstance, String[], BinaryTree> {

	private static final long serialVersionUID = 2611698157784018704L;
	
	public TreeCRFInstance(int instanceId, double weight) {
		super(instanceId, weight);
	}
	
	public String[] duplicateInput(){
		return input == null ? null : Arrays.copyOf(input, input.length);
	}
	
	public BinaryTree duplicateOutput(){
		return output == null ? null : output.clone();
	}

	public BinaryTree duplicatePrediction(){
		return prediction == null ? null : prediction.clone();
	}

	@Override
	public int size() {
		return input.length;
	}

	public String toString(){
		if(output != null){
			return output.toString();
		} else {
			return Arrays.asList(input).toString();
		}
	}

}
