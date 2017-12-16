package org.statnlp.example.linear_ne;

import org.statnlp.hypergraph.neural.NeuralNetworkCore;

public class ECRFMLP extends NeuralNetworkCore {

	private static final long serialVersionUID = -6457902817484777222L;

	public ECRFMLP(int numLabels) {
		super(numLabels);
		config.put("class", "MultiLayerPerceptron");
	}

	@Override
	public int hyperEdgeInput2OutputRowIndex(Object edgeInput) {
		return this.getNNInputID(edgeInput);
	}

	@Override
	public Object hyperEdgeInput2NNInput(Object edgeInput) {
		return edgeInput;
	}

}
