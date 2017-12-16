package org.statnlp.example.linear_ne;

import org.statnlp.hypergraph.neural.NeuralNetworkCore;

public class EmbeddingLayer extends NeuralNetworkCore {

	private static final long serialVersionUID = 4951822203204790448L;

	public EmbeddingLayer(int numLabels) {
		super(numLabels);
		config.put("class", "EmbeddingLayer");
        config.put("hiddenSize", 30);
        config.put("embedding", "random");
        config.put("fixEmbedding", false);
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
