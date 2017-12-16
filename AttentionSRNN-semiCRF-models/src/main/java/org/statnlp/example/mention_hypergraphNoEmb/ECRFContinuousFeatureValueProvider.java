package org.statnlp.example.mention_hypergraphNoEmb;

import java.util.ArrayList;

import org.statnlp.hypergraph.neural.ContinuousFeatureValueProvider;



public class ECRFContinuousFeatureValueProvider extends ContinuousFeatureValueProvider {

	private static final long serialVersionUID = 7652212330450652821L;

	public ECRFContinuousFeatureValueProvider(int numFeatureValues, int numLabels) {
		super(numFeatureValues, numLabels);
	}
	
	public ECRFContinuousFeatureValueProvider(int numLabels) {
		super(numLabels);
	}

	@Override
	public double[] getFeatureValue(Object input) {
		String inputStr = (String)input;
		double fv[] = new double[50];
		ArrayList<Double> value = Embedding.getembed(inputStr);
		for(int i = 0; i < value.size(); i++) {
			fv[i] = value.get(i);
		}
		return fv;
	}
}