package org.statnlp.example.linear_ne;

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
		double[] fv = new double[2];
		double val2 = 0.2;
		if (inputStr.length() > 5){
			val2 = 0.8;
		}
		fv[0] = inputStr.length();
		fv[1] = val2;
		return fv;
	}

}
