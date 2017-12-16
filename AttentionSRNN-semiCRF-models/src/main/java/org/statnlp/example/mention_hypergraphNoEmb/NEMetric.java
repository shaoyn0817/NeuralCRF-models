package org.statnlp.example.mention_hypergraphNoEmb;

import org.statnlp.hypergraph.decoding.Metric;

public class NEMetric implements Metric {

	double precision;
	double recall;
	double fscore;

	public NEMetric(double fscore) {
		this(-1, -1, fscore);
	}
	
	public NEMetric(double precision, double recall, double fscore) {
		this.precision = precision;
		this.recall = recall;
		this.fscore = fscore;
	}

	@Override
	public boolean isBetter(Metric other) {
		NEMetric metric = (NEMetric)other;
		return fscore > metric.fscore;
	}

	@Override
	public Double getMetricValue() {
		return this.fscore;
	}

	

}
