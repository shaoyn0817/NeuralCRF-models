package org.statnlp.example.logstic_regression;

import java.util.List;

import org.statnlp.commons.types.Sentence;
import org.statnlp.example.base.BaseInstance;
import org.statnlp.hypergraph.NetworkException;

public class LRInstance extends BaseInstance<LRInstance, LGInput, RelationType> {

	private static final long serialVersionUID = 6971863806062822950L;
	
	public LRInstance(int instanceId, double weight) {
		super(instanceId, weight);
	}

	public LRInstance(int instanceId, double weight, LGInput input, RelationType output) {
		this(instanceId, weight);
		this.input = input;
		this.output = output;
	}


	@Override
	public int size() {
		throw new NetworkException ("The instance does not have size.");
	}
	
	public LGInput duplicateInput(){
		return input;
	}
	
	public RelationType duplicateOutput() {
		return this.output;
	}

}

class LGInput {

	protected Sentence sent;
	
	/**
	 * Only consider the entity span.
	 */
	protected List<Span> spans;
	
	/**
	 * The index is the index in the span.
	 */
	protected int leftSpanIdx;
	protected int rightSpanIdx;
	
	/**
	 * Input Class to logistic regression model.
	 * @param sent
	 * @param spans
	 * @param leftSpanIdx: may not be the arg1Idx, depends on the label
	 * @param rightSpanIdx: may not be the arg2idx, depends on the label
	 */
	public LGInput(Sentence sent, List<Span> spans, int leftSpanIdx, int rightSpanIdx) {
		this.sent = sent;
		this.spans = spans;
		this.leftSpanIdx = leftSpanIdx;
		this.rightSpanIdx = rightSpanIdx;
	}

}

