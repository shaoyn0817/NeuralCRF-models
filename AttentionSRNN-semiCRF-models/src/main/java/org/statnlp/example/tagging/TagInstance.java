package org.statnlp.example.tagging;

import java.util.List;

import org.statnlp.commons.types.Sentence;
import org.statnlp.example.base.BaseInstance;

public class TagInstance extends BaseInstance<TagInstance, Sentence, List<String>> {

	public TagInstance(int instanceId, double weight) {
		this(instanceId, weight, null, null);
	}
	
	public TagInstance(int instanceId, double weight, Sentence sent, List<String> output) {
		super(instanceId, weight);
		this.input = sent;
		this.output = output;
	}

	private static final long serialVersionUID = 1L;

	@Override
	public int size() {
		return this.input.length();
	}

	public Sentence duplicateInput(){
		return this.input;
	}
	
}
