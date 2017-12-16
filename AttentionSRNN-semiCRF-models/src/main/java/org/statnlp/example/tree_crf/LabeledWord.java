package org.statnlp.example.tree_crf;

import java.io.Serializable;

public class LabeledWord implements Serializable{
	
	private static final long serialVersionUID = 3975391045469498758L;
	public Label label;
	public String word;

	public LabeledWord(Label label, String word) {
		this.label = label;
		this.word = word;
	}
	
	public LabeledWord(String label, String word){
		this.label = Label.get(label);
		this.word = word;
	}

}
