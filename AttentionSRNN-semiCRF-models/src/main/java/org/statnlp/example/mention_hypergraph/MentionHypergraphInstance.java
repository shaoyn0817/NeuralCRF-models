package org.statnlp.example.mention_hypergraph;

import java.io.Serializable;
import java.util.List;

import org.statnlp.example.base.BaseInstance;
import org.statnlp.example.mention_hypergraph.MentionHypergraphInstance.WordsAndTags;

public class MentionHypergraphInstance extends BaseInstance<MentionHypergraphInstance, WordsAndTags, List<Span>>{
	
	public static class WordsAndTags implements Serializable{
		private static final long serialVersionUID = -9022041107687594823L;
		public AttributedWord[] words;
		public String[] posTags;
		public WordsAndTags(WordsAndTags other){
			this.words = other.words;
			this.posTags = other.posTags;
		}
		public WordsAndTags(AttributedWord[] words, String[] posTags){
			this.words = words;
			this.posTags = posTags;
		}
	}
	
	private static final long serialVersionUID = -9133939568122739620L;
	
	public MentionHypergraphInstance(int instanceId, double weight){
		super(instanceId, weight);
	}

	@Override
	public int size() {
		return input.words.length;
	}

	public String toString(){
		StringBuilder builder = new StringBuilder();
		builder.append(getInstanceId()+":");
		for(int i=0; i<input.words.length; i++){
			if(i > 0) builder.append(" ");
			builder.append(input.words[i]+"/"+input.posTags[i]);
		}
		if(hasOutput()){
			builder.append("\n");
			for(Span span: output){
				builder.append(span+"|");
			}
		}
		return builder.toString();
	}
	
}
