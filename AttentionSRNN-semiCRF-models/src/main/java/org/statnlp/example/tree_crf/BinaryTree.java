package org.statnlp.example.tree_crf;

import java.io.Serializable;
import java.security.InvalidParameterException;
import java.text.CharacterIterator;
import java.text.StringCharacterIterator;
import java.util.ArrayList;
import java.util.List;

import org.statnlp.example.tree_crf.Label.LabelType;

public class BinaryTree implements Serializable{
	
	private static final long serialVersionUID = 8898060016998116072L;
	public BinaryTree left;
	public BinaryTree right;
	public LabeledWord value;
	
	public LabeledWord[] leaves;
	public String[] words;
	
	public static final BinaryTree EMPTY = new BinaryTree();

	public BinaryTree() {}
	
	public BinaryTree clone(){
		BinaryTree result = new BinaryTree();
		if(left != null){
			result.left = left.clone();
		}
		if(right != null){
			result.right = right.clone();
		}
		result.value = value;
		return result;
	}
	
	public String[] getWords(){
		if(words == null){
			LabeledWord[] leaves = getLeaves();
			words = new String[leaves.length];
			for(int i=0; i<leaves.length; i++){
				words[i] = leaves[i].word;
			}
		}
		return words;
	}
	
	public void clearLeaves(){
		leaves = null;
		if(left != null) left.clearLeaves();
		if(right != null) right.clearLeaves();
	}
	
	public LabeledWord[] getLeaves(){
		if(leaves == null){
			fillLeaves();
		}
		return leaves;
	}
	
	public void fillLeaves(){
		if(left == null || right == null){
			leaves = new LabeledWord[]{value};
			return;
		}
		LabeledWord[] leftLeaves = left.getLeaves();
		LabeledWord[] rightLeaves = right.getLeaves();
		leaves = new LabeledWord[leftLeaves.length+rightLeaves.length];
		for(int i=0; i<leftLeaves.length; i++){
			leaves[i] = leftLeaves[i];
		}
		for(int i=0; i<rightLeaves.length; i++){
			leaves[i+leftLeaves.length] = rightLeaves[i];
		}
	}
	
	public String toString(){
		return toString(0);
	}
	
	public String toString(int level){
		if(value == null && (left == null || right == null)){
			return "-NULL-";
		}
		StringBuilder builder = new StringBuilder();
		if(left == null){
			for(int i=0; i<level; i++) builder.append("  ");
			if(value == null){
				builder.append("-NULL-");
			} else {
				builder.append("("+value.label.form+" "+value.word+")");
			}
		} else {
			for(int i=0; i<level; i++) builder.append("  ");
			builder.append("(");
			builder.append(value.label.form);
			builder.append("\n");
			if(left == null){
				for(int i=0; i<level+1; i++) builder.append("  ");
				builder.append("-NULL-");
			} else {
				builder.append(left.toString(level+1));
			}
			builder.append("\n");
			if(right == null){
				for(int i=0; i<level+1; i++) builder.append("  ");
				builder.append("-NULL-");
			} else {
				builder.append(right.toString(level+1));
			}
			builder.append(")");
		}
		return builder.toString();
	}
	
	/**
	 * Parse a serialized <strong>binary</strong> tree from the given string.
	 * Each tree should start with a bracket, followed by the label, then a space.
	 * Then followed by either:
	 * a. A terminal
	 * b. Two serialized trees in the same format
	 * And finally ended with a closing bracket.
	 * @param bracketedTree
	 * @return
	 */
	public static BinaryTree parse(String bracketedTree){
		CharacterIterator iterator = new StringCharacterIterator(bracketedTree);
		return parse(iterator);
	}
	
	/**
	 * Parse a serialized <strong>binary</strong> tree from the given CharacterIterator
	 * @param input
	 * @return
	 * @throws InvalidParameterException
	 */
	private static BinaryTree parse(CharacterIterator input) throws InvalidParameterException {
		BinaryTree result = new BinaryTree();
		StringBuilder read = new StringBuilder();
		try{
			char c;
			String label = "";
			while((c=next(read, input)) != ' '){
				label += c;
			}
			
			// Take only the base label (e.g., NP-SBJ-1 into NP)
			int dashIndex = label.indexOf("-");
			if(dashIndex != -1){
				label = label.substring(0, dashIndex);
			}
			Label labelObj = Label.get(label);
			if((c=next(read, input)) == '('){ // Another tree
				result.left = parse(input);
				result.right = parse(input);
				labelObj.type = LabelType.NON_TERMINAL;
				result.value = new LabeledWord(labelObj, "");
			} else { // Word
				String word = ""+c;
				while((c=next(read, input)) != ')'){
					word += c;
				}
				labelObj.type = LabelType.TERMINAL;
				result.value = new LabeledWord(labelObj, word);
			}
			input.next();
		} catch (RuntimeException e){
			System.out.println(getReadTree(read, input));
			throw e;
		}
		return result;
	}
	
	/**
	 * Get the next character in the iterator, saving it to the given StringBuilder while also returning it
	 * @param read
	 * @param rest
	 * @return
	 */
	private static char next(StringBuilder read, CharacterIterator rest){
		char c = rest.next();
		if(c != CharacterIterator.DONE) read.append(c);
		return c;
	}
	
	/**
	 * Return the tree given to the parser
	 * @param read
	 * @param rest
	 * @return
	 */
	private static String getReadTree(StringBuilder read, CharacterIterator rest){
		char c;
		while((c=rest.next()) != CharacterIterator.DONE) read.append(c);
		return read.toString();
	}
	
	/**
	 * Return the list of constituents (phrases) of this tree
	 * @return
	 */
	public List<String> getConstituents(){
		ArrayList<String> result = new ArrayList<String>();
		if(left == null){
			result.add(String.format("(%s %s)", value.label.form, value.word));
		} else {
			result.addAll(left.getConstituents());
			result.addAll(right.getConstituents());
			String constituent = "";
			for(String word: getWords()){
				if(constituent.length() != 0) constituent += " ";
				constituent += word;
			}
			result.add(String.format("(%s %s)", value.label.form, constituent));
		}
		return result;
	}

}
