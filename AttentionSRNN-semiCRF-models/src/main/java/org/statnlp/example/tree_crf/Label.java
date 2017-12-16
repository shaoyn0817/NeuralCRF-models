package org.statnlp.example.tree_crf;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

public class Label implements Comparable<Label>, Serializable{
	
	private static final long serialVersionUID = -3314363044582374266L;
	public static final Map<String, Label> LABELS = new HashMap<String, Label>();
	public static final Map<Integer, Label> LABELS_INDEX = new HashMap<Integer, Label>();
	
	public static Label get(String form){
		if(!LABELS.containsKey(form)){
			Label label = new Label(form, LABELS.size());
			LABELS.put(form, label);
			LABELS_INDEX.put(label.id, label);
		}
		return LABELS.get(form);
	}
	
	public static Label get(int id){
		return LABELS_INDEX.get(id);
	}
	
	public enum LabelType{
		TERMINAL,
		NON_TERMINAL,
	}
	
	public String form;
	public int id;
	public LabelType type;
	
	private Label(String form, int id) {
		this.form = form;
		this.id = id;
		this.type = LabelType.TERMINAL;
	}

	@Override
	public int hashCode() {
		return form.hashCode();
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (!(obj instanceof Label))
			return false;
		Label other = (Label) obj;
		if (form == null) {
			if (other.form != null)
				return false;
		} else if (!form.equals(other.form))
			return false;
		return true;
	}
	
	public String toString(){
		return String.format("%s(%d)-%s", form, id, type);
	}

	@Override
	public int compareTo(Label o) {
		return Integer.compare(id, o.id);
	}
	
	public static int compare(Label o1, Label o2){
		if(o1 == null){
			if(o2 == null) return 0;
			else return -1;
		} else {
			if(o2 == null) return 1;
			else return o1.compareTo(o2);
		}
	}
}
