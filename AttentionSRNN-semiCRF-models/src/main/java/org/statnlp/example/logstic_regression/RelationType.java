package org.statnlp.example.logstic_regression;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

public class RelationType implements Comparable<RelationType>, Serializable{
	
	private static final long serialVersionUID = -3314363044582374266L;
	public static final Map<String, RelationType> RELS = new HashMap<String, RelationType>();
	public static final Map<Integer, RelationType> RELS_INDEX = new HashMap<Integer, RelationType>();
	private static boolean locked = false;
	
	public static RelationType get(String form){
		if(!RELS.containsKey(form)){
			if (!locked) {
				RelationType label = new RelationType(form, RELS.size());
				RELS.put(form, label);
				RELS_INDEX.put(label.id, label);
			} else {
				throw new RuntimeException("the map is locked");
			}
		}
		return RELS.get(form);
	}
	
	public static RelationType get(int id){
		if (!RELS_INDEX.containsKey(id))
			throw new RuntimeException("the map does not have id: "+ id);
		return RELS_INDEX.get(id);
	}
	
	public static void lock () {
		locked = true;
	}
	
	public String form;
	public int id;
	
	private RelationType(String form, int id) {
		this.form = form;
		this.id = id;
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
		if (!(obj instanceof RelationType))
			return false;
		RelationType other = (RelationType) obj;
		if (form == null) {
			if (other.form != null)
				return false;
		} else if (!form.equals(other.form))
			return false;
		return true;
	}
	
	public String toString(){
		return String.format("%s(%d)", form, id);
	}

	@Override
	public int compareTo(RelationType o) {
		return Integer.compare(id, o.id);
	}
	
	public static int compare(RelationType o1, RelationType o2){
		if(o1 == null){
			if(o2 == null) return 0;
			else return -1;
		} else {
			if(o2 == null) return 1;
			else return o1.compareTo(o2);
		}
	}
}
