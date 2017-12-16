package semi_crf;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;


public class Label implements Serializable{
	
	private static final long serialVersionUID = -5006849791095171763L;
	
	public static final Map<String, Label> LABELS = new HashMap<String, Label>();
	public static final Map<Integer, Label> LABELS_INDEX = new HashMap<Integer, Label>();
	
	public static Label get(String form){
		if(!LABELS.containsKey(form)){
			Label label = new Label(form, LABELS.size());
			LABELS.put(form, label);
			LABELS_INDEX.put(label._id, label);
		}
		
		return LABELS.get(form);
	}
	
	public static Label get(int id){
		return LABELS_INDEX.get(id);
	}
	
	public static void reset(){
		LABELS.clear();
		LABELS_INDEX.clear();
	}
	
	public String _form;
	public int _id;
	
	public Label(Label lbl){
		this._form = lbl._form;
		this._id = lbl._id;
	}
	
	private Label(String form, int id){
		this._form = form;
		this._id = id;
	}
	
	public void setId(int id){
		this._id = id;
	}
	
	public int getId(){
		return this._id;
	}
	
	public String getForm(){
		return this._form;
	}
	
	public boolean equals(Object o){
		if(o instanceof Label){
			Label l = (Label)o;
			return this._form.equals(l._form);
		}
		return false;
	}
	
	public int hashCode(){
		return _form.hashCode();
	}
	
	public String toString(){
		return _form;
	}
	
}
