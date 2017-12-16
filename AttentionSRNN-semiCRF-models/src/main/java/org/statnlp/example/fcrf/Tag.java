/** Statistical Natural Language Processing System
    Copyright (C) 2014-2016  Lu, Wei

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
/**
 * 
 */
package org.statnlp.example.fcrf;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

/**
 * @author wei_lu
 *
 */
public class Tag implements Serializable{
	
	private static final long serialVersionUID = -5006849791095171763L;
	
	private static boolean locked = false;
	public static final Map<String, Tag> TAGS = new HashMap<String, Tag>();
	public static final Map<Integer, Tag> TAGS_INDEX = new HashMap<Integer, Tag>();
	
	public static Tag get(String form){
		if(!TAGS.containsKey(form)){
			if(locked) 
				throw new RuntimeException("Unknown tag type:"+form);
			Tag label = new Tag(form, TAGS.size());
			TAGS.put(form, label);
			TAGS_INDEX.put(label._id, label);
		}
		return TAGS.get(form);
	}
	
	public static void lock(){locked = true;}
	public static Tag get(int id){
		return TAGS_INDEX.get(id);
	}
	
	private String _form;
	private int _id;
	
	public Tag(Tag lbl){
		this._form = lbl._form;
		this._id = lbl._id;
	}
	
	private Tag(String form, int id){
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
		if(o instanceof Tag){
			Tag l = (Tag)o;
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
