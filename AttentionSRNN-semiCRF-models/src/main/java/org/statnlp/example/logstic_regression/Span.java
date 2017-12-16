package org.statnlp.example.logstic_regression;


public class Span implements Comparable<Span>{

	public String entity;
	public int start;
	public int end;
	
	/**
	 * Span constructor
	 * @param start: inclusive
	 * @param end: inclusive
	 * @param entity
	 */
	public Span(int start, int end, String entity) {
		this.start = start;
		this.end = end;
		this.entity = entity;
	}
	
	public boolean equals(Object o){
		if(o instanceof Span){
			Span s = (Span)o;
			if(start != s.start) return false;
			if(end != s.end) return false;
			return entity.equals(s.entity);
		}
		return false;
	}

	@Override
	public int compareTo(Span o) {
		if(start < o.start) return -1;
		if(start > o.start) return 1;
		if(end < o.end) return -1;
		if(end > o.end) return 1;
		return entity.compareTo(o.entity);
	}
	
	public int comparePosition(Span o) {
		if(start < o.start) return -1;
		if(start > o.start) return 1;
		if(end < o.end) return -1;
		if(end > o.end) return 1;
		return 0;
	}
	
	public boolean overlap(Span other) {
		if (other.start > this.end) return false;
		if (other.end < this.start) return false;
		return true;
	}
	
	public String toString(){
		return String.format("%d,%d %s", start, end, entity);
	}
}
