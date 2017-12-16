package org.statnlp.example.mention_hypergraph;

public class Span implements Comparable<Span>{
	
	public Label label;
	public int start;
	public int end;
	public int headStart;
	public int headEnd;

	public Span(int start, int end, int headStart, int headEnd, Label label) {
		this.start = start;
		this.end = end;
		this.headStart = headStart;
		this.headEnd = headEnd;
		this.label = label;
	}
	
	public boolean equals(Object o){
		if(o instanceof Span){
			Span s = (Span)o;
			if(start != s.start) return false;
			if(end != s.end) return false;
//			if(headStart != s.headStart) return false;
//			if(headEnd != s.headEnd) return false;
			return label.equals(s.label);
		}
		return false;
	}
	
	public int hashCode(){
		return Integer.hashCode(start) ^ Integer.hashCode(end);
	}

	@Override
	public int compareTo(Span o) {
		if(start < o.start) return -1;
		if(start > o.start) return 1;
		if(end < o.start) return -1;
		if(end > o.end) return 1;
		return label.compareTo(o.label);
	}
	
	public String toString(){
		return String.format("%d,%d,%d,%d %s", start, end, headStart, headEnd, label);
	}

}
