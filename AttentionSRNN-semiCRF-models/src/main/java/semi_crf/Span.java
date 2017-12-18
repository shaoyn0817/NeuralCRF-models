package semi_crf;

public class Span {
	public Label _label;
	public int _end;
	public int _start;
	
	public Span(int end, int Id){
		this._end = end;
		this._label = Label.get(Id);
	}

	public Span(int start, int end, int Id){
		this._end = end;
		this._label = Label.get(Id);
	}
}
