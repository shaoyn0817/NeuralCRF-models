package semi_crf;

import java.util.ArrayList;

import org.statnlp.example.base.BaseInstance;





public class semiCRFInstance extends BaseInstance<semiCRFInstance, ArrayList<String>, ArrayList<Span>>{
	
	private static final long serialVersionUID = 6415577909487373660L;
	
	public semiCRFInstance(int instanceId, double weight){
		super(instanceId, weight);
	}
	
	public semiCRFInstance(int instanceId, double weight, ArrayList<String> inputs, ArrayList<Span> outputs) {
		super(instanceId, weight);
		this.input = inputs;
		this.output = outputs;
	}
	
	public ArrayList<String> duplicateInput(){
		return input == null ? null : new ArrayList<String>(input);
	}
	
	public ArrayList<Span> duplicateOutput(){
		return output == null ? null : new ArrayList<Span>(output);
	}

	public ArrayList<Span> duplicatePrediction(){
		return prediction == null ? null : new ArrayList<Span>(prediction);
	}
	
	@Override
	public int size() {
		return this.input.size();
	}


	
}
