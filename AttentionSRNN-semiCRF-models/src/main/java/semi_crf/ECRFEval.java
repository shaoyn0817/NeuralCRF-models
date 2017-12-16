package semi_crf;


import java.io.IOException; 
import java.util.ArrayList;
import java.util.List;
import org.statnlp.commons.types.Instance;
import org.statnlp.hypergraph.decoding.Metric;

public class ECRFEval {
	
	public static boolean windows = false;
	
	/**
	 * 
	 * @param testInsts
	 * @param nerOut: word, true pos, true entity, pred entity
	 * @throws IOException
	 */
	public static Metric evalNER(Instance[] testInsts, String nerOut){
        double f = segmentEvaluation.eval(testInsts);
	    return new NEMetric(f);
	}
	
	
	
	
	
	 
}
