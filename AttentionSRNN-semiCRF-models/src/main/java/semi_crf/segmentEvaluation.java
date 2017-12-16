package semi_crf;


import java.util.ArrayList;

import org.statnlp.commons.types.Instance;

import semi_crf.*;
import semi_crf.Span;
import semi_crf.semiCRFInstance;

public class segmentEvaluation {
	
    public segmentEvaluation(Instance[] instances) {
    	int labelNum = 0;
    	int all[] = new int[13];
    	int find[] = new int[13];
    	int right[] = new int[13];
    	int sumlabels = 0;
    	int instanceNum = 0;
        double micro = 0;
    	for(int i = 0; i < instances.length; i++){
    		semiCRFInstance instance = (semiCRFInstance) instances[i];
    		ArrayList<Span> originOutput = instance.getPrediction();
    		ArrayList<Span> originAnswer = instance.getOutput();
    		Integer output[] = recover(originOutput);
    		Integer answer[] = recover(originAnswer);
    		int outlabel = -1;
    		int answerlabel = -1;
    		ArrayList<String> countOut = new ArrayList<String>();
    		ArrayList<String> countAnswer = new ArrayList<String>();
    		int start = -1;
    		for(int m = 0; m < output.length; m++){
    			if(outlabel != output[m]){
    				if(outlabel==-1)
    					{
    					    start = 0;
    					    outlabel = output[m];
    					}
    				else{
    					int end = m-1;
    			        countOut.add(start+":"+end+":"+outlabel);
    					outlabel = output[m];
    					start = m;
    				}
    			}
    		}
    		
    		for(int m = 0; m < answer.length; m++){
    			if(answerlabel != answer[m]){
    				if(answerlabel == -1)
    					{
    					    start = 0;
    					    answerlabel = answer[m];
    					}
    				else{
    					int end = m-1;
    			        countAnswer.add(start+":"+end+":"+answerlabel);
    					answerlabel = answer[m];
    					start = m;
    				}
    			}
    		}
    		
    		
    		//check correctness
    		int microfind[] = new int[13];
    		int microall[] = new int[13];
    		int microright[] = new int[13];
    		for(int m = 0; m < countOut.size(); m++){
    			String parts[] = countOut.get(m).split(":");
    			int begin = Integer.valueOf(parts[0]);
    			int end = Integer.valueOf(parts[1]);
    			int label = Integer.valueOf(parts[2]);
    			find[label]++;
    			
    			for(int n = 0; n < countAnswer.size(); n++){
    				String parts2[] = countAnswer.get(n).split(":");
        			int begin2 = Integer.valueOf(parts2[0]);
        			int end2 = Integer.valueOf(parts2[1]);
        			int label2 = Integer.valueOf(parts2[2]);
        			if(begin == begin2 && end == end2 && label == label2){
        				right[label]++;
        				break;
        			}
    			}		
    		}
    		
    		
    		
    		for(int n = 0; n < countAnswer.size(); n++){
				String parts2[] = countAnswer.get(n).split(":");
    			int label2 = Integer.valueOf(parts2[2]);
    			all[label2]++;
    			sumlabels++;
    		}	
    		
    		if(countAnswer.size() != countOut.size())
    			instanceNum++;
    		else {
    			int flag = 0;
    		    for(int m = 0; m < countAnswer.size(); m++){
    		    	if(!countAnswer.get(m).equals(countOut.get(m))){
    		    		flag = 1;
    		    		break;
    		    	}
    		    }	
    		    if(flag == 1)
    		    	instanceNum++;
    		    else
    		    	micro++;
    		}
    	}
    	
    	double microprecision = (double)micro/instanceNum;
    	//System.out.println("微观准确率为:"+microprecision);
    	//System.out.println("微观召回率:"+(double)micro/instances.length);
    	//System.out.println("以下是基于segment的evaluation,共有label："+sumlabels);
    	int ra = 0;
    	int fa = 0;
    	int aa = 0;
    	for(int i = 0; i < semi_crf.Label.LABELS.size(); i++){
    		if(semi_crf.Label.get(i)._form.equals('O'))
    			continue;
    		ra += right[i];
    		fa += find[i];
    		aa += all[i];
    		double pre = (double)right[i]/find[i];
    		double rec = (double)right[i]/all[i];
    		System.out.println(semi_crf.Label.get(i)._form+"   Precision:"+pre
    				+"   Recall:"+rec+ "   F1:"+2*pre*rec/(pre+rec));
    	}
	} 
    
    public static Integer[] recover(ArrayList<Span> output){
    	ArrayList<Integer> result = new ArrayList<Integer>();
    	int curr = -1;
    	for(int i = 0; i < output.size(); i ++){
    		int len = output.get(i)._end - curr;
            for(int j = 0; j < len; j++){
            	result.add(output.get(i)._label._id);
            }
            curr = output.get(i)._end;
    	}
    	Integer a[] = new Integer[result.size()];
    	result.toArray(a);
    	return a;
    }
    
    
    public static double eval(Instance[] instances) {
    	int labelNum = 0;
    	int all[] = new int[13];
    	int find[] = new int[13];
    	int right[] = new int[13];
    	int sumlabels = 0;
    	int instanceNum = 0;
        double micro = 0;
    	for(int i = 0; i < instances.length; i++){
    		semiCRFInstance instance = (semiCRFInstance) instances[i];
    		ArrayList<Span> originOutput = instance.getPrediction();
    		ArrayList<Span> originAnswer = instance.getOutput();
    		Integer output[] = recover(originOutput);
    		Integer answer[] = recover(originAnswer);
    		int outlabel = -1;
    		int answerlabel = -1;
    		ArrayList<String> countOut = new ArrayList<String>();
    		ArrayList<String> countAnswer = new ArrayList<String>();
    		int start = -1;
    		for(int m = 0; m < output.length; m++){
    			if(outlabel != output[m]){
    				if(outlabel==-1)
    					{
    					    start = 0;
    					    outlabel = output[m];
    					}
    				else{
    					int end = m-1;
    			        countOut.add(start+":"+end+":"+outlabel);
    					outlabel = output[m];
    					start = m;
    				}
    			}
    		}
    		
    		for(int m = 0; m < answer.length; m++){
    			if(answerlabel != answer[m]){
    				if(answerlabel == -1)
    					{
    					    start = 0;
    					    answerlabel = answer[m];
    					}
    				else{
    					int end = m-1;
    			        countAnswer.add(start+":"+end+":"+answerlabel);
    					answerlabel = answer[m];
    					start = m;
    				}
    			}
    		}
    		
    		
    		//check correctness
    		int microfind[] = new int[13];
    		int microall[] = new int[13];
    		int microright[] = new int[13];
    		for(int m = 0; m < countOut.size(); m++){
    			String parts[] = countOut.get(m).split(":");
    			int begin = Integer.valueOf(parts[0]);
    			int end = Integer.valueOf(parts[1]);
    			int label = Integer.valueOf(parts[2]);
    			find[label]++;
    			
    			for(int n = 0; n < countAnswer.size(); n++){
    				String parts2[] = countAnswer.get(n).split(":");
        			int begin2 = Integer.valueOf(parts2[0]);
        			int end2 = Integer.valueOf(parts2[1]);
        			int label2 = Integer.valueOf(parts2[2]);
        			if(begin == begin2 && end == end2 && label == label2){
        				right[label]++;
        				break;
        			}
    			}		
    		}
    		
    		
    		
    		for(int n = 0; n < countAnswer.size(); n++){
				String parts2[] = countAnswer.get(n).split(":");
    			int label2 = Integer.valueOf(parts2[2]);
    			all[label2]++;
    			sumlabels++;
    		}	
    		
    		if(countAnswer.size() != countOut.size())
    			instanceNum++;
    		else {
    			int flag = 0;
    		    for(int m = 0; m < countAnswer.size(); m++){
    		    	if(!countAnswer.get(m).equals(countOut.get(m))){
    		    		flag = 1;
    		    		break;
    		    	}
    		    }	
    		    if(flag == 1)
    		    	instanceNum++;
    		    else
    		    	micro++;
    		}
    	}
    	
    	double microprecision = (double)micro/instanceNum;
    	//System.out.println("微观准确率为:"+microprecision);
    	//System.out.println("微观召回率:"+(double)micro/instances.length);
    	//System.out.println("以下是基于segment的evaluation,共有label："+sumlabels);
    	int ra = 0;
    	int fa = 0;
    	int aa = 0;
    	for(int i = 0; i < semi_crf.Label.LABELS.size(); i++){
    		if(semi_crf.Label.get(i)._form.equals('O'))
    			continue;
    		ra += right[i];
    		fa += find[i];
    		aa += all[i];
    		double pre = (double)right[i]/find[i];
    		double rec = (double)right[i]/all[i];
    		System.out.println(semi_crf.Label.get(i)._form+"   Precision:"+pre
    				+"   Recall:"+rec+ "   F1:"+2*pre*rec/(pre+rec));
    	}

    	double pre = (double)ra/fa;
    	double rec = (double)ra/aa;
    	System.out.println("overall  Precision:" +pre+"   Recall:"+rec+"   F1:"+2*pre*rec/(pre+rec));
    	return 2*pre*rec/(pre+rec);
	} 
}
