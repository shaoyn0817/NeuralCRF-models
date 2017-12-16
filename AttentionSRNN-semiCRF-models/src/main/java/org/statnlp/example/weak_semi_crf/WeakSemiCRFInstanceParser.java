/**
 * 
 */
package org.statnlp.example.weak_semi_crf;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.statnlp.commons.types.Instance;
import org.statnlp.commons.types.Label;
import org.statnlp.commons.types.LinearInstance;
import org.statnlp.commons.types.Span;
import org.statnlp.util.Pipeline;
import org.statnlp.util.instance_parser.InstanceParser;

/**
 * 
 */
public class WeakSemiCRFInstanceParser extends InstanceParser {

	private static final long serialVersionUID = 4072193433227503755L;
	
	public final Map<String, Label> LABELS = new HashMap<String, Label>();
	public final Map<Integer, Label> LABELS_INDEX = new HashMap<Integer, Label>();
	
	public Label getLabel(String form){
		if(!LABELS.containsKey(form)){
			Label label = new Label(form, LABELS.size());
			LABELS.put(form, label);
			LABELS_INDEX.put(label.getId(), label);
		}
		return LABELS.get(form);
	}
	
	public Label getLabel(int id){
		return LABELS_INDEX.get(id);
	}
	
	public void reset(){
		LABELS.clear();
		LABELS_INDEX.clear();
	}
	
	public boolean COMBINE_OUTSIDE_CHARS = true;
	public boolean USE_SINGLE_OUTSIDE_TAG = true;

	/**
	 * @param pipeline
	 */
	public WeakSemiCRFInstanceParser(Pipeline<?> pipeline) {
		super(pipeline);
		try{
			if(pipeline.hasParameter("combineOutsideChars")){
				COMBINE_OUTSIDE_CHARS = Boolean.parseBoolean(pipeline.getParameter("combineOutsideChars"));
			}
		} catch (Exception e){}
		try{
			if(pipeline.hasParameter("useSingleOutsideTag")){
				USE_SINGLE_OUTSIDE_TAG = Boolean.parseBoolean(pipeline.getParameter("useSingleOutsideTag"));
			}
		} catch (Exception e){}
	}

	/* (non-Javadoc)
	 * @see com.statnlp.util.instance_parser.InstanceParser#buildInstances(java.lang.String[])
	 */
	@SuppressWarnings("unchecked")
	@Override
	public LinearInstance<Span>[] buildInstances(String... sources) throws FileNotFoundException {
		List<Instance> instances = new ArrayList<Instance>();
		for(String source: sources){
			try {
				instances.addAll(Arrays.asList(readData(source)));
			} catch (IOException e) {
				throw new RuntimeException(e);
			}
		}
		return instances.toArray(new LinearInstance[0]);
	}
	
	/**
	 * Read data from a file with three-line format:<br>
	 * - First line the input string<br>
	 * - Second line the list of spans in the format "start,end Label" separated by pipe "|"<br>
	 * - Third line an empty line
	 * @param fileName
	 * @param isLabeled
	 * @return
	 * @throws IOException
	 */
	@SuppressWarnings("unchecked")
	private LinearInstance<Span>[] readData(String fileName) throws IOException{
		InputStreamReader isr = new InputStreamReader(new FileInputStream(fileName), "UTF-8");
		BufferedReader br = new BufferedReader(isr);
		ArrayList<LinearInstance<Span>> result = new ArrayList<LinearInstance<Span>>();
		String input = null;
		List<String[]> inputTokenized = null;
		List<Span> output = null;
		int instanceId = 1;
		while(br.ready()){
			input = br.readLine();
			inputTokenized = new ArrayList<String[]>();
			for(int i=0; i<input.length(); i++){
				inputTokenized.add(new String[]{input.substring(i,i+1)});
			}
			int length = input.length();
			output = new ArrayList<Span>();
			String[] spansStr = br.readLine().split("\\|");
			List<Span> spans = new ArrayList<Span>();
			for(String span: spansStr){
				if(span.length() == 0){
					continue;
				}
				String[] startend_label = span.split(" ");
				Label label = getLabel(startend_label[1]);
				String[] start_end = startend_label[0].split(",");
				int start = Integer.parseInt(start_end[0]);
				int end = Integer.parseInt(start_end[1]);
				spans.add(new Span(start, end, label));
			}
			Collections.sort(spans); // Ensure it is sorted
			
			int prevEnd = 0;
			for(Span span: spans){
				int start = span.start;
				int end = span.end;
				Label label = span.label;
				if(prevEnd < start){
					createOutsideSpans(input, output, prevEnd, start);
				}
				prevEnd = end;
				output.add(new Span(start, end, label));
			}
			createOutsideSpans(input, output, prevEnd, length);
			LinearInstance<Span> instance = new LinearInstance<Span>(instanceId, 1.0, inputTokenized, output);
			result.add(instance);
			instanceId += 1;
			br.readLine();
		}
		br.close();
		return result.toArray(new LinearInstance[result.size()]);
	}

	
	/**
	 * Create the outside spans in the specified substring
	 * @param input
	 * @param output
	 * @param start
	 * @param end
	 */
	private void createOutsideSpans(String input, List<Span> output, int start, int end){
		int length = input.length();
		int curStart = start;
		while(curStart < end){
			int curEnd = input.indexOf(' ', curStart);
			Label outsideLabel = null;
			if(USE_SINGLE_OUTSIDE_TAG){
				outsideLabel = getLabel("O");
				if(curEnd == -1 || curEnd > end){
					curEnd = end;
				} else if(curStart == curEnd){
					curEnd += 1;
				}
			} else {
				if(curEnd == -1 || curEnd > end){ // No space
					curEnd = end;
					if(curStart == start){ // Start directly after previous tag: this is between tags
						if(curStart == 0){ // Unless this is the start of the string
							if(curEnd == length){
								outsideLabel = getLabel("O"); // Case |<cur>|
							} else {
								outsideLabel = getLabel("O-B"); // Case |<cur>###
							}
						} else {
							if(curEnd == length){
								outsideLabel = getLabel("O-A"); // Case ###<cur>|
							} else {
								outsideLabel = getLabel("O-I"); // Case ###<cur>###
							}
						}
					} else { // Start not immediately: this is before tags (found space before)
						if(curEnd == length){
							outsideLabel = getLabel("O"); // Case ### <cur>|
						} else {
							outsideLabel = getLabel("O-B"); // Case ### <cur>###
						}
					}
				} else if(curStart == curEnd){ // It is immediately a space
					curEnd += 1;
					outsideLabel = getLabel("O"); // Tag space as a single outside token
				} else if(curStart < curEnd){ // Found a non-immediate space
					if(curStart == start){ // Start immediately after previous tag: this is after tag
						if(curStart == 0){
							outsideLabel = getLabel("O"); // Case |<cur> ###
						} else {
							outsideLabel = getLabel("O-A"); // Case ###<cur> ###
						}
					} else { // Start not immediately: this is a separate outside token
						outsideLabel = getLabel("O"); // Case ### <cur> ###
					}
				}
			}
			output.add(new Span(curStart, curEnd, outsideLabel));
			curStart = curEnd;
		}
	}

}
