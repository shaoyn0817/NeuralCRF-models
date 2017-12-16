package org.statnlp.example.tagging;

import java.awt.geom.Point2D;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Map;

import org.statnlp.commons.types.Label;
import org.statnlp.commons.types.Sentence;
import org.statnlp.example.tagging.TagNetworkCompiler.NodeType;
import org.statnlp.ui.visualize.type.VNode;
import org.statnlp.ui.visualize.type.VisualizationViewerEngine;
import org.statnlp.ui.visualize.type.VisualizeGraph;
import org.statnlp.util.instance_parser.InstanceParser;

public class TaggingViewer extends VisualizationViewerEngine {

	
	
	static double span_width = 100;

	static double span_height = 100;
	
	static double offset_width = 100;
	
	static double offset_height = 100;
	
	/**
	 * The list of instances to be visualized
	 */
	protected TagInstance instance;
	
	protected Sentence inputs;
	
	protected ArrayList<String> outputs;
	
	/**
	 * The list of labels to be used in the visualization.<br>
	 * This member is used to support visualization on those codes that do not utilize
	 * the generic pipeline. In the pipeline, use {@link #instanceParser} instead.
	 */
	protected Map<Integer, Label> labels;
	
	public TaggingViewer(Map<Integer, Label> labels){
		super(null);
		this.labels = labels;
	}
	
	public TaggingViewer(InstanceParser instanceParser) {
		super(instanceParser);
	}
	
	
	@SuppressWarnings("unchecked")
	protected void initData()
	{
		this.instance = (TagInstance)super.instance;
		this.inputs = (Sentence)super.inputs;
		this.outputs = (ArrayList<String>)super.outputs;
	}
	
	@Override
	protected String label_mapping(VNode node) {
		int[] ids = node.ids;
		String label =  Arrays.toString(ids);
		
		return  label;
	}
	
	protected void initNodeColor(VisualizeGraph vg)
	{
		if (colorMap != null){
			for(VNode node : vg.getNodes())
			{
				int[] ids = node.ids;
				int nodeType = ids[2];
				if(nodeType == NodeType.leaf.ordinal()){
					
					node.color = colorMap[0];
					
					
				} else if (nodeType == NodeType.tag.ordinal()) {
					node.color = colorMap[1];
				}
				else
				{ //root
					node.color = colorMap[2];
				}
			}
		}
		
	}
	
	protected void initNodeCoordinate(VisualizeGraph vg)
	{
		for(VNode node : vg.getNodes())
		{
			int[] ids = node.ids;
//			int size = this.inputs.size();
			int pos = ids[0];
			int labelId = ids[1];
			int nodeType = ids[2];
			
			double x = pos * span_width;
			int mappedId = labelId;
			
			double y = mappedId * span_height + offset_height;
			
			if(nodeType == NodeType.root.ordinal()){
				x = (pos + 1) * span_width;
				y = 3 * span_height + offset_height;
			}
			else if (nodeType == NodeType.leaf.ordinal()){
				x = (-1) * span_width;
				y = 3 * span_height + offset_height;
			}
			node.point = new Point2D.Double(x,y);
			layout.setLocation(node, node.point);
			layout.lock(node, true);
		}
	}
	

}
