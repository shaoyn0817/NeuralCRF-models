package org.statnlp.example.linear_crf;

import java.awt.geom.Point2D;
import java.util.ArrayList;
import java.util.Map;

import org.statnlp.commons.types.Label;
import org.statnlp.commons.types.LinearInstance;
import org.statnlp.example.linear_crf.LinearCRFNetworkCompiler.NODE_TYPES;
import org.statnlp.ui.visualize.type.VNode;
import org.statnlp.ui.visualize.type.VisualizationViewerEngine;
import org.statnlp.ui.visualize.type.VisualizeGraph;
import org.statnlp.util.instance_parser.DelimiterBasedInstanceParser;
import org.statnlp.util.instance_parser.InstanceParser;



public class LinearCRFViewer extends VisualizationViewerEngine {
	
	static double span_width = 100;

	static double span_height = 100;
	
	static double offset_width = 100;
	
	static double offset_height = 100;
	
	/**
	 * The list of instances to be visualized
	 */
	protected LinearInstance<Label> instance;
	
	protected ArrayList<String[]> inputs;
	
	protected ArrayList<Label> outputs;
	
	/**
	 * The list of labels to be used in the visualization.<br>
	 * This member is used to support visualization on those codes that do not utilize
	 * the generic pipeline. In the pipeline, use {@link #instanceParser} instead.
	 */
	protected Map<Integer, Label> labels;
	
	public LinearCRFViewer(Map<Integer, Label> labels){
		super(null);
		this.labels = labels;
	}
	
	public LinearCRFViewer(InstanceParser instanceParser) {
		super(instanceParser);
	}
	
	@SuppressWarnings("unchecked")
	protected void initData()
	{
		this.instance = (LinearInstance<Label>)super.instance;
		this.inputs = (ArrayList<String[]>)super.inputs;
		this.outputs = (ArrayList<Label>)super.outputs;
		//WIDTH = instance.Length * span_width;
	}
	
	@Override
	protected String label_mapping(VNode node) {
		int[] ids = node.ids;
//		int size = instance.size();
		int pos = ids[0]-1; // position
		int nodeId = ids[1];
		int nodeType = ids[4];
		if(nodeType == NODE_TYPES.LEAF.ordinal()){
			return "LEAF";
		} else if (nodeType == NODE_TYPES.ROOT.ordinal()){
			return "ROOT";
		}
//		ids[1]; // tag_id
//		ids[4]; // node type
		if(getLabel(nodeId).getForm().equals("O")){
			return inputs.get(pos)[0];
		}
		return getLabel(nodeId).toString();
	}
	
	protected void initNodeColor(VisualizeGraph vg)
	{
		if (colorMap != null){
			for(VNode node : vg.getNodes())
			{
				int[] ids = node.ids;
//				int pos = ids[0];
				int nodeId = ids[1]-0;
				int nodeType = ids[4];
				if(nodeType != NODE_TYPES.NODE.ordinal()){
					node.color = colorMap[0];
				} else {
					node.color = colorMap[1 + (nodeId % 8)];
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
			int pos = ids[0]-1;
			int labelId = ids[1];
			int nodeType = ids[4];
			
			double x = pos * span_width;
			int mappedId = labelId;
			switch(mappedId){
			case 0:
				mappedId = 1; break;
			case 1:
				mappedId = 6; break;
			case 6:
				mappedId = 0; break;
			}
			double y = mappedId * span_height + offset_height;
			if(nodeType == NODE_TYPES.ROOT.ordinal()){
				x = (pos + 1) * span_width;
				y = 3 * span_height + offset_height;
			}
			
			node.point = new Point2D.Double(x,y);
			layout.setLocation(node, node.point);
			layout.lock(node, true);
		}
	}
	
	private Label getLabel(int labelId){
		if(this.instanceParser != null){
			return ((DelimiterBasedInstanceParser)this.instanceParser).getLabel(labelId);
		} else {
			return this.labels.get(labelId);
		}
	}

}
