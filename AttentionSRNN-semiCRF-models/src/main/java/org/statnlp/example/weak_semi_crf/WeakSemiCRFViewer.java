package org.statnlp.example.weak_semi_crf;

import java.awt.geom.Point2D;
import java.util.ArrayList;
import java.util.List;

import org.statnlp.commons.types.LinearInstance;
import org.statnlp.commons.types.Span;
import org.statnlp.example.weak_semi_crf.WeakSemiCRFNetworkCompiler.NodeType;
import org.statnlp.ui.visualize.type.VNode;
import org.statnlp.ui.visualize.type.VisualizationViewerEngine;
import org.statnlp.ui.visualize.type.VisualizeGraph;
import org.statnlp.util.instance_parser.InstanceParser;



public class WeakSemiCRFViewer extends VisualizationViewerEngine {
	
	static double span_width = 100;

	static double span_height = 100;
	
	static double offset_width = 100;
	
	static double offset_height = 100;
	
	protected LinearInstance<Span> instance;
	
	protected String[] inputs;
	
	protected ArrayList<Span> outputs;
	
	public WeakSemiCRFViewer(InstanceParser instanceParser) {
		super(instanceParser);
	}
	
	@SuppressWarnings("unchecked")
	protected void initData()
	{
		this.instance = (LinearInstance<Span>)super.instance;
		this.inputs = new String[this.instance.input.size()];
		for(int i=0; i<this.inputs.length; i++){
			this.inputs[i] = this.instance.input.get(i)[0];
		}
		this.outputs = (ArrayList<Span>)super.outputs;
		//WIDTH = instance.Length * span_width;
	}
	
	@Override
	protected String label_mapping(VNode node) {
		int[] ids = node.ids;
//		int size = instance.size();
//		int pos = ids[0]; // position
		int nodeId = ids[2];
		int nodeType = ids[1];
		if(nodeType == NodeType.LEAF.ordinal()){
			return "LEAF";
		} else if (nodeType == NodeType.ROOT.ordinal()){
			return "ROOT";
		}
//		ids[1]; // tag_id
//		ids[4]; // node type
//		if(Label.get(nodeId).form.equals("O")){
//			return inputs[pos];
//		}
		return ((WeakSemiCRFInstanceParser)instanceParser).getLabel(nodeId).toString();
	}
	
	protected void initNodeColor(VisualizeGraph vg)
	{
		if (colorMap != null){
			for(VNode node : vg.getNodes())
			{
				int[] ids = node.ids;
//				int pos = ids[0];
//				int nodeId = ids[2];
				int nodeType = ids[1];
				if(nodeType == NodeType.LEAF.ordinal() || nodeType == NodeType.ROOT.ordinal()){
					node.color = colorMap[0];
				} else if(nodeType == NodeType.BEGIN.ordinal()){
					node.color = colorMap[1];
				} else if(nodeType == NodeType.END.ordinal()){
					node.color = colorMap[2];
				}
			}
		}
		
	}
	
	protected void initNodeCoordinate(VisualizeGraph vg)
	{
		List<VNode> newNodes = new ArrayList<VNode>();
		for(VNode node : vg.getNodes())
		{
			int[] ids = node.ids;
//			int size = this.inputs.length;
			int pos = ids[0];
			int labelId = ids[2];
			int nodeType = ids[1];
			if(nodeType == NodeType.BEGIN.ordinal()){
				VNode newNode = new VNode(node.id*10+1, node.index*10+1);
				newNode.ids = new int[]{pos, -1, -1};
				newNodes.add(newNode);
			}
			
			double x = pos * span_width * 2;
			if(nodeType == NodeType.END.ordinal()){
				x += 0.5*span_width;
			}
			int mappedId = labelId;
//			switch(mappedId){
//			case 0:
//				mappedId = 1; break;
//			case 1:
//				mappedId = 6; break;
//			case 6:
//				mappedId = 0; break;
//			}
			double y = mappedId * span_height + offset_height;
			if(nodeType == NodeType.ROOT.ordinal()){
				x = (pos + 1) * span_width * 2;
				y = 3 * span_height + offset_height;
			}
			
			node.point = new Point2D.Double(x,y);
			layout.setLocation(node, node.point);
			layout.lock(node, true);
		}
		
		for(VNode node: newNodes){
			vg.addNode(node);
			int pos = node.ids[0];
			double x = pos * span_width * 2;
			double y = 2 * span_height + offset_height;
			node.point = new Point2D.Double(x,y);
			layout.setLocation(node, node.point);
			layout.lock(node, true);
		}
		
	}

}
