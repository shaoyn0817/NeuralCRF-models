package org.statnlp.example.logstic_regression;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.statnlp.commons.io.RAWF;
import org.statnlp.commons.types.Sentence;
import org.statnlp.commons.types.WordToken;

/**
 * The reader used to read the data. Example data format:
 * 	Rodriguez and agent Scott Boras were expected to meet Monday with Rangers owner Tom Hicks .
 *	NNP CC NN NNP NNP VBD VBN TO VB NNP IN NNPS NN NNP NNP .
 *	0,0,PER|3,4,PER|2,2,PER|13,14,PER|12,12,PER|11,11,ORG
 *	Personal/Social 0,0 PER 3,4 PER|Employee/Membership/Subsidiary 13,14 PER 11,11 ORG
 *
 *  The first line represents words.
 *  The second line represents POS tags
 *  The third line represents the entities in the sentence
 *  The last line represent the relationship between entities in the form that "Relation1 entity1 entity2" split by "|"
 * @author allanjie
 *
 */
public class LRReader {

	public static LRInstance[] readInsts(String file, boolean isTrainingInsts) throws IOException {
		return readInsts(file, isTrainingInsts, -1);
	}
	
	/**
	 * This reader only read the mention span, not all the spans include O
	 * @param file
	 * @param number
	 * @return
	 * @throws IOException
	 */
	public static LRInstance[] readInsts(String file, boolean isTrainingInsts, int number) throws IOException {
		ArrayList<LRInstance> insts = new ArrayList<LRInstance>();
		String line = null;
		BufferedReader br = RAWF.reader(file);
		int instId = 1;
		int numSent = 0;
		int numRel = 0;
		int numEntityPair = 0;
		int numMentions = 0;
		while ((line = br.readLine())!= null ) {
			String[] words = line.split(" ");
			line = br.readLine(); // read the pos tag.
			String[] tags = line.split(" ");
			WordToken[] wts = new WordToken[words.length];
			for (int t = 0; t < wts.length; t++) {
				wts[t] = new WordToken(words[t], tags[t]);
			}
			Sentence sent = new Sentence(wts);
			String ners = br.readLine();
			List<Span> spans = new ArrayList<>();
			if (!ners.equals("")) {
				String[] spanInfos = ners.split("\\|");
				for (String spanInfo : spanInfos) {
					String[] spanArr = spanInfo.split(",");
					Span span = new Span(Integer.valueOf(spanArr[0]), Integer.valueOf(spanArr[1]), spanArr[2]);
					if (!spans.contains(span)) {
						spans.add(span);
					}
				}
				Collections.sort(spans);
			}
			numSent++;
			numMentions += spans.size();
			String allRelations = br.readLine();
			int[][] filled = new int[spans.size()][spans.size()];
			if (spans.size() == 0) {
				numSent--;
			}
			if (allRelations.equals("")) {
				
			} else {
				String[] vals = allRelations.split("\\|");
				for (String oneRelation : vals) {
					numRel++;
					String[] indices = oneRelation.split(" ");
					String relType = indices[0];
					String arg1Entity = indices[2];
					String arg2Entity = indices[4];
					String[] firstIndices = indices[1].split(",");
					String[] secondIndices = indices[3].split(",");
					int start1 = Integer.valueOf(firstIndices[0]);
					int end1 = Integer.valueOf(firstIndices[1]);
					int start2 = Integer.valueOf(secondIndices[0]);
					int end2 = Integer.valueOf(secondIndices[1]);
					Span span1 = new Span(start1, end1, arg1Entity);
					Span span2 = new Span(start2, end2, arg2Entity);
					int span1Idx = spans.indexOf(span1);
					int span2Idx = spans.indexOf(span2);
					int leftIdx = span1Idx < span2Idx ? span1Idx : span2Idx;
					int rightIdx = span1Idx < span2Idx ? span2Idx : span1Idx;
					filled[leftIdx][rightIdx] = 1;
					LGInput input = new LGInput(sent, spans, leftIdx, rightIdx);
					//define the direction.
					/**means the reversed direction for the relation**/
					String direction = span1Idx < span2Idx ? "" : Config.REV_SUFF;
					relType = relType + direction;
					RelationType relationType = RelationType.get(relType);
					LRInstance inst = new LRInstance(instId, 1.0, input, relationType);
					if (isTrainingInsts)
						inst.setLabeled();
					else inst.setUnlabeled();
					insts.add(inst);
					instId++;
					if (span1Idx < 0 || span2Idx < 0)
						throw new RuntimeException("smaller than 0?");
				}
			}
			for (int i = 0; i < spans.size(); i++) {
				for (int j = i + 1; j < spans.size(); j++) {
					numEntityPair++;
					if (filled[i][j] == 0) {
						LGInput input = new LGInput(sent, spans, i, j);
						RelationType nr  = RelationType.get(Config.NR);
						LRInstance inst = new LRInstance(instId, 1.0, input, nr);
						if (isTrainingInsts) {
							inst.setLabeled();
						} else {
							inst.setUnlabeled();
						}
						insts.add(inst);
						instId++;
					}
				}
			}
			if (number != -1 && insts.size() > number) {
				break;
			}
			spans = new ArrayList<>();
			line = br.readLine(); //empty line
		}
		br.close();
		RelationType.get(Config.NR);
		System.out.println("number of sents with entities: " + numSent);
		System.out.println("number of relations:" + numRel);
		System.out.println("number of mention pairs:" + numEntityPair);
		System.out.println("number of mentions:" + numMentions);
		return insts.toArray(new LRInstance[insts.size()]);
	}
	

}
