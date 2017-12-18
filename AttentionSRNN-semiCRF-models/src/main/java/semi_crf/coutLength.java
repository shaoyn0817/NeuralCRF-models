package semi_crf;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;

public class coutLength {
public static void main(String []args) throws Exception{
	File file = new File("experiment/TrainData.txt");
	FileInputStream fin = new FileInputStream(file);
	Long l = file.length();
	byte[] read = new byte[l.intValue()];
	fin.read(read);
	String ss = new String(read);
	String text[] = ss.split("\r\n\r\n");
	int maxSentenceLen = 0;
	int maxLabelLen = 0;
	HashMap<String, Integer> dict = new HashMap<String, Integer>();
	for(int i = 0; i < text.length; i++){
		String[] piece = text[i].split("\r\n");
		if(piece.length > maxSentenceLen)
			maxSentenceLen = piece.length;
		int number = 0;
		String label = "";
		for(int j = 0; j < piece.length; j++){
			String word[] = piece[j].split(" ");
			String ll = word[word.length - 1];
			if(!ll.equals(label))
			{
				if(dict.containsKey(label)){
					if(label.equals("title"))
					    System.out.println(number);
					if(label.equals("author") && number==0)
						System.out.println("dasd"+piece[j].length());
					Integer num = dict.get(label);
					if(number > num){
						dict.put(label, number);
					}
					label = ll;
					number = 1;
				} else {
					if(label.equals("title"))
					System.out.println(number);
					if(label.equals("author") && number==0)
						System.out.println("dasd"+piece[j].length());
					dict.put(label, number);
					label = ll;
					number = 1;
				}
			} else number++;
		}
	}
	fin.close();
	Iterator<String> it = dict.keySet().iterator();
	while(it.hasNext()){
		String word = it.next();
	    int number = dict.get(word);
	    System.out.println(word+": "+ number);
	}
}
}
