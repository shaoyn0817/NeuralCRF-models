package semi_crf;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;

public class shortenTestSet {
    public static void main(String args[]) throws Exception{
    	File file = new File("experiment/TestData.txt");
    	FileInputStream fin = new FileInputStream(file);
    	Long l = file.length();
    	byte[] read = new byte[l.intValue()];
    	fin.read(read);
    	String ss = new String(read);
    	String text[] = ss.split("\r\n\r\n");
    	int number = 0;
    	int delete = 0;
    	FileOutputStream fout = new FileOutputStream(new File("experiment/ShortTestData.txt"));
    	for(int i = 0; i < text.length; i++){
    		String[] piece = text[i].split("\r\n");
    		if(piece.length < 25){ //限定训练集中单句训练样例的最长长度 为 25
    		    fout.write((text[i].trim()+"\r\n\r\n").getBytes());
    		    number++;
    		} else {
				delete++;
			}
    	}
    	System.out.println("删除了"+delete);
    	System.out.println("还剩余"+number);
    	fin.close();
    	fout.close();
    }
}
