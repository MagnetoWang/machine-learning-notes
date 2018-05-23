package Media;
import java.util.*;
public class LZW {

	/** 
	*
	* @ClassName : LZW.java
	* @author : Magneto_Wang
	* @date  2018年5月23日 下午6:37:05
	* @Description  TODO
	* 
	*/
	
	 /** Compress a string to a list of output symbols. */
    public static List<Integer> compress(String uncompressed) {
        // 原字符的大小位数 8 位
        int dictSize = 256;
        Map<String,Integer> dictionary = new HashMap<String,Integer>();
        for (int i = 0; i < 256; i++)
            dictionary.put("" + (char)i, i);
 
        String w = "";
        List<Integer> result = new ArrayList<Integer>();
        System.out.println("===============================");
        //制作字典
        System.out.println("形成字典");
        for (char c : uncompressed.toCharArray()) {
            String wc = w + c;
            if (dictionary.containsKey(wc)){
            	//制造新的字符串
                w = wc;
            }
            else {
                result.add(dictionary.get(w));

                // 增加新的字典字符串
                dictionary.put(wc, dictSize++);
                w = "" + c;
            }
            System.out.print(w+" : ");
            System.out.print(dictionary.get(w)+"  ");
            System.out.print(dictSize+"  ");
//            System.out.println();
        }
 
        // 输出w
        
        if (!w.equals("")){
        	System.out.print(w);
        	result.add(dictionary.get(w));
        }
        System.out.println();
        System.out.println("===============================");
        return result;
    }
 
    /** 
     * 
     * Decompress a list of output ks to a string. */
    public static String decompress(List<Integer> compressed) {
        // Build the dictionary.
        int dictSize = 256;
        Map<Integer,String> dictionary = new HashMap<Integer,String>();
        for (int i = 0; i < 256; i++)
            dictionary.put(i, "" + (char)i);

        System.out.println("===============================");
        System.out.println("解压过程");
        String w = "" + (char)(int)compressed.remove(0);
        StringBuffer result = new StringBuffer(w);
        for (int k : compressed) {
            String entry;
            if (dictionary.containsKey(k)){
            	entry = dictionary.get(k);
            }
                
            else if (k == dictSize){
            	entry = w + w.charAt(0);
            }
            else{
            	throw new IllegalArgumentException("Bad compressed k: " + k);
            }
                
 
            System.out.print(entry+" ");
            result.append(entry);
 
            // Add w+entry[0] to the dictionary.
            dictionary.put(dictSize++, w + entry.charAt(0));
 
            w = entry;
        }
        System.out.println();
        System.out.println("===============================");
        return result.toString();
    }
 
    public static void main(String[] args) {
    	Scanner in = new Scanner(System.in);
    	String line=in.nextLine();
    	//"TOBEORNOTTOBEORTOBEORNOT"
    	//TOTOBTOTOTOTOTOTOTOTOEORNOTTOBEORTOBEORNOT
        List<Integer> compressed = compress("TOTOBTOTOTOTOTOTOTOTOEORNOTTOBEORTOBEORNOT");
//        for(Integer e:compressed){
//        	System.out.print(e+" ");
//        }
//        System.out.println();
        System.out.println(" zip : "+compressed);
        String decompressed = decompress(compressed);
        System.out.println(" input line  : "+line);
        System.out.println(" output line : "+decompressed);
    }
}
