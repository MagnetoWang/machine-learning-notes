package Media;


import java.util.ArrayList;
import java.util.HashMap;
import java.util.Scanner;

public class LZ78 {
	public static void main(String[] args) {
		Scanner in =new Scanner(System.in);
		int[] term=new int [2];
		//SDFSDFSDFSDSAASDFSDF
//		BABAABRRRA
		//asdklfjlskdfjlskadfjlsadkjflksadjflkasadf
		
		String input=in.nextLine();//"ABBCBCABABCAABCAAB";//in.nextLine();
		ArrayList<int[]> records=lz78(input);
		System.out.println("------------解压--------------");
		String result=unlz78(records);
		System.out.println(" 输入 = "+input);
		System.out.println(" 结果 = "+result);
		in.close();
	}
	public static ArrayList<int[]> lz78(String input){
		ArrayList<int[]> records=new ArrayList<>();
		int[] term=new int [2];
		HashMap<String,Integer> map=new HashMap<>();
		int group=0;
		for(int i=0;i<input.length();i++){
			term=new int [2];
			System.out.println("  input.charAt(i) = "+input.charAt(i));
			if(map.containsKey(input.substring(i, i+1))==false){
				group++;
				map.put(input.substring(i, i+1),group);
				
				term[0]=0;
				term[1]=input.charAt(i)-'A';
				
			}else{
				for(int j=i+1;j<input.length()+1;j++){
					
					if(map.containsKey(input.substring(i, j))==false){
						group++;
						map.put(input.substring(i, j),group);
						
						term[1]=input.charAt(j-1)-'A';
						i=j-1;
//						term[0]++;
						break;
				
					}else{
						
						term[0]=map.get(input.substring(i, j));
					}
				}
				
			}
			records.add(term);
			
			System.out.println("term[0] = "+term[0]+" term[1] = "+(char)(term[1]+'A'));
		}
		
		return records;
	}
	public static String unlz78(ArrayList<int[]> records){
		StringBuilder result =new StringBuilder();
		int [] term; 
		HashMap<Integer,String> map=new HashMap<>();
		int group=0;
		for(int i=0;i<records.size();i++){
			term=records.get(i);
			if(term[0]==0){
				group++;
				// String tString=new String(term[1]+'A');
				
				result.append((char)(term[1]+'A'));
				//长度 和 i 要分清楚。不要想当然直接写
				map.put(group,result.substring(result.length()-1,result.length()) );
				System.out.println(map.get(group));
			}else{
//				if(i==records.size()-1){
//					result.append(map.get(term[0]));
//					break;
//				}
				result.append(map.get(term[0]));
				result.append((char)(term[1]+'A'));
				group++;
				map.put(group,result.substring(result.length()-map.get(term[0]).length(),result.length()) );
				System.out.println(map.get(group));
			}
			System.out.println(" result = "+result);
			
		}
		
		
		return result.toString();
	}

}
