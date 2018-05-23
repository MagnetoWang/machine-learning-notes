package Media;
//http://www.cnblogs.com/en-heng/p/4992916.html
import java.util.ArrayList;
import java.util.Scanner;

public class LZSS {
	public static  int windows_size=6;
	public static int buffer_size=4;
	public static int offset=0;
	public static int length_match = 0;
	public static char nextChar; 
	public static void main(String[] args) {
		Scanner in =new Scanner(System.in);
//		aabbccddeeffgg
//		qwertyuiopasdfghjklzxcvbnm
//		asdklfjlskdfjlskadfjlsadkjflksadjflkasadf
//		lookahead
		String input=in.nextLine();//"aacaacabcabaaac";//in.nextLine();
		ArrayList<int[]> records=lzss(input);
		System.out.println("--------------解压过程-------------");
		String result = unlzss(records);
		System.out.println(" 结果 = "+result);
		System.out.println(" 输入 = "+input);
		in.close();
	}
	//return (p,l,c);
	//
	public static ArrayList<int[]> lzss(String input){
		int i=0;
		ArrayList<int[]> records=new ArrayList<>();
		int [] newRecords=new int[3];
//		newRecords[0]=0;
//		newRecords[1]=0;
//		newRecords[2]=input.charAt(0)-'a';
		//i为每次匹配完最大的字符串后的下一位字符
		while(i<input.length()){
			longest_match(input,i);
			newRecords=new int[3];
			newRecords[0]=offset;
			newRecords[1]=length_match;
			newRecords[2]=nextChar-'a';
			System.out.println("        offset = "+offset+" length_match = "+length_match+" nextChar = "+nextChar);
			System.out.println();
			records.add(newRecords);
			
			i+=(length_match+1);
		}
		return records;
		
	}
	public static void longest_match(String input,int index){
		int p=-1;
		int l=-1; 
		char c=' ';
		//选取需要查询的字符串的最后一个位置。可能是整个字符串的最后一位，也可能是index + 字符串窗口大小
		int end_buffer =Math.min(index+buffer_size, input.length());
		for(int j=index;j<end_buffer;j++){
			//选择一个字符串区域，作为查询字典
			//查询字典是不断的变化的
			int start_index =Math.max(0, index-windows_size);
			//开始要查询的字符串，也是不断更新
			String subStr = input.substring(index,j+1);
			System.out.println("subStr = "+subStr+" j = "+ j);
			//开始在查询字典中匹配
			for( int i = start_index;i<index;i++){
				//可能的重复次数 = 需要查询的字符串 / 查询字典
				//最后一位的位置 =需要查询的字符串 % 查询字典

				int repetition  = subStr.length()/(index-i+1);
				int last = subStr.length()%(index-i+1);
				System.out.println(" repetition = "+ repetition+" last = "+ last);

				String matchString ="";//= input.substring(i, index+1)*repetition+input.substring(i,i+last)
				
				//构造子字符串
				for(int over=0;over<repetition;over++){
					matchString=matchString +input.substring(i, index+1);
				}
				matchString+=input.substring(i,i+last);
				System.out.println(" matchString = "+ matchString+" subStr.length() = "+subStr.length());
				//构造之后比较
				//字符串不能用== 来比较。切记
//				if(matchString == subStr && subStr.length()>=1){
				if(matchString.equals(subStr) && subStr.length()>=1){

					if(j+1<end_buffer){
						p=index-i;
						l=subStr.length();
						c=input.charAt(j+1);
					}
					else{
						break;
					}
					System.out.println(" success"+" char = "+ c);
					
				}
				System.out.println(" p = "+p+" l = "+l+" c = "+ c);
				System.out.println();
				
			}
//			System.out.println(" p = "+p+" l = "+l);
			if(p==-1&&l==-1){
				offset=0;
				length_match=0;
				nextChar=input.charAt(index);
			}else{
				offset=p;
				length_match=l;
				nextChar=c;
				
			}
		}
	}
	public static String unlzss(ArrayList<int[]> records){
		StringBuilder result=new StringBuilder();
		int index=0;
		for(int i=0;i<records.size();i++){
			
			int[] dic=records.get(i);
			System.out.println("index = "+index+" dic[0] = "+dic[0]+" dic[1] = "+dic[1]+" dic[2] = "+((char)(dic[2]+'a')));
			
			if(dic[0]==0&&dic[1]==0){
				int c=dic[2]+'a';
				result.append((char)c);
			}else{
				if(dic[0]>=dic[1]){
					
					result.append(result.substring(index-dic[0]	, index-dic[0]+dic[1]));
					result.append((char)(dic[2]+'a'));
				}else{
					if(dic[0]<dic[1]){
						int repetition = dic[1]/dic[0];
						int last = dic[1]%dic[0];
						System.out.println(" repetition = "+ repetition+" last = "+ last);
						//还原的时候要注意下标容易复制错误的字符串
						for(int over=0;over<repetition;over++){
							result.append(result.substring(index-dic[0], index-dic[0]+dic[0]));
						}
						result.append(result.substring(index-dic[0],index-dic[0]+last));
						result.append((char)(dic[2]+'a'));
						

					}
				}
			}
			index+=(dic[1]+1);
			System.out.println(result);
		}
		



		return result.toString();
	}
	
}
