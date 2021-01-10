import java.util.Hashtable;

public class CharTokenizer {
    private Hashtable<Integer,Character> indexToChar = new Hashtable<Integer,Character>();
    private Hashtable<Character,Integer> charToIndex = new Hashtable<Character,Integer>();

    private int count = 0;

    public CharTokenizer(){

    }

    public void Tokenize(String str){
        for (int i = 0; i < str.length(); i++){
            char c = str.charAt(i);
            if(!indexToChar.contains(c)){
                indexToChar.put(count, c);
                charToIndex.put(c,count);
                count = count +1;

            }
        }

    }

    //TODO remove this function when debug is not needed anymore
    public void p(){
        System.out.println(indexToChar);
        System.out.println(charToIndex);
    }

    public int getIndex(char c){
        return charToIndex.get(c);
    }

    public char getChar(int i){
        return indexToChar.get(i);
    }

    public int getSize(){
        return count+1;
    }
}
