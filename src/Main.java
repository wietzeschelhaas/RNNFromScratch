import java.io.File;  // Import the File class
import java.io.FileNotFoundException;  // Import this class to handle errors
import java.util.ArrayList;
import java.util.Scanner; // Import the Scanner class to read text files


public class Main {


    static CharTokenizer charTokenizer;
    public static void main(String[] args){

        int seqLength = 25;
        int inputSize = 0;

        String fullData = "";
        charTokenizer = new CharTokenizer();


        try {
            File myObj = new File("C:/Users/wietz/Desktop/textGen/alice.txt");
            Scanner myReader = new Scanner(myObj);
            while (myReader.hasNextLine()) {
                String data = myReader.nextLine();
                charTokenizer.Tokenize(data);
                fullData += data;
            }
            myReader.close();
        } catch (FileNotFoundException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }

        ArrayList<Matrix> sequence = new ArrayList<Matrix>();
        //get first seq of chars
        for (int i = 0; i < seqLength; i++) {
            Matrix input = new Matrix(charTokenizer.getSize(),1);
            input.matrix[charTokenizer.getIndex(fullData.charAt(i))][0] = 1;

            sequence.add(input);

        }

        charTokenizer.p();


        RNN rnn = new RNN(charTokenizer.getSize(),50,charTokenizer.getSize(),seqLength);

        Matrix out = rnn.feedForward(sequence);

//        float sum = 0;
//        for (int i = 0; i < out.rows; i++) {
//            sum += out.matrix[i][0];
//        }
//
//        System.out.println(sum);

        System.out.println(out);

    }


}
