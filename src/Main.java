import java.io.File;  // Import the File class
import java.io.FileNotFoundException;  // Import this class to handle errors
import java.util.ArrayList;
import java.util.Scanner; // Import the Scanner class to read text files


public class Main {


    static CharTokenizer charTokenizer;

    public static void main(String[] args) {

        int seqLength = 25;
        int numEpoch = 1;

        String fullData = "";
        charTokenizer = new CharTokenizer();


        try {
            File myObj = new File("C:/Users/ietze/Desktop/test.txt");
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
        Matrix target = new Matrix(seqLength, 1);

        RNN rnn = new RNN(charTokenizer.getSize(),40,charTokenizer.getSize(),seqLength);


        for (int i = 0; i < fullData.length()-1; i+=seqLength) {
            if(i%500 == 0){
                System.out.println(i);
                System.out.println(rnn.totalLoss);
            }


            for (int j = 0; j < seqLength; j++) {
                Matrix input = new Matrix(charTokenizer.getSize(), 1);
                input.matrix[charTokenizer.getIndex(fullData.charAt(i+j))][0] = 1;

                sequence.add(input);
                target.matrix[j][0] = charTokenizer.getIndex(fullData.charAt(i+j + 1));


            }
            rnn.train(sequence,target);
            sequence.clear();




        }



        //charTokenizer.p();


        //RNN rnn = new RNN(charTokenizer.getSize(),50,charTokenizer.getSize(),seqLength);

        //Matrix out = rnn.feedForward(sequence);

        //test to see if results sums to 1
//        float sum = 0;
//        for (int i = 0; i < out.rows; i++) {
//            sum += out.matrix[i][0];
//        }
//
//        System.out.println(sum);

        //rnn.train(sequence,target);

    }

}




