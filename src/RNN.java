import java.util.ArrayList;

public class RNN {

    //weights from input to hidden layer
    Matrix weightsInputHidden;

    //weights from hidden to hidden layer, this is the recurrent part
    Matrix weightsHiddenHidden;

    //weights from hidden to output
    Matrix weightsHiddenOutput;

    Matrix biasHidden;
    Matrix biasOutput;

    float learningRate = 0.1f;

    int sequenceLength;

    Matrix prevHiddenState;

    ArrayList<Matrix> prevHiddenStates = new ArrayList<Matrix>();

    public RNN(int numInput, int numHidden, int numOutput,int sequenceLength){
        weightsInputHidden = new Matrix(numHidden, numInput);
        weightsHiddenHidden = new Matrix(numHidden, numHidden);
        weightsHiddenOutput = new Matrix(numOutput, numHidden);

        biasHidden = new Matrix(numHidden,1);
        biasOutput = new Matrix(numOutput,1);

        weightsInputHidden.randomize(-1,1);
        weightsHiddenHidden.randomize(-1,1);
        weightsHiddenOutput.randomize(-1,1);

        biasHidden.randomize(-1,1);
        biasOutput.randomize(-1,1);

        //previous h state, this is the memory of the rnn
        // this is initialized with all zeros.
        prevHiddenState = new Matrix(numHidden,1);

        this.sequenceLength = sequenceLength;


    }

    float softmax(float x, Matrix o){
        float exp = (float) Math.exp(x);

        float sum = 0;
        for (int i = 0; i < o.rows; i++) {
            float expS = (float) Math.exp(o.matrix[i][0]);
            sum += expS;
        }
        return exp/sum;
    }

    //p is prediction, q is actual one hot vector
    float crossEntropy(Matrix p, Matrix q){
        float sum = 0;
        for (int i = 0; i < p.rows; i++) {
            double d = q.matrix[i][0];
            sum += p.matrix[i][0] * Math.log(d) ;
        }

        return sum;
    }


    //inputs is a matrix with size (numInput,1)
    public Matrix feedForward(ArrayList<Matrix> input){
        Matrix softVal = null;
        for (int i = 0; i < input.size(); i++) {


            //h_t = tanh(W_hh * h_t-1 + W_xh * x_t +bias)

            //results is a (numhidden,1) size matrix
            Matrix inputOut =  Matrix.mult(this.weightsInputHidden,input.get(i));

            //results is a (numhidden,1) size matrix
            Matrix hiddenOutput = Matrix.mult(this.weightsHiddenHidden,prevHiddenState);


            //add them together
            Matrix add = Matrix.add(inputOut,hiddenOutput);
            //add bias
            add.add(biasHidden);

            //tanh activation
            for (int j = 0; j < add.rows; j++) {
                float tanh = (float) Math.tanh(add.matrix[j][0]);
                add.matrix[j][0] = tanh;
            }

            //make a copy of current hidden state and store it in prevHidden state.
            prevHiddenState = new Matrix(prevHiddenState.rows,prevHiddenState.columns);
            for (int j = 0; j < add.rows; j++) {
                prevHiddenState.matrix[i][0] = add.matrix[i][0];
            }



            //y_t = W_hy * ht + bias
            Matrix output = Matrix.mult(this.weightsHiddenOutput,add);
            output.add(biasOutput);

            softVal = new Matrix(output.rows,output.columns);

            for (int j = 0; j < output.rows; j++) {
                softVal.matrix[j][0] = softmax(output.matrix[j][0],output);
            }


        }
        return softVal;
    }

    //backprop through time
    //this will feedforward a sequence once, calculate loss at every timestep and then update weights by dL/dw
    public void train(ArrayList<Matrix> input, ArrayList<Matrix> actual){
        Matrix softVal = null;

        float totalLoss = 0;

        //forward step
        for (int i = 0; i < input.size(); i++) {


            //h_t = tanh(W_hh * h_t-1 + W_xh * x_t +bias)

            //results is a (numhidden,1) size matrix
            Matrix inputOut =  Matrix.mult(this.weightsInputHidden,input.get(i));

            //results is a (numhidden,1) size matrix
            Matrix hiddenOutput = Matrix.mult(this.weightsHiddenHidden,prevHiddenState);


            //add them together
            Matrix add = Matrix.add(inputOut,hiddenOutput);
            //add bias
            add.add(biasHidden);

            //tanh activation
            for (int j = 0; j < add.rows; j++) {
                float tanh = (float) Math.tanh(add.matrix[j][0]);
                add.matrix[j][0] = tanh;
            }

            //make a copy of current hidden state and store it in prevHidden state.
            prevHiddenState = new Matrix(prevHiddenState.rows,prevHiddenState.columns);
            for (int j = 0; j < add.rows; j++) {
                prevHiddenState.matrix[i][0] = add.matrix[i][0];
            }

            // this is used later when backpropping
            prevHiddenStates.add(prevHiddenState);


            //y_t = W_hy * ht + bias
            Matrix output = Matrix.mult(this.weightsHiddenOutput,add);
            output.add(biasOutput);

            softVal = new Matrix(output.rows,output.columns);

            for (int j = 0; j < output.rows; j++) {
                softVal.matrix[j][0] = softmax(output.matrix[j][0],output);
            }

            float loss = crossEntropy(softVal,actual.get(i));
            totalLoss += loss;

        }

        //these will hold the gradients
        Matrix deltaWeightsInputHidden = new Matrix(weightsInputHidden.rows,weightsInputHidden.columns);
        Matrix deltaWeightsHiddenHiddenn = new Matrix(weightsHiddenHidden.rows,weightsHiddenHidden.columns);
        Matrix deltaWeightsHiddenOutput = new Matrix(weightsHiddenOutput.rows,weightsHiddenOutput.columns);

        Matrix deltaBiasHidden = new Matrix(biasHidden.rows,biasHidden.columns);
        Matrix deltaBiasOutput = new Matrix(biasOutput.rows,biasOutput.columns);



    }
}
