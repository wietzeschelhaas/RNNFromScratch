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
    //this is used because otherwise we haev div by 0
    float adaGradEpsilon = 0.000000001f;


    int sequenceLength;

    Matrix prevHiddenState;

    float totalLoss = 0;





    public RNN(int numInput, int numHidden, int numOutput, int sequenceLength) {

        weightsInputHidden = new Matrix(numHidden, numInput);
        weightsHiddenHidden = new Matrix(numHidden, numHidden);
        weightsHiddenOutput = new Matrix(numOutput, numHidden);

        biasHidden = new Matrix(numHidden, 1);
        biasOutput = new Matrix(numOutput, 1);

        weightsInputHidden.scl(0.01f);
        weightsHiddenHidden.scl(0.01f);
        weightsHiddenOutput.scl(0.01f);

        biasHidden.scl(0.01f);
        biasOutput.scl(0.01f);

        weightsInputHidden.randomize(-1, 1);
        weightsHiddenHidden.randomize(-1, 1);
        weightsHiddenOutput.randomize(-1, 1);

        biasHidden.randomize(-1, 1);
        biasOutput.randomize(-1, 1);

        //previous h state, this is the memory of the rnn
        // this is initialized with all zeros.
        prevHiddenState = new Matrix(numHidden, 1);

        this.sequenceLength = sequenceLength;



    }
    float tanhDerivation(float tanh){
        return 1 - tanh * tanh;
    }

    float softmax(float x, Matrix o) {
        float exp = (float) Math.exp(x);
        float sum = 0;
        for (int i = 0; i < o.rows; i++) {
            float expS = (float) Math.exp(o.matrix[i][0]);
            sum += expS;


        }
        return exp / sum;
    }

    //p is prediction, q is actual one hot vector
    float crossEntropy(Matrix p, Matrix q) {
        float sum = 0;
        for (int i = 0; i < p.rows; i++) {
            double d = p.matrix[i][0];
            sum += q.matrix[i][0] * Math.log(d);

        }
        return -sum;
    }

    float distance(Matrix p, Matrix q){
        float sum = 0;
        for (int i = 0; i < p.rows; i++) {
            double d = p.matrix[i][0];
            double l = q.matrix[i][0];
            sum += Math.sqrt(Math.pow(d,2) + Math.pow(l,2));
        }

        return sum;
    }

    //inputs is a matrix with size (numInput,1)
    public Matrix feedForward(ArrayList<Matrix> input) {
        Matrix softVal = null;
        for (int i = 0; i < input.size(); i++) {


            //h_t = tanh(W_hh * h_t-1 + W_xh * x_t +bias)

            //results is a (numhidden,1) size matrix
            Matrix inputOut = Matrix.mult(this.weightsInputHidden, input.get(i));

            //results is a (numhidden,1) size matrix
            Matrix hiddenOutput = Matrix.mult(this.weightsHiddenHidden, prevHiddenState);


            //add them together
            Matrix add = Matrix.add(inputOut, hiddenOutput);
            //add bias
            add.add(biasHidden);

            //tanh activation
            for (int j = 0; j < add.rows; j++) {
                float tanh = (float) Math.tanh(add.matrix[j][0]);
                add.matrix[j][0] = tanh;
            }

            //make a copy of current hidden state and store it in prevHidden state.
            prevHiddenState = new Matrix(prevHiddenState.rows, prevHiddenState.columns);
            for (int j = 0; j < add.rows; j++) {
                prevHiddenState.matrix[i][0] = add.matrix[i][0];
            }


            //y_t = W_hy * ht + bias
            Matrix output = Matrix.mult(this.weightsHiddenOutput, add);
            output.add(biasOutput);

            softVal = new Matrix(output.rows, output.columns);

            for (int j = 0; j < output.rows; j++) {
                softVal.matrix[j][0] = softmax(output.matrix[j][0], output);
            }


        }
        return softVal;
    }

    //backprop through time
    //this will feedforward a sequence once, calculate loss at every timestep and then update weights by dL/dw

    //for derviation of the formulas look here:
    //https://medium.com/learn-love-ai/step-by-step-walkthrough-of-rnn-training-part-ii-7141084d274b

    //for pseudo/pytohn code look here
    //http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/
    //note that the code in the above article, for every output looks back a number of hidden states, where here only one
    public void train(ArrayList<Matrix> input, Matrix actual) {

        totalLoss = 0;


        ArrayList<Matrix> prevYval = new ArrayList<Matrix>();
        ArrayList<Matrix> prevHiddenStates = new ArrayList<Matrix>();

        Matrix softVal = null;

        //forward step
        for (int i = 0; i < input.size(); i++) {


            //h_t = tanh(W_hh * h_t-1 + W_xh * x_t +bias)

            //results is a (numhidden,1) size matrix
            Matrix inputOut = Matrix.mult(this.weightsInputHidden, input.get(i));

            //results is a (numhidden,1) size matrix
            Matrix hiddenOutput = Matrix.mult(this.weightsHiddenHidden, prevHiddenState);


            //add them together
            Matrix add = Matrix.add(inputOut, hiddenOutput);
            //add bias
            add.add(biasHidden);

            //tanh activation
            for (int j = 0; j < add.rows; j++) {
                float tanh = (float) Math.tanh(add.matrix[j][0]);
                add.matrix[j][0] = tanh;

            }

            //make a copy of current hidden state and store it in prevHidden state.
            prevHiddenState = new Matrix(prevHiddenState.rows, prevHiddenState.columns);
            for (int j = 0; j < add.rows; j++) {
                prevHiddenState.matrix[j][0] = add.matrix[j][0];
            }

            // this is used later when backpropping
            prevHiddenStates.add(prevHiddenState);


            //y_t = W_hy * ht + bias
            Matrix output = Matrix.mult(this.weightsHiddenOutput, add);
            output.add(biasOutput);

            softVal = new Matrix(output.rows, output.columns);

            for (int j = 0; j < output.rows; j++) {
                softVal.matrix[j][0] = softmax(output.matrix[j][0], output);
            }
            //this is also used later when backpropping
            prevYval.add(softVal);

            Matrix actualOnehot = new Matrix(softVal.rows,1);
            actualOnehot.matrix[(int)actual.matrix[i][0]][0] = 1;

            float loss = crossEntropy(softVal, actualOnehot);
            totalLoss += loss;



        }

        //these will hold the gradients
        Matrix deltaWeightsInputHidden = new Matrix(weightsInputHidden.rows, weightsInputHidden.columns);
        Matrix deltaWeightsHiddenHidden = new Matrix(weightsHiddenHidden.rows, weightsHiddenHidden.columns);
        Matrix deltaWeightsHiddenOutput = new Matrix(weightsHiddenOutput.rows, weightsHiddenOutput.columns);

        Matrix deltaBiasHidden = new Matrix(biasHidden.rows, biasHidden.columns);
        Matrix deltaBiasOutput = new Matrix(biasOutput.rows, biasOutput.columns);

        //used for adagrad later
        Matrix previousDeltaWeightsInputHidden = new Matrix(weightsInputHidden.rows, weightsInputHidden.columns);
        Matrix previousDeltaWeightsHiddenHidden = new Matrix(weightsHiddenHidden.rows, weightsHiddenHidden.columns);
        Matrix previousDeltaWeightsHiddenOutput = new Matrix(weightsHiddenOutput.rows, weightsHiddenOutput.columns);

        Matrix previousDeltaBiasHidden = new Matrix(biasHidden.rows, biasHidden.columns);
        Matrix previousDeltaBiasOutput = new Matrix(biasOutput.rows, biasOutput.columns);

        //backwards
        for (int t = input.size()-1; t>1 ; t--) {
            Matrix dLdY = prevYval.get(t);
            // delta y is just -=1 for the actual class when using softmax and cross entropy
            // see http://cs231n.github.io/neural-networks-case-study/#grad
            dLdY.matrix[(int)actual.matrix[t][0]][0] -=1;

            //delta output weights is simply derivate of softmax/crossentropy  * the hidden state produced at timestep t
            Matrix dwy_t = Matrix.mult(dLdY,Matrix.transpose(prevHiddenStates.get(t)));

            //add to both weight matrix and bias
            deltaWeightsHiddenOutput.add(dwy_t);
            deltaBiasOutput.add(dLdY);


            Matrix dLdh = new Matrix(weightsHiddenHidden.rows,1);
            for (int i = 0; i < prevHiddenStates.get(t).rows; i++) {
                dLdh.matrix[i][0] = tanhDerivation(prevHiddenStates.get(t).matrix[i][0]);
            }

            //dLdh = derivation of activation function * weightsHiddenOutput.T X dy
            //where * is hadamard mult mand X is matrix mult
            dLdh.hadamardMult(Matrix.mult(Matrix.transpose(weightsHiddenOutput),dLdY));

            deltaBiasHidden.add(dLdh);


            deltaWeightsHiddenHidden.add(Matrix.mult(dLdh,Matrix.transpose(prevHiddenStates.get(t-1))));
            deltaWeightsInputHidden.add(Matrix.mult(dLdh,Matrix.transpose(input.get(t))));


        }

        // do not update weights with constant lr,
        //instead use adagrad to determine best lr for weights??
        deltaWeightsInputHidden.scl(learningRate);
        deltaWeightsHiddenHidden.scl(learningRate);
        deltaWeightsHiddenOutput.scl(learningRate);

        deltaBiasHidden.scl(learningRate);
        deltaBiasOutput.scl(learningRate);

        //Adagrad
        //see this for formula: https://ruder.io/optimizing-gradient-descent/index.html#adagrad

        /*previousDeltaWeightsInputHidden.add(Matrix.hadamardMult(deltaWeightsInputHidden,deltaWeightsInputHidden));
        previousDeltaWeightsHiddenHidden.add(Matrix.hadamardMult(deltaWeightsHiddenHidden,deltaWeightsHiddenHidden));
        previousDeltaWeightsHiddenOutput.add(Matrix.hadamardMult(deltaWeightsHiddenOutput,deltaWeightsHiddenOutput));

        previousDeltaBiasHidden.add(Matrix.hadamardMult(biasHidden,biasHidden));
        previousDeltaBiasOutput.add(Matrix.hadamardMult(biasOutput,biasOutput));


        previousDeltaWeightsInputHidden.add(adaGradEpsilon);
        previousDeltaWeightsInputHidden.sqrt();
        deltaWeightsInputHidden.div(previousDeltaWeightsInputHidden);

        previousDeltaWeightsHiddenHidden.add(adaGradEpsilon);
        previousDeltaWeightsHiddenHidden.sqrt();
        deltaWeightsHiddenHidden.div(previousDeltaWeightsHiddenHidden);

        previousDeltaWeightsHiddenOutput.add(adaGradEpsilon);
        previousDeltaWeightsHiddenOutput.sqrt();
        deltaWeightsHiddenOutput.div(previousDeltaWeightsHiddenOutput);

        previousDeltaBiasHidden.add(adaGradEpsilon);
        previousDeltaBiasHidden.sqrt();
        deltaBiasHidden.div(previousDeltaBiasHidden);

        previousDeltaBiasOutput.add(adaGradEpsilon);
        previousDeltaBiasOutput.sqrt();
        deltaBiasOutput.div(previousDeltaBiasOutput);

        deltaWeightsInputHidden.scl(learningRate);
        deltaWeightsHiddenHidden.scl(learningRate);
        deltaWeightsHiddenOutput.scl(learningRate);

        deltaBiasHidden.scl(learningRate);
        deltaBiasOutput.scl(learningRate);*/



        this.weightsInputHidden.sub(deltaWeightsInputHidden);
        this.weightsHiddenHidden.sub(deltaWeightsHiddenHidden);
        this.weightsHiddenOutput.sub(deltaWeightsHiddenOutput);


        this.biasHidden.sub(deltaBiasHidden);
        this.biasOutput.sub(deltaBiasOutput);



    }
}
