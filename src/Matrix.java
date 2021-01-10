import java.util.Iterator;

public class Matrix {
    float[][] matrix;
    int rows;
    int columns;

    public Matrix(int rows,int columns)

    {
        this.rows = rows;
        this.columns = columns;

        matrix = new float[rows][columns];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                matrix[i][j] = 0;
            }
        }


    }

    public float[] getRow(int index){
        return this.matrix[index];
    }
    public float[] getColumn (int index){
        float col[] = new float[this.rows];
        for (int i = 0; i < this.rows; i++) {
            col[i] = this.matrix[i][index];
        }
        return col;
    }

    static Matrix transpose(Matrix a){
        Matrix result = new Matrix(a.columns,a.rows);
        float[][] trans = new float[a.columns][a.rows];
        for (int i = 0; i < a.columns; i++) {
            trans[i] = a.getColumn(i);
        }
        result.matrix = trans;
        return result;
    }

    public float sum(){
        float result = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                result += this.matrix[i][j];
            }
        }
        return result;
    }
    public void randomize(float min, float max){
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                this.matrix[i][j] = ((float) Math.random()*(max-min)) + min;
            }
        }
    }

    static Matrix fromArray(float[] arr){
        Matrix m = new Matrix(arr.length,1);
        for (int i = 0; i < arr.length; i++) {
            m.matrix[i][0] = arr[i];
        }
        return m;
    }


    public Float[] toArray(){
        Float[] arr = new Float[this.rows];
        for (int i = 0; i < this.rows; i++) {
            //TODO test this function, not sure if this is right
            arr[i] = this.matrix[i][0];
        }
        return arr;
    }


    // scale matrix
    public void scl(float n){
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                this.matrix[i][j] *= n;
            }
        }
    }
    public void add(float n){
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                this.matrix[i][j] += n;
            }
        }
    }

    public void sub(Matrix m){
        if(m.rows != this.rows || m.columns != this.columns){
            System.out.println("incompatible dimensions");
            return;
        }

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                matrix[i][j] -= m.matrix[i][j];
            }
        }
    }

    static Matrix sub(Matrix a,Matrix b){
        if(a.rows != b.rows || a.columns != b.columns){
            System.out.println("incompatible dimensions");
            return null;
        }
        Matrix result = new Matrix(a.rows,a.columns);
        for (int i = 0; i < a.rows; i++) {
            for (int j = 0; j < a.columns; j++) {
                result.matrix[i][j] = a.matrix[i][j] - b.matrix[i][j];
            }
        }
        return result;
    }

    public void add(Matrix m){
        if(m.rows != this.rows || m.columns != this.columns){
            System.out.println("incompatible dimensions");
            return;
        }

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                matrix[i][j] += m.matrix[i][j];
            }
        }
    }
    public static Matrix add(Matrix a, Matrix b){
        if(a.rows != b.rows || a.columns != b.columns){
            System.out.println("incompatible dimensions");
            return null;
        }
        Matrix result = new Matrix(a.rows,a.columns);
        for (int i = 0; i < a.rows; i++) {
            for (int j = 0; j < a.columns; j++) {
                result.matrix[i][j] = a.matrix[i][j] + b.matrix[i][j];
            }
        }
        return result;
    }
    static Matrix mult(Matrix a, Matrix b){
        if(a.columns != b.rows){
            System.out.println("incompatible dimensions");
            return null;
        }
        Matrix multed = new Matrix(a.rows,b.columns);
        for (int i = 0; i < multed.rows; i++) {
            for (int j = 0; j < multed.columns; j++) {
                multed.matrix[i][j] = dot(a.getRow(i),b.getColumn(j));
            }
        }
        return multed;
    }

    void hadamardMult(Matrix b){
        if(b.rows != this.rows || b.columns != this.columns){
            System.out.println("incompatible dimensions at hadamardmult");
            return;
        }
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                matrix[i][j] *= b.matrix[i][j];
            }
        }
    }
    static Matrix hadamardMult(Matrix a,Matrix b){
        if(b.rows != a.rows || b.columns != a.columns){
            System.out.println("incompatible dimensions at hadamardmult");
            return null;
        }
        Matrix multed = new Matrix(a.rows,b.columns);
        for (int i = 0; i < a.rows; i++) {
            for (int j = 0; j < a.columns; j++) {
                multed.matrix[i][j] = a.matrix[i][j] * b.matrix[i][j];
            }
        }
        return multed;
    }

     public void div(Matrix m){
        if(m.rows != this.rows || m.columns != this.columns){
            System.out.println("incompatible dimensions at div");
            return;
        }

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                float a = matrix[i][j];
                matrix[i][j] = a/m.matrix[i][j];
            }
        }
    }

    public void sqrt(){
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                double a = matrix[i][j];
                matrix[i][j] = (float)Math.sqrt(a);
            }
        }
    }

    //returns the dot product between vectors a and b represented as an array of floats
    private static float dot(float[] a,float[] b) {
        float d = 0;
        for (int i = 0; i < a.length; i++) {
            d += a[i] * b[i];
        }
        return d;
    }

    @Override
    public String toString() {
        String res = "";
        for (int i = 0; i < rows; i++) {
            String r = "";
            for (int j = 0; j < columns; j++) {
                r += " "+this.matrix[i][j];
            }
            res += r + " \n";
        }
        return res;
    }
}
