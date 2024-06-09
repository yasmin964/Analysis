#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip> // for setprecision

using namespace std;

// Define a class for Matrix
class Matrix {
private:
    vector<double> data;
    int size;

public:
    // Constructor
    Matrix(vector<double> input) : data(input), size(sqrt(input.size())) {}

    // Get the element at position (i, j)
    double getElement(int i, int j) const {
        return data[i * size + j];
    }

    // Set the value of an element at position (i, j)
    void setElement(int i, int j, double value) {
        data[i * size + j] = value;
    }

    // Get the size of the matrix (assuming it's square)
    int getSize() const {
        return size;
    }
};

// Define a class for Vector
class Vector {
private:
    vector<double> data;

public:
    // Constructor
    Vector(vector<double> input) : data(input) {}

    // Get the element at index i
    double getElement(int i) const {
        return data[i];
    }

    // Update the value of an element at index i
    void updateElement(int i, double value) {
        data[i] = value;
    }

    // Get the size of the vector
    int getSize() const {
        return data.size();
    }
};

// Define a class for IterativeMethod
class IterativeMethod {
protected:
    vector<vector<double>> alpha;
    vector<double> beta;
    double epsilon;

public:
    IterativeMethod(  vector<vector<double>> alpha, vector<double> beta, double epsilon)
            : alpha(alpha), beta(beta), epsilon(epsilon) {}

    virtual void solve() = 0;
};

class SeidelMethod : public IterativeMethod {
private:
    vector<Vector *> steps;
    vector<double> accuracies;
public:
    SeidelMethod(vector<vector<double>> alpha,  vector<double>beta,double epsilon): IterativeMethod(alpha, beta,epsilon) {
    }

    void decompose(const vector<vector<double>> &A, vector<vector<double>> &B, vector<vector<double>> &C,  const vector<double>& b) {

        int n = A.size();

        for (int i = 0; i < n; ++i) {
            beta[i] = b[i] / A[i][i];
            for (int j = 0; j < n; ++j) {
                if (i == j) {
                    B[i][j] = 0.0;
                    C[i][j] = 0.0;
                } else {
                    B[i][j] = -A[i][j] / A[i][i];

                    C[i][j] = -A[i][j] / A[i][i];

                }
            }
        }

        // Update alpha = B + C
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                alpha[i][j] = B[i][j];
            }
        }

    }

    vector<double> initializeX0(int n) {
        return vector<double>(n, 0.0);

    }

    vector<double> seidelIteration(const vector<vector<double>>& B,  vector<vector<double>>& C, const vector<double>& b, const vector<double>& x, const vector<vector<double>>& I_minus_B) {
        int n = B.size(); // Assuming B is square, so its size is the number of rows or columns
        // Compute (I - B)^(-1) using the inverse method
        vector<vector<double>> I_minus_B_inverse = inverse(I_minus_B);
        vector<double> result(n, 0.0);
        vector<double> current_x = x.empty() ? b : x;
        vector<vector<double>> I_minus_B_inv_C = matrixMatrixProduct(I_minus_B_inverse, C);
        vector<double> I_minus_B_inv_Cx = matrixVectorProduct(I_minus_B_inv_C, current_x);

        // Compute (I - B)^(-1) C x^{(k)}
        for (int i = 0; i < n; ++i) {
            result[i] += I_minus_B_inv_Cx[i];
        }

        // Add (I - B)^(-1) beta
        vector<double> I_minus_B_inv_beta = matrixVectorProduct(I_minus_B_inverse, b); // Assuming you have a function for matrix-vector multiplication
        for (int i = 0; i < n; ++i) {
            result[i] += I_minus_B_inv_beta[i];
        }
        return result;
    }



    double calculateAccuracy(const vector<double> &x_new, const vector<double> &x) {

        int n = x.size();
        double max_diff = 0.0;
        for (int i = 0; i < n; ++i) {
            double diff = (x_new[i] - x[i]) * (x_new[i] - x[i]);
            max_diff += diff;

        }
        max_diff = sqrt(max_diff);

        return max_diff;

    }


    void swapRows(vector<vector<double>> matrix, int row1, int row2) {

        int n = matrix.size();    // Access the size of the matrix
        for (int j = 0; j < n; ++j) {

            double temp = matrix[row1][j];

            matrix[row1][j]= matrix[row2] [j];

            matrix[row2] [j] =temp;

        }
    }

    vector<double> matrixVectorProduct(const vector<vector<double>> &A, const vector<double>  &v) {
        int n = A.size();

        vector<double> result(vector<double>(n, 0.0));


        for (int i = 0; i < n; ++i) {

            double sum = 0.0;

            for (int j = 0; j < n; ++j) {

                sum += A[i][j] * v[j];

            }
            result[i] =sum;

        }

        return result;

    }

    vector<vector<double>> matrixMatrixProduct(vector<vector<double>> &A, vector<vector<double>> &B) {
        int n = A.size(); // Assuming A and B are square matrices of the same size

        // Initialize result matrix
        vector<vector<double>> result(n, vector<double>(n, 0.0));

        // Compute each element of the result matrix
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                double sum = 0.0;
                for (int k = 0; k < n; ++k) {
                    sum += A[i][k] * B [k][j];
                }
                result[i][j] = sum;
            }
        }

        return result;
    }


    // Function to compute the inverse of a matrix using Gaussian elimination
    vector<vector<double>> inverse(vector<vector<double>> A) {

        int n = A.size();

        vector<vector<double>> identity (n, vector<double>(n, 0.0));


        // Initialize the identity matrix
        for (int i = 0; i < n; ++i) {

            identity[i][i] = 1.0;

        }

        double determinant = 1.0;


        // Gaussian elimination with partial pivoting
        for (int j = 0; j < n; ++j) {

            // Find the row with the maximum absolute value in the current column
            int maxIdx = j;

            for (int i = j + 1; i < n; ++i) {

                if (abs(A[i] [j]) > abs(A[maxIdx][j])) {

                    maxIdx = i;

                }

            }


            // Swap rows if necessary to bring the pivot element to the diagonal
            if (maxIdx != j) {

                swapRows(alpha, maxIdx, j);


                determinant *= -1.0;    // Each row swap changes the sign of the determinant
            }


            // Check if the pivot element is zero, matrix A is singular
            if (A[j][j] == 0) {

                cout << "Error: Matrix A is singular" << endl;

                return vector<vector<double>> (n, vector<double>(n, 0.0));    // Return zero matrix
            }


            // Scale the row to make the pivot element 1
            double pivot = A[j][j];

            for (int k = j; k < n; ++k) {

                A[j][k]= A[j][k] / pivot;

            }
            for (int k = 0; k < n; ++k) {

                identity [j] [k] =identity[j] [k] / pivot;

            }

            // Perform row operations to make all other elements in the column zero
            for (int i = 0; i < n; ++i) {

                if (i != j) {

                    double factor = A[i] [j];

                    for (int k = 0; k < n; ++k) {

                        A[i][k]=A[i][k] -factor * A[j][k];

                        identity[i][k]=identity [i][k]-
                                            factor * identity[j][k];


                    }
                }

            }
        }

        return identity;

    }

    void solve() override {

        int n = alpha.size();
        int m = beta.size();


        vector<vector<double>> B(n, vector<double>(n, 0));
        vector<vector<double>> C(n, vector<double>(n, 0));


        if(n!=m){
            cout<< "The method is not applicable\n";
            return;
        }

        bool isDominant = true;
        bool isSymmetric = true;
        double sum;
        for (int i = 0; i < n; i++) {
            sum = 0.0;
            for (int j = 0; j < n; j++) {
                if(i!=j){
                    sum+=alpha[i][j];
                    if(alpha[i][j]!=alpha[j][i]){
                        isSymmetric = false;
                    }
                }
            }
            if(alpha[i][i]<=sum){
                isDominant =false;
            }
            if (alpha[i][i] < sum) {
                isDominant = false;
                break;
            }
        }

        if(!isDominant && !isSymmetric){
            cout<< "The method is not applicable\n";
            return;
        }
        decompose(alpha, B, C, beta);


        vector<double> x = initializeX0(beta.size());


        vector<double> x_new;


        const int maxIterations = 1000;

        int iterationCount = 0;



        // Print alpha and beta
        cout << "alpha:" << endl;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                cout << fixed << setprecision(4) << alpha[i][j] << " ";
            }
            cout << endl;
        }

        cout << "beta:" << endl;
        for (int i = 0; i < n; ++i) {
            cout << fixed << setprecision(4) << beta[i] << endl;
        }

        cout << "B:" << endl;


        for (int i = 0; i < n; ++i) {


            for (int j = 0; j < n; ++j) {


                if (j <= i) {


                    cout << fixed << setprecision(4) << B[i][j] <<
                         " ";


                } else {


                    cout << fixed << setprecision(4) << 0.0 << " ";    // Print 0 for upper triangular elements
                }


            }


            cout << endl;


        }



// Print C as upper triangular matrix
        cout << "C:" << endl;


        for (int i = 0; i < n; ++i) {


            for (int j = 0; j < n; ++j) {


                if (j >= i) {


                    cout << fixed << setprecision(4) << C[i][j] <<
                         " ";


                } else {


                    cout << fixed << setprecision(4) << 0.0 << " ";    // Print 0 for lower triangular elements
                }


            }


            cout << endl;


        }



// Print (I - B) and (I - B)^(-1)
        cout << "I-B:" << endl;


        vector<vector<double>> I_minus_B(n, vector<double>(n, 0));


        for (int i = 0; i < n; ++i) {


            for (int j = 0; j < n; ++j) {

                if (i == j) {
                    I_minus_B[i][j]= 1.0 - B[i][j];
                }

                if (j < i) {


                    I_minus_B[i][j]= - B[i][j];


                }


                cout << fixed << setprecision(4) << I_minus_B[i][j] << " ";


            }


            cout << endl;


        }


        cout << "(I-B)_-1:" << endl;

        vector<vector<double>> inverse_IB = inverse(I_minus_B);

        int f = inverse_IB.size();

        for (int i = 0; i < f; ++i) {

            for (int j = 0; j < f; ++j) {

                cout << fixed << setprecision(4) << inverse_IB[i][j]
                                                                          <<
                     " ";

            }
            cout << endl;

        }
        do {


            x_new = seidelIteration(B, C, beta, x, I_minus_B);


            // Store the iteration result
            steps.push_back(new Vector(x_new));


            // Calculate accuracy
            accuracies.push_back(calculateAccuracy(x_new, x));


            // Update x for the next iteration
            x = x_new;

            iterationCount++;

        } while (accuracies.back() > epsilon &&
                 iterationCount < maxIterations);    // Check convergence or iteration limit



        // Print x(i) and accuracies
        for (size_t i = 1; i < steps.size(); ++i) {

            cout << "x(" << i << "):" << endl;

            int size = steps[i]->getSize();

            for (int j = 0; j < size; ++j) {

                cout << fixed << setprecision(4) << steps[i]->
                        getElement(j) << endl;

            }
            cout << "e: ";

            cout << fixed << setprecision(4) << accuracies[i] << endl;

        }

        // Output the final approximation
        cout << "x~:" << endl;

        int size = steps.back()->getSize();

        for (int i = 0; i < size; ++i) {

            cout << fixed << setprecision(4) << steps.back()->
                    getElement(i) << endl;

        }
        cout << endl;

        // Clean up allocated memory
        for (size_t i = 0; i < steps.size(); ++i) {
            delete steps[i];
        }
    }
};

int main() {
    int n;
    int m;
    double epsilon;


    cin >> n;
    vector<double> A(n * n);
    for (int i = 0; i < n * n; ++i) {
        cin >> A[i];
    }
    cin >> m;
    vector<double> b(m);
    for (int i = 0; i < m; ++i) {
        cin >> b[i];
    }
    if(n!=m){
        cout<< "The method is not applicable\n";
        return 0;
    }

    cin >> epsilon;

    vector<vector<double>> alpha;
    for (int i = 0; i < n; ++i) {
        alpha.push_back(vector<double>(A.begin() + i * n, A.begin() + (i + 1) * n));
    }

    vector<double> beta(b);

    SeidelMethod seidel(alpha, beta, epsilon);
    seidel.solve();

    return 0;
}
