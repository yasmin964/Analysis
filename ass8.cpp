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

class JacobiMethod : public IterativeMethod {
private:
    vector<Vector*> steps;
    vector<double> accuracies;

public:
    JacobiMethod(  vector<vector<double>> alpha,vector<double> beta, double epsilon)
            : IterativeMethod(alpha, beta, epsilon) {}

    void decompose(vector<vector<double>> A, vector<double> b) {
        int n = A.size();

        for (int i = 0; i < n; ++i) {
            // Update the elements of the vector beta
            beta[i]= b[i] / A[i][i];
            for (int j = 0; j < n; ++j) {
                if (i != j) {
                    // Assign values to the off-diagonal elements of alpha
                    alpha[i][j]= - A[i][j] / A[i][i];
                    if(alpha[i][j] == -0.0){
                        alpha[i][j] ==
                    }
                } else {
                    // Assign zeros to the diagonal elements of alpha
                    alpha[i][j]= 0.0;
                }
            }
        }
    }



    vector<double> initializeX0(int n) {
        return vector<double>(n, 0.0);
    }

    vector<double> jacobiIteration(const Vector& b, const vector<double>& x_prev) {
        int n = x_prev.size();
        vector<double> x_curr(n, 0.0);

        // Compute the current iteration using Jacobi method
        for (int i = 0; i < n; ++i) {
            x_curr[i] = beta[i];
            for (int j = 0; j < n; ++j) {
                if (i != j) {
                    x_curr[i] += alpha[i][j] * x_prev[j];
                }
            }
        }
        return x_curr;

    }



    double calculateAccuracy(const vector<double>& x_new, const vector<double>& x) {
        int n = x.size();
        double max_diff = 0.0;
        for (int i = 0; i < n; ++i) {
            double diff = (x_new[i] - x[i]) * (x_new[i] - x[i]);
            max_diff += diff;
        }
        max_diff = sqrt(max_diff);
        return max_diff;
    }

    void solve() override {
        // Initialize alpha matrix and beta vector
        int n = alpha.size();
        int m = beta.size();
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


        decompose(alpha, beta);

        // Initialize x with zeros
        vector<double> x = initializeX0(beta.size());
        vector<double> x_new;
        const int maxIterations = 1000; // Adjust this value as needed
        int iterationCount = 0;

        // Iterate until convergence
        do {
            // Perform Jacobi iteration
            x_new = jacobiIteration(beta, x);

            // Store the iteration result
            steps.push_back(new Vector(x_new));

            // Calculate accuracy
            accuracies.push_back(calculateAccuracy(x_new, x));

            // Update x for the next iteration
            x = x_new;
            iterationCount++;
        } while (accuracies.back() > epsilon && iterationCount < maxIterations); // Check convergence or iteration limit



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

        // Print x(i) and accuracies
        for (size_t i = 1; i < steps.size(); ++i) {
            cout << "x(" << i << "):" << endl;
            int size = steps[i]->getSize();
            for (int j = 0; j < size; ++j) {
                cout << fixed << setprecision(4) << steps[i]->getElement(j) << endl;
            }
            cout << "e:";
            cout << fixed << setprecision(4) << accuracies[i] << endl;
        }

        // Output the final approximation
        cout << "x~:" << endl;
        int size = steps.back()->getSize();
        for (int i = 0; i < size; ++i) {
            cout << fixed << setprecision(4) << steps.back()->getElement(i) << endl;
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

    JacobiMethod jacobi(alpha, beta, epsilon);
    jacobi.solve();

    return 0;
}
