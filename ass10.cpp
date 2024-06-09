#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
using namespace std;

struct Result {
    std::vector<double> t;
    std::vector<double> v;
    std::vector<double> k;
};

Result predatorPreyModel(double v0, double k0, double alpha1, double beta1, double alpha2, double beta2, double T, int N) {
    Result result;
    double dt = T / static_cast<double>(N);
    double V = v0 - alpha2/beta2;
    double K = k0 - alpha1/beta1;
    for (int i = 0; i < N; ++i) {
        double t = dt * i;

        double vt = (V * cos(sqrt(alpha1 * alpha2) * t)) - ((K * (sqrt(alpha2) * (beta1) * sin(sqrt(alpha1 * alpha2) * t))) / beta2 * sqrt(alpha1));
        double kt = ((V * (sqrt(alpha1) * (beta2)  * sin(sqrt(alpha1 * alpha2) * t)))/ beta1* sqrt(alpha2)) + K * cos(sqrt(alpha1 * alpha2) * t);
        result.t.push_back(t);
        result.v.push_back(vt);
        result.k.push_back(kt);
    }


    return result;
}

int main() {
    double v0, k0, alpha1, beta1, alpha2, beta2, T;
    int N;

    std::cin >> v0 >> k0 >> alpha1 >> beta1 >> alpha2 >> beta2 >> T >> N;
    Result result = predatorPreyModel(v0, k0, alpha1, beta1, alpha2, beta2, T, N);

    std::cout << "t:";
    for (double t : result.t)
        std::cout <<fixed << setprecision(2) << " " << t;
    std::cout << std::endl;

    std::cout << "v:";
    for (double v : result.v)
        std::cout <<fixed << setprecision(2) << " " << v;
    std::cout << std::endl;

    std::cout << "k:";
    for (double k : result.k)
        std::cout <<fixed << setprecision(2) << " " << k;
    std::cout << std::endl;

    return 0;
}
