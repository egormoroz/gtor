#define _USE_MATH_DEFINES

#include <vector>
#include <array>
#include <random>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <memory>

using namespace std;

constexpr int NP = 1000;
constexpr int NDIMS = 1000;

constexpr int P_NP = max(1, int(0.05 * NP));
constexpr float C = 0.1;

constexpr int EPOCHS = 100000;

using Specie = array<float, NDIMS>;
using Population = array<Specie, NP>;

template<typename It>
float mean_A(It begin, It end) {
    float s = 0;
    int n = end - begin;
    for (; begin != end; ++begin) {
        s += *begin;
    }
    return s / n;
}

template<typename It>
float mean_L(It begin, It end) {
    float s1 = 0, s2 = 0;
    for (; begin != end; ++begin) {
        s1 += *begin;
        s2 += *begin * *begin;
    }

    return fabs(s1) < 1e-8 ? 0 : s2 / s1;
}

float fitness(const Specie& x) {
    constexpr float A = 10;

    float s = A * NDIMS;
    for (int i = 0; i < NDIMS; ++i)
        s += x[i] * x[i] - A * cosf(2 * M_PI * x[i]);

    return s;
}

pair<int, float> best_fitness(const vector<Specie>& pops) {
    int best_idx = 0;
    float best_val = fitness(pops[0]);
    for (int i = 1; i < NP; ++i) {
        float f = fitness(pops[i]);
        if (f < best_val) {
            best_val = f;
            best_idx = i;
        }
    }
        
    return {best_idx, best_val};
}

float mean_fitness(const vector<Specie>& pops) {
    float s = 0;
    for (const Specie& x: pops)
        s += fitness(x);
    return s / NP;
}

int main() {
    vector<Specie> pops(NP);
    default_random_engine rng(1337);

    uniform_real_distribution<float> pdst(0, 1);
    normal_distribution<float> ndst;
    uniform_int_distribution<int> idst(0, NP - 1);
    uniform_int_distribution<int> rdst(0, NDIMS - 1);

    for (Specie& x: pops)
        for (float& i: x)
            i = ndst(rng);

    vector<Specie> old_pops(NP);
    Specie y;
    float best_val = best_fitness(pops).second;

    vector<Specie> archive;
    archive.reserve(2 * NP);

    vector<int> sorted_ics(NP);
    iota(begin(sorted_ics), end(sorted_ics), 0);

    float nu_cr = 0.5, nu_f = 0.5;
    vector<float> s_cr, s_f;
    s_cr.reserve(NP);
    s_f.reserve(NP);

    for (int epoch = 0; epoch < EPOCHS && best_val > 1e-8; ++epoch) {
        old_pops = pops;
        partial_sort(begin(sorted_ics), begin(sorted_ics) + P_NP, end(sorted_ics), 
                [&](int p, int q) { return fitness(pops[p]) < fitness(pops[q]); });

        normal_distribution<float> randn(nu_cr, 0.1);
        cauchy_distribution<float> randc(nu_f, 0.1);

        s_cr.clear();
        s_f.clear();

        //TODO: parallelize me!
        for (int i = 0; i < NP; ++i) {
            int x1_idx = 0, x2_idx = 0;
            int pbest_idx = sorted_ics[idst(rng) % P_NP];
            int R = rdst(rng);


            for(x1_idx = idst(rng); x1_idx == i || x1_idx == pbest_idx; 
                    x1_idx = idst(rng));

            uniform_int_distribution<int> d(0, NP + archive.size() - 1);
            for(x2_idx = d(rng); x2_idx == i || x2_idx == pbest_idx 
                    || x2_idx == x1_idx; x2_idx = d(rng));

            const Specie& x = old_pops[i];
            const Specie& pbest = old_pops[pbest_idx];
            const Specie& x1 = old_pops[x1_idx];
            const Specie& x2 = x2_idx >= NP 
                ? archive[x2_idx - NP] : old_pops[x2_idx];

            float cr = min(1.f, max(0.f, randn(rng))), 
                  f = min(1.f, max(0.f, randc(rng)));
            for (int j = 0; j < NDIMS; ++j) {
                int flag = pdst(rng) <= cr || j == R;
                y[j] = x[j] + (f * (pbest[j] - x[j]) + f * (x1[j] - x2[j])) * flag;
            }

            float fx = fitness(x), fy = fitness(y);
            if (fy <= fx) {
                pops[i] = y;
                archive.push_back(x);
                s_cr.push_back(cr);
                s_f.push_back(f);
            }
        }

        for (int i = NP; i < (int)archive.size(); ++i)
            swap(archive[i - NP], archive[i]);
        if (archive.size() > NP)
            archive.resize(NP);

        nu_cr = (1 - C) * nu_cr + C * mean_A(s_cr.begin(), s_cr.end());
        nu_f = (1 - C) * nu_f + C * mean_L(s_f.begin(), s_f.end());

        float old_best_val = best_val;
        best_val = best_fitness(pops).second;

        if (best_val < old_best_val) {
            float mean = mean_fitness(pops);
            printf("%03d mean: %.8f best: %.8f\n", epoch, mean, best_val);
        }
    }

    return 0;
}

