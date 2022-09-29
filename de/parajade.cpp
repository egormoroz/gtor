#define _USE_MATH_DEFINES

#include <vector>
#include <array>
#include <random>
#include <cmath>
#include <algorithm>
#include <memory>
#include <tuple>
#include <cassert>

#include "thread_pool.hpp"

using namespace std;
using BS::thread_pool_light;

constexpr int NP = 200;
constexpr int NDIMS = 1000;

constexpr int P_NP = max(1, int(0.05 * NP));
constexpr float C = 0.1f;

constexpr int EPOCHS = 10;

constexpr int n_threads = 4;
static_assert(NP % n_threads == 0);
constexpr int n_thread_pop = NP / n_threads;

using Specie = array<float, NDIMS>;
using Population = array<Specie, NP>;

float fitness(const Specie& x) {
    constexpr float A = 10;

    float s = A * NDIMS;
    for (int i = 0; i < NDIMS; ++i)
        s += x[i] * x[i] - A * cosf(2 * M_PI * x[i]);

    return s;
}

pair<int, float> best_fitness(const vector<Specie>& pop) {
    int best_idx = 0;
    float best_val = fitness(pop[0]);
    for (int i = 1; i < NP; ++i) {
        float f = fitness(pop[i]);
        if (f < best_val) {
            best_val = f;
            best_idx = i;
        }
    }
        
    return {best_idx, best_val};
}

float mean_fitness(const vector<Specie>& pop) {
    float s = 0;
    for (const Specie& x: pop)
        s += fitness(x);
    return s / NP;
}

struct WorkerContext {
    vector<tuple<int, float, Specie>> offspring;
    vector<float> s_cr, s_f;

    default_random_engine rng;

    //read-only data
    const Specie* pop;
    const float* pop_ft;
    const int* sorted_ics;
    const Specie* archive;
    int archive_size;

    float nu_cr, nu_f;

    int first_idx;
};

void worker_routine(WorkerContext* pctx) {
    WorkerContext& ctx = *pctx;
    ctx.offspring.clear();
    ctx.s_cr.clear();
    ctx.s_f.clear();

    uniform_int_distribution<int> randi(0, 2 * NP - 1);
    normal_distribution<float> randn(ctx.nu_cr, 0.1f);
    cauchy_distribution<float> randc(ctx.nu_f, 0.1f);
    uniform_real_distribution<float> randf(0, 1);

    Specie y;
    int last_idx = ctx.first_idx + n_thread_pop - 1;
    for (int i = ctx.first_idx; i <= last_idx; ++i) {
        int x1_idx = 0, x2_idx = 0;
        int pbest_idx = ctx.sorted_ics[randi(ctx.rng) % P_NP];
        int R = randi(ctx.rng);

        for(x1_idx = randi(ctx.rng) % NP; 
            x1_idx == i || x1_idx == pbest_idx; 
            x1_idx = randi(ctx.rng) % NP);
        
        int n = NP + ctx.archive_size;
        for(x2_idx = randi(ctx.rng) % n; 
            x2_idx == i || x2_idx == pbest_idx || x2_idx == x1_idx; 
            x2_idx = randi(ctx.rng) % n);

        const Specie& x = ctx.pop[i];
        const Specie& pbest = ctx.pop[pbest_idx];
        const Specie& x1 = ctx.pop[x1_idx];
        const Specie& x2 = x2_idx >= NP
            ? ctx.archive[x2_idx - NP] : ctx.pop[x2_idx];

        float cr = clamp(randn(ctx.rng), 0.f, 1.f),
              f = clamp(randn(ctx.rng), 0.f, 1.f);
        for (int j = 0; j < NDIMS; ++j) {
            int flag = randf(ctx.rng) <= cr || j == R;
            y[j] = x[j] + (f * (pbest[j] - x[j]) + f * (x1[j] - x2[j])) * flag;
        }

        float fx = ctx.pop_ft[i], fy = fitness(y);
        if (fy <= fx) {
            ctx.offspring.emplace_back(i, fy, y);
            ctx.s_cr.push_back(cr);
            ctx.s_f.push_back(f);
        }
    }
}

int main() {
    vector<Specie> pop(NP);
    vector<float> pop_ft(NP);
    vector<int> sorted_ics(NP);

    default_random_engine rng(1337);
    normal_distribution<float> randn;
    for (int i = 0; i < NP; ++i) {
        for (int j = 0; j < NDIMS; ++j)
            pop[i][j] = randn(rng);
        pop_ft[i] = fitness(pop[i]);
        sorted_ics[i] = i;
    }

    vector<Specie> archive;
    archive.reserve(2 * NP);

    WorkerContext wcs[n_threads];
    uniform_int_distribution<unsigned int> randi(0, 0xFFFFFFFF);
    for (int i = 0; i < n_threads; ++i) {
        WorkerContext& ctx = wcs[i];

        ctx.offspring.reserve(n_thread_pop);
        ctx.s_cr.reserve(n_thread_pop);
        ctx.s_f.reserve(n_thread_pop);

        ctx.rng.seed(randi(rng));
        ctx.pop = pop.data();
        ctx.pop_ft = pop_ft.data();
        ctx.sorted_ics = sorted_ics.data();
        ctx.archive = archive.data();
        ctx.archive_size = 0;

        ctx.first_idx = n_thread_pop * i;
    }

    thread_pool_light pool(n_threads);

    float nu_cr = 0.5, nu_f = 0.5;
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        partial_sort(
            begin(sorted_ics), 
            begin(sorted_ics) + P_NP, 
            end(sorted_ics), 
            [&](int p, int q) { 
                return fitness(pop[p]) < fitness(pop[q]); 
            }
        );

        for (int i = 0; i < n_threads; ++i) {
            wcs[i].nu_cr = nu_cr;
            wcs[i].nu_f = nu_f;
            wcs[i].archive_size = archive.size();
            pool.push_task(worker_routine, &wcs[i]);
        }

        pool.wait_for_tasks();

        int total_successes = 0;
        float sum_cr = 0, sum_f = 0, sum_f2 = 0;
        for (int i = 0; i < n_threads; ++i) {
            WorkerContext& ctx = wcs[i];
            for (auto& [idx, ft, y]: ctx.offspring) {
                archive.push_back(pop[idx]);
                pop[idx] = y;
                pop_ft[idx] = ft;
            }

            int successes = ctx.offspring.size();
            total_successes += successes;
            for (int j = 0; j < successes; ++j) {
                sum_cr += ctx.s_cr[j];
                sum_f += ctx.s_f[j];
                sum_f2 += ctx.s_f[j] * ctx.s_f[j];
            }
        }

        for (int i = NP; i < (int)archive.size(); ++i)
            swap(archive[i - NP], archive[i]);
        if (archive.size() > NP)
            archive.resize(NP);

        if (total_successes) {
            assert(sum_f > 1e-8);
            float mean_A = sum_cr / total_successes,
                  mean_L = sum_f2 / sum_f;
            nu_cr = (1 - C) * nu_cr + C * mean_A;
            nu_f = (1 - C) * nu_f + C * mean_L;
        }

        float mean = mean_fitness(pop),
              best = best_fitness(pop).second;
        printf("%03d mean: %.8f best: %.8f\n", epoch, 
                mean, best);
    }
}

