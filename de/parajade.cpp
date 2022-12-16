#define _USE_MATH_DEFINES
#define _CRT_SECURE_NO_WARNINGS

#include <vector>
#include <array>
#include <random>
#include <cmath>
#include <algorithm>
#include <tuple>
#include <cassert>
#include <numeric>
#include <ctime>
#include <cstdio>

#include "thread_pool.hpp"

using namespace std;
using BS::thread_pool_light;

constexpr int NP = 64;
constexpr int NDIMS = 200;

constexpr size_t LAST_K = 5000;
constexpr double MIN_MEAN_DELTA = 1e-10;

constexpr int P_NP = max(1, int(0.05 * NP));
constexpr double C = 0.1;

constexpr int MAX_EPOCHS = 10'000'000;

constexpr int n_threads = 8;
static_assert(NP % n_threads == 0);
constexpr int n_thread_pop = NP / n_threads;

using Specie = array<double, NDIMS>;

double mean_last_diff(const double* x, size_t n, size_t k) {
    assert(n >= 2 && k >= 2);
    double s = 0;
    k = min(k, n);
    for (size_t i = n - k; i < n; ++i)
        s += fabs(x[i] - x[i - 1]);
    return s / (n - 1);
}

double fitness(const Specie& x) {
    constexpr double A = 10;

    double s = A * NDIMS;
    for (int i = 0; i < NDIMS; ++i)
        s += x[i] * x[i] - A * cos(2 * M_PI * x[i]);

    return s;
}

struct WorkerContext {
    vector<tuple<int, double, Specie>> offspring;
    vector<double> s_cr, s_f;

    default_random_engine rng;

    //read-only data
    const Specie* pop;
    const double* pop_ft;
    const int* sorted_ics;
    const Specie* archive;
    int archive_size;

    double nu_cr, nu_f;

    int first_idx;
};

void worker_routine(WorkerContext* pctx) {
    WorkerContext& ctx = *pctx;
    ctx.offspring.clear();
    ctx.s_cr.clear();
    ctx.s_f.clear();

    uniform_int_distribution<int> randi(0, 2 * NP - 1);
    normal_distribution<double> randn(ctx.nu_cr, 0.1);
    cauchy_distribution<double> randc(ctx.nu_f, 0.1);
    uniform_real_distribution<double> randf(0, 1);

    Specie y;
    int last_idx = ctx.first_idx + n_thread_pop - 1;
    for (int i = ctx.first_idx; i <= last_idx; ++i) {
        int x1_idx = 0, x2_idx = 0;
        int pbest_idx = ctx.sorted_ics[randi(ctx.rng) % P_NP];
        int R = randi(ctx.rng) % NDIMS;

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

        double cr = clamp(randn(ctx.rng), 0.0, 1.0),
              f = clamp(randc(ctx.rng), 0.0, 1.0);
        for (int j = 0; j < NDIMS; ++j) {
            int flag = randf(ctx.rng) <= cr || j == R;
            y[j] = x[j] + (f * (pbest[j] - x[j]) + f * (x1[j] - x2[j])) * flag;
        }

        double fx = ctx.pop_ft[i], fy = fitness(y);
        if (fy <= fx) {
            ctx.offspring.emplace_back(i, fy, y);
            ctx.s_cr.push_back(cr);
            ctx.s_f.push_back(f);
        }
    }
}

void run(uint64_t seed, vector<double> &hist, bool silent = false) 
{
    vector<Specie> pop(NP);
    vector<double> pop_ft(NP);
    vector<int> sorted_ics(NP);

    default_random_engine rng(seed);
    normal_distribution<double> randn;
    for (int i = 0; i < NP; ++i) {
        for (int j = 0; j < NDIMS; ++j)
            pop[i][j] = randn(rng);
        pop_ft[i] = fitness(pop[i]);
        sorted_ics[i] = i;
    }

    vector<Specie> archive;
    archive.reserve(2 * NP);

    vector<WorkerContext> wcs(n_threads);
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

    double nu_cr = 0.5, nu_f = 0.5;
    for (int epoch = 0; epoch < MAX_EPOCHS; ++epoch) {
        partial_sort(
            begin(sorted_ics), 
            begin(sorted_ics) + P_NP, 
            end(sorted_ics), 
            [&](int p, int q) { 
                return pop_ft[p] < pop_ft[q]; 
            }
        );

        constexpr int T = 10;
        if (epoch % T == 0) {
			hist.push_back(pop_ft[sorted_ics[0]]);
			if (hist.size() > 1) {
				double diff = mean_last_diff(hist.data(), hist.size(), LAST_K / T);
				if (diff < MIN_MEAN_DELTA)
					break;
			}
        }

        for (int i = 0; i < n_threads; ++i) {
            wcs[i].nu_cr = nu_cr;
            wcs[i].nu_f = nu_f;
            wcs[i].archive_size = archive.size();
            pool.push_task(worker_routine, &wcs[i]);
        }

        pool.wait_for_tasks();

        int total_successes = 0;
        double sum_cr = 0, sum_f = 0, sum_f2 = 0;
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
            double mean_A = sum_cr / total_successes,
                  mean_L = sum_f > 1e-8 ? sum_f2 / sum_f : 0;
            nu_cr = (1 - C) * nu_cr + C * mean_A;
            nu_f = (1 - C) * nu_f + C * mean_L;
        }

        if (epoch % 100 == 0 || epoch + 1 == MAX_EPOCHS) {
            double mean = accumulate(pop_ft.begin(), pop_ft.end(), 0.0) / NP;
            double best = *min_element(pop_ft.begin(), pop_ft.end());

            time_t t = time(0);
            auto now = localtime(&t);

            char output[256]{};
            sprintf(output, "[ %02d:%02d:%02d ] %05d mean: %.8f best: %.8f\n",
                    now->tm_hour, now->tm_min, now->tm_sec, 
                    epoch, mean, best);
            if (!silent)
                printf("%s", output);
        }
    }
}

int main() {
    vector<double> hist;
    hist.reserve(10240);

    char fname[256]{};
	sprintf(fname, "%d_%d.txt", NDIMS, NP);
	FILE* file = fopen(fname, "w");

    for (int i = 0; i < 20; ++i) {
        hist.clear();
        run(i, hist, false);

        for (double j : hist)
            fprintf(file, "%f ", j);
        fprintf(file, "\n");
    }

	fclose(file);
}
