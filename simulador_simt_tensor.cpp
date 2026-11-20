#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <omp.h>
#include <cmath>
#include <algorithm>

using namespace std;

constexpr int N = 1024;
constexpr int TILE = 16;
constexpr int WARP_SIZE = 32;

struct GpuTelemetry {
    long long cuda_core_fma_ops = 0;
    long long tensor_core_wmma_ops = 0;
    long long warp_serialized_passes = 0;
};

float to_fp16(float x) {
    return (float)(_Float16)x;
}

void gemm_cpu(const vector<float>& A,
              const vector<float>& B,
              vector<float>& C) {

    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++) {
            float a = A[i*N+k];
            for (int j = 0; j < N; j++)
                C[i*N+j] += a * B[k*N+j];
        }
}

void gemm_cuda_sim(const vector<float>& A,
                   const vector<float>& B,
                   vector<float>& C,
                   GpuTelemetry& stats) {

    #pragma omp parallel for collapse(2) schedule(static)
    for (int by = 0; by < N; by += TILE) {
        for (int bx = 0; bx < N; bx += TILE) {

            float acc[TILE][TILE] = {};

            for (int t = 0; t < N; t += TILE) {
                float sA[TILE][TILE];
                float sB[TILE][TILE];

                for (int i = 0; i < TILE; i++)
                    for (int j = 0; j < TILE; j++) {
                        sA[i][j] = A[(by+i)*N+t+j];
                        sB[i][j] = B[(t+i)*N+bx+j];
                    }

                for (int i = 0; i < TILE; i++)
                    for (int k = 0; k < TILE; k++)
                        for (int j = 0; j < TILE; j++)
                            acc[i][j] += sA[i][k]*sB[k][j];
            }

            for (int i = 0; i < TILE; i++)
                for (int j = 0; j < TILE; j++)
                    C[(by+i)*N+bx+j] = acc[i][j];
        }
    }

    stats.cuda_core_fma_ops = 1LL*N*N*N;
    stats.warp_serialized_passes = (1LL*N*N*N)/2;
}

void gemm_tensor_sim(const vector<float>& A,
                     const vector<float>& B,
                     vector<float>& C,
                     GpuTelemetry& stats) {

    #pragma omp parallel for collapse(2) schedule(static)
    for (int by = 0; by < N; by += TILE) {
        for (int bx = 0; bx < N; bx += TILE) {

            float acc[TILE][TILE] = {};

            for (int t = 0; t < N; t += TILE)
                for (int k = 0; k < TILE; k++)
                    for (int i = 0; i < TILE; i++) {
                        float a = to_fp16(A[(by+i)*N+t+k]);

                        for (int j = 0; j < TILE; j++) {
                            float b = to_fp16(B[(t+k)*N+bx+j]);
                            acc[i][j] += a*b;
                        }
                    }

            for (int i = 0; i < TILE; i++)
                for (int j = 0; j < TILE; j++)
                    C[(by+i)*N+bx+j] = acc[i][j];
        }
    }

    stats.tensor_core_wmma_ops =
        1LL*(N/TILE)*(N/TILE)*(N/TILE);
}

double tiempo(void (*func)(const vector<float>&,
                           const vector<float>&,
                           vector<float>&),
              const vector<float>& A,
              const vector<float>& B,
              vector<float>& C) {

    auto inicio = chrono::high_resolution_clock::now();

    func(A,B,C);

    auto fin = chrono::high_resolution_clock::now();

    return chrono::duration<double,milli>(fin-inicio).count();
}

int main() {

    cout << "========================================\n";
    cout << " SIMULADOR ARQUITECTONICO SIMT / WMMA\n";
    cout << " TEL-420 SISTEMAS PARALELOS - MODULO M4\n";
    cout << "========================================\n";

    cout << "Tamano de matrices: " << N << " x " << N << endl;
    cout << "Hilos OpenMP disponibles: "
         << omp_get_max_threads() << endl;

    vector<float> A(N*N, 0.5f);
    vector<float> B(N*N, 0.25f);

    vector<float> C1(N*N, 0.0f);
    vector<float> C2(N*N, 0.0f);
    vector<float> C3(N*N, 0.0f);

    GpuTelemetry stats;

    cout << "\n[1] GEMM CPU SECUENCIAL\n";

    double t1 = tiempo(gemm_cpu,A,B,C1);

    cout << "Tiempo: " << t1 << " ms\n";

    cout << "\n[2] SIMULACION CUDA CORES + TILES\n";

    auto inicio = chrono::high_resolution_clock::now();

    gemm_cuda_sim(A,B,C2,stats);

    auto fin = chrono::high_resolution_clock::now();

    double t2 = chrono::duration<double,milli>(fin-inicio).count();

    cout << "Tiempo: " << t2 << " ms\n";

    cout << "\n[3] SIMULACION TENSOR CORES WMMA\n";

    inicio = chrono::high_resolution_clock::now();

    gemm_tensor_sim(A,B,C3,stats);

    fin = chrono::high_resolution_clock::now();

    double t3 = chrono::duration<double,milli>(fin-inicio).count();

    cout << "Tiempo: " << t3 << " ms\n";

    float error = 0.0f;

    for (int i = 0; i < N*N; i++)
        error = max(error,abs(C1[i]-C3[i]));

    cout << "\n========== TELEMETRIA ==========\n";

    cout << "CUDA Core FMA: "
         << stats.cuda_core_fma_ops << endl;

    cout << "Instrucciones WMMA simuladas: "
         << stats.tensor_core_wmma_ops << endl;

    cout << "Reduccion teorica de despacho: "
         << (double)stats.cuda_core_fma_ops /
            stats.tensor_core_wmma_ops << "x\n";

    cout << "Pasadas serializadas simuladas: "
         << stats.warp_serialized_passes << endl;

    cout << "Eficiencia SIMT modelada con divergencia: 50%\n";

    cout << "Error maximo FP16 vs FP32: "
         << scientific << error << endl;

    cout << "\nValor esperado de C[0][0]: "
         << C1[0] << endl;

    cout << "\n========================================\n";
    cout << " FIN DE LA SIMULACION\n";
    cout << "========================================\n";

    return 0;
}
