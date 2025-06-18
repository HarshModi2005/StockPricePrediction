#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define N 10    // Number of states
#define M 20   // Number of observation symbols
#define MAX_T 4147  // Maximum number of time periods
#define MAX_ITER 100  // Maximum number of iterations for Baum-Welch

// Stock data structure
typedef struct {
    double open;
    double high;
    double low;
    double close;
} StockData;

// Function to normalize stock data using min-max normalization
void normalize_stock_data(StockData stock_prices[], int num_periods) {
    double min_open = stock_prices[0].open, max_open = stock_prices[0].open;
    double min_high = stock_prices[0].high, max_high = stock_prices[0].high;
    double min_low = stock_prices[0].low, max_low = stock_prices[0].low;
    double min_close = stock_prices[0].close, max_close = stock_prices[0].close;

    // Find min and max values for each feature
    for (int i = 1; i < num_periods; i++) {
        if (stock_prices[i].open < min_open) min_open = stock_prices[i].open;
        if (stock_prices[i].open > max_open) max_open = stock_prices[i].open;
        if (stock_prices[i].high < min_high) min_high = stock_prices[i].high;
        if (stock_prices[i].high > max_high) max_high = stock_prices[i].high;
        if (stock_prices[i].low < min_low) min_low = stock_prices[i].low;
        if (stock_prices[i].low > max_low) max_low = stock_prices[i].low;
        if (stock_prices[i].close < min_close) min_close = stock_prices[i].close;
        if (stock_prices[i].close > max_close) max_close = stock_prices[i].close;
    }

    // Apply min-max normalization to each feature
    for (int i = 0; i < num_periods; i++) {
        stock_prices[i].open = (stock_prices[i].open - min_open) / (max_open - min_open);
        stock_prices[i].high = (stock_prices[i].high - min_high) / (max_high - min_high);
        stock_prices[i].low = (stock_prices[i].low - min_low) / (max_low - min_low);
        stock_prices[i].close = (stock_prices[i].close - min_close) / (max_close - min_close);
    }
}

// Function to categorize stock price movements
int categorize_movement(double open, double close, double high, double low) {
    double change = ((close - open) / open) * 100;  // Percentage change
    double range = high - low;

    // Example categories with percentage change classification
    if (change > 3.0 && range < 1.0) return 0;   // Strong small increase
    if (change > 2.0) return 1;                  // Large increase
    if (change > 1.0) return 2;                  // Moderate increase
    if (change > 0.0) return 3;                  // Small increase
    if (change < -3.0) return 4;                 // Strong large decrease
    if (change < -2.0) return 5;                 // Large decrease
    if (change < -1.0) return 6;                 // Moderate decrease
    if (change < 0.0) return 7;                  // Small decrease
    if (range > 3.0) return 8;                   // Large volatility, stable close
    return 9;                                    // Neutral or small change
}

// Function to read stock data from a CSV file
int read_stock_data(const char *filename, StockData stock_prices[], int *num_periods) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Error: Could not open file.\n");
        return 1;
    }

    char line[256];
    int i = 0;

    // Skip the header line
    fgets(line, sizeof(line), file);

    // Read stock data from the file
    while (fgets(line, sizeof(line), file) && i < MAX_T) {
        // Line format: "Index Name","Date","Open","High","Low","Close"
        char index_name[50], date[50];
        if (sscanf(line, "\"%[^\"]\",\"%[^\"]\",\"%lf\",\"%lf\",\"%lf\",\"%lf\"",
                   index_name, date,
                   &stock_prices[i].open,
                   &stock_prices[i].high,
                   &stock_prices[i].low,
                   &stock_prices[i].close) != 6) {
            printf("Error: Incorrect file format on line %d: %s\n", i + 1, line);
            fclose(file);
            return 1;
        }
        i++;
    }

    fclose(file);
    *num_periods = i;
    return 0;
}

// Initial HMM parameters
double A[N][N] = {
    {0.3, 0.2, 0.2, 0.2, 0.1},
    {0.2, 0.3, 0.2, 0.1, 0.2},
    {0.1, 0.2, 0.4, 0.1, 0.2},
    {0.3, 0.1, 0.1, 0.4, 0.1},
    {0.2, 0.2, 0.2, 0.2, 0.2}
};

double B[N][M] = {
    {0.3, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05},
    {0.1, 0.3, 0.2, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05},
    {0.05, 0.1, 0.3, 0.2, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05},
    {0.05, 0.05, 0.1, 0.3, 0.2, 0.1, 0.05, 0.05, 0.05, 0.05},
    {0.05, 0.05, 0.05, 0.1, 0.3, 0.2, 0.1, 0.05, 0.05, 0.05}
};

double pi[N] = {0.2, 0.2, 0.2, 0.2, 0.2};  // Initial state probabilities

// Forward algorithm for HMM with scaling
double forward_algorithm(int T, double alpha[][N], double A[N][N], double B[N][M], double pi[N], int observations[], double c[]) {
    // Initialization
    c[0] = 0;
    for (int i = 0; i < N; i++) {
        alpha[0][i] = pi[i] * B[i][observations[0]];
        c[0] += alpha[0][i];
    }
    // Scale alpha[0][i]
    if (c[0] == 0) c[0] = 1e-10;  // Avoid division by zero
    c[0] = 1.0 / c[0];
    for (int i = 0; i < N; i++) {
        alpha[0][i] *= c[0];
    }

    // Induction
    for (int t = 1; t < T; t++) {
        c[t] = 0;
        for (int j = 0; j < N; j++) {
            alpha[t][j] = 0.0;
            for (int i = 0; i < N; i++) {
                alpha[t][j] += alpha[t - 1][i] * A[i][j];
            }
            alpha[t][j] *= B[j][observations[t]];
            c[t] += alpha[t][j];
        }
        // Scale alpha[t][j]
        if (c[t] == 0) c[t] = 1e-10;
        c[t] = 1.0 / c[t];
        for (int j = 0; j < N; j++) {
            alpha[t][j] *= c[t];
        }
    }

    // Compute log probability
    double logProb = 0.0;
    for (int t = 0; t < T; t++) {
        logProb += log(c[t]);
    }
    logProb = -logProb;

    return logProb;  // Log probability of the observation sequence
}

// Backward algorithm for HMM with scaling
void backward_algorithm(int T, double beta[][N], double A[N][N], double B[N][M], int observations[], double c[]) {
    // Initialization
    for (int i = 0; i < N; i++) {
        beta[T - 1][i] = c[T - 1];
    }

    // Induction
    for (int t = T - 2; t >= 0; t--) {
        for (int i = 0; i < N; i++) {
            beta[t][i] = 0.0;
            for (int j = 0; j < N; j++) {
                beta[t][i] += A[i][j] * B[j][observations[t + 1]] * beta[t + 1][j];
            }
            // Scale beta[t][i]
            beta[t][i] *= c[t];
        }
    }
}

// Compute gamma values
void compute_gamma(int T, double alpha[][N], double beta[][N], double gamma[][N]) {
    for (int t = 0; t < T; t++) {
        double denom = 0.0;
        for (int i = 0; i < N; i++) {
            gamma[t][i] = alpha[t][i] * beta[t][i];
            denom += gamma[t][i];
        }
        // Normalize gamma[t][i]
        if (denom == 0) denom = 1e-10;
        for (int i = 0; i < N; i++) {
            gamma[t][i] /= denom;
        }
    }
}

// Compute xi values
void compute_xi(int T, double alpha[][N], double beta[][N], double xi[][N][N], double A[N][N], double B[N][M], int observations[]) {
    for (int t = 0; t < T - 1; t++) {
        double denom = 0.0;
        // Compute denominator
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                denom += alpha[t][i] * A[i][j] * B[j][observations[t + 1]] * beta[t + 1][j];
            }
        }
        if (denom == 0) denom = 1e-10;
        // Compute xi[t][i][j]
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                xi[t][i][j] = (alpha[t][i] * A[i][j] * B[j][observations[t + 1]] * beta[t + 1][j]) / denom;
            }
        }
    }
}

// Update model parameters
void update_model(int T, double gamma[][N], double xi[][N][N], int observations[], double A[N][N], double B[N][M], double pi[N]) {
    // Update initial state probabilities pi
    for (int i = 0; i < N; i++) {
        pi[i] = gamma[0][i];
    }

    // Update transition probabilities A
    for (int i = 0; i < N; i++) {
        double denom = 0.0;
        for (int t = 0; t < T - 1; t++) {
            denom += gamma[t][i];
        }
        if (denom == 0) denom = 1e-10;
        for (int j = 0; j < N; j++) {
            double numer = 0.0;
            for (int t = 0; t < T - 1; t++) {
                numer += xi[t][i][j];
            }
            A[i][j] = numer / denom;
            if (A[i][j] < 0) A[i][j] = 0;
        }
        // Normalize A[i][:]
        double sumA = 0.0;
        for (int j = 0; j < N; j++) {
            sumA += A[i][j];
        }
        if (sumA == 0) sumA = 1e-10;
        for (int j = 0; j < N; j++) {
            A[i][j] /= sumA;
        }
    }

    // Update emission probabilities B
    for (int j = 0; j < N; j++) {
        double denom = 0.0;
        for (int t = 0; t < T; t++) {
            denom += gamma[t][j];
        }
        if (denom == 0) denom = 1e-10;
        for (int k = 0; k < M; k++) {
            double numer = 0.0;
            for (int t = 0; t < T; t++) {
                if (observations[t] == k) {
                    numer += gamma[t][j];
                }
            }
            B[j][k] = numer / denom;
            if (B[j][k] < 0) B[j][k] = 0;
        }
        // Normalize B[j][:]
        double sumB = 0.0;
        for (int k = 0; k < M; k++) {
            sumB += B[j][k];
        }
        if (sumB == 0) sumB = 1e-10;
        for (int k = 0; k < M; k++) {
            B[j][k] /= sumB;
        }
    }
}

// Baum-Welch algorithm for training HMM with scaling
void baum_welch(int T, int observations[], double A[N][N], double B[N][M], double pi[N]) {
    double alpha[MAX_T][N];
    double beta[MAX_T][N];
    double gamma[MAX_T][N];
    double xi[MAX_T][N][N];
    double c[MAX_T];

    double oldLogProb = -INFINITY;

    for (int iter = 0; iter < MAX_ITER; iter++) {
        // Compute alpha
        double logProb = forward_algorithm(T, alpha, A, B, pi, observations, c);
        // Compute beta
        backward_algorithm(T, beta, A, B, observations, c);
        // Compute gamma
        compute_gamma(T, alpha, beta, gamma);
        // Compute xi
        compute_xi(T, alpha, beta, xi, A, B, observations);
        // Update model parameters
        update_model(T, gamma, xi, observations, A, B, pi);

        // Check for convergence
        if (iter > 0 && fabs(logProb - oldLogProb) < 1e-4) {
            break;
        }
        oldLogProb = logProb;
    }
}

// Main function
int main() {
    srand(time(0));  // Initialize random seed

    StockData stock_prices[MAX_T];
    int num_periods = 0;

    // Specify the path to the CSV file
    const char *filename = "../data/sample_nasa_data.csv";

    // Read stock data from file
    if (read_stock_data(filename, stock_prices, &num_periods) != 0) {
        return 1;
    }

    // Normalize stock data before categorizing movements
    normalize_stock_data(stock_prices, num_periods);

    // Generate observation sequence based on stock price movements
    int observations[MAX_T];
    for (int i = 0; i < num_periods; i++) {
        observations[i] = categorize_movement(stock_prices[i].open, stock_prices[i].close, stock_prices[i].high, stock_prices[i].low);
    }

    // Prediction lengths
    int prediction_lengths[] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    int num_predictions = 10;

    for (int idx = 0; idx < num_predictions; idx++) {
        int pred_len = prediction_lengths[idx];
        int T_train = num_periods - pred_len;

        // Copy initial A, B, pi to local variables
        double A_local[N][N];
        double B_local[N][M];
        double pi_local[N];

        // Initialize A_local, B_local, pi_local to initial values
        memcpy(A_local, A, sizeof(A));
        memcpy(B_local, B, sizeof(B));
        memcpy(pi_local, pi, sizeof(pi));

        // Prepare training observations
        int observations_train[MAX_T];
        memcpy(observations_train, observations, T_train * sizeof(int));

        // Train HMM on observations_train[0..T_train -1]
        baum_welch(T_train, observations_train, A_local, B_local, pi_local);

        // Run forward_algorithm to get alpha[0..T_train -1][i], c[0..T_train -1]
        double alpha[MAX_T][N];
        double c[MAX_T];
        forward_algorithm(T_train, alpha, A_local, B_local, pi_local, observations_train, c);

        // Run backward_algorithm to get beta[0..T_train -1][i]
        double beta[MAX_T][N];
        backward_algorithm(T_train, beta, A_local, B_local, observations_train, c);

        // Compute gamma[0..T_train -1][i]
        double gamma[MAX_T][N];
        compute_gamma(T_train, alpha, beta, gamma);

        // Get current_state_probs[i] = gamma[T_train -1][i]
        double current_state_probs[N];
        for (int i = 0; i < N; i++) {
            current_state_probs[i] = gamma[T_train - 1][i];
        }

        // Initialize loss to 0
        double loss = 0.0;

        // Predict and compute loss
        for (int t = T_train; t < T_train + pred_len; t++) {
            // Update state probabilities: current_state_probs = current_state_probs * A_local
            double new_state_probs[N];
            for (int i = 0; i < N; i++) {
                new_state_probs[i] = 0.0;
                for (int j = 0; j < N; j++) {
                    new_state_probs[i] += current_state_probs[j] * A_local[j][i];
                }
            }

            // Compute observation probabilities
            double observation_probs[M];
            for (int k = 0; k < M; k++) {
                observation_probs[k] = 0.0;
                for (int i = 0; i < N; i++) {
                    observation_probs[k] += new_state_probs[i] * B_local[i][k];
                }
            }

            // Predicted observation: argmax_k observation_probs[k]
            int predicted_observation = 0;
            double max_prob = observation_probs[0];
            for (int k = 1; k < M; k++) {
                if (observation_probs[k] > max_prob) {
                    max_prob = observation_probs[k];
                    predicted_observation = k;
                }
            }

            // Actual observation
            int actual_observation = observations[t];

            // Compute loss
            double diff = predicted_observation - actual_observation;
            loss += diff * diff;

            // Update current_state_probs
            for (int i = 0; i < N; i++) {
                current_state_probs[i] = new_state_probs[i];
            }
        }

        // Calculate average loss
        double average_loss = loss / pred_len;
        // Root the average loss
        average_loss = sqrt(average_loss);

        // Report average loss
        printf("Prediction for %d days, Average Loss = %lf\n", pred_len, average_loss);
    }

    return 0;
}

