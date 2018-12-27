class KalmanObject:
    def __init__(self):
        self.A = 5
        self.R = .1
        self.x_k = 0 * self.A
        self.p_k = 1 * self.A

    # Prediction Function
    def predict_and_update(self, observe_value):
        self.K_k = self.p_k / (self.p_k + self.R)

        x_k_hat = self.x_k + self.K_k * (observe_value - self.x_k)

        # Override x_k with x_k_hat
        self.x_k = self.A * x_k_hat

        new_p_k = (1 - self.K_k) * self.p_k

        # Override p_k with new_p_k
        self.p_k = self.A * new_p_k * self.A

        print(f'p_k: {self.p_k}')

        return x_k_hat
