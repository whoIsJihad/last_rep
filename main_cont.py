import os
import numpy as np
import matplotlib.pyplot as plt


# Ensure the output directory exists
output_dir = "cont_plots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

class ContinuousSignal:
    def __init__(self, func):
        self.func = func

    def get_value_at_time(self, time):
        return self.func(time)

    def add(self, other):
        def add_func(x):
            return self.func(x) + other.func(x)
        return ContinuousSignal(add_func)

    def shift_signal(self, shift):
        def shifted_func(x):
            return self.func(x - shift)
        return ContinuousSignal(shifted_func)

    def multiply(self, other):
        def multiplied_func(x):
            return self.func(x) * other.func(x)
        return ContinuousSignal(multiplied_func)

    def multiply_const_factor(self, scaler):
        def scaled_func(x):
            return self.func(x) * scaler
        return ContinuousSignal(scaled_func)

    def plot(self, start, end, label):
        x = np.linspace(start, end, 1000)
        y = self.func(x)
        plt.plot(x, y, label=label)
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.grid(True)

    @staticmethod
    def show(save_path):
        plt.legend()
        plt.savefig(save_path)
        plt.clf()  # Clear figure for the next plot

def u(t):
    return np.where(t >= 0, 1, 0)

def y(t):
    return (1 - np.exp(-t)) * u(t)

class LTI_Continuous:
    def __init__(self, impulse_response):
        self.impulse_response = impulse_response
        self.INF = 3

    def linear_combination_of_impulses(self, input_signal, delta):
        decomposed = []
        components = []
        shifts = []
        result = ContinuousSignal(lambda x: 0)  # initialize with zero function

        for i in np.arange(-self.INF, self.INF, delta):
            coefficient = input_signal.get_value_at_time(i)
            
            def func2(t):
                return np.where((t >= 0) & (t <= delta), 1, 0)
            
            impulse = ContinuousSignal(func2)
            shiftedImpulse = impulse.shift_signal(i)
            components.append(shiftedImpulse.multiply_const_factor(coefficient))
            shifts.append(i)
            decomposed.append((shiftedImpulse, coefficient, i))
            result = result.add(shiftedImpulse.multiply_const_factor(coefficient))
        
        self.plot_components_subplots(components, shifts, result, f"{output_dir}/decomposed.png", delta)
        return decomposed

    def plot_components_subplots(self, components, shifts, final_output, fileName, delta):
        num_components = len(components)
        total_plots = num_components + 1
        num_rows = (total_plots + 2) // 3
        fig, axs = plt.subplots(num_rows, 3, figsize=(16, 48))
        axs = axs.flatten()

        for idx, (component, time_shift) in enumerate(zip(components, shifts)):
            time_points = np.linspace(-self.INF, self.INF, 1000)
            y = component.func(time_points)
            axs[idx].plot(time_points, y, label=r'$\delta(t - (%.1f)) \times x(%.1f)$' % (time_shift, time_shift))
            axs[idx].set_xlabel('t (Time Index)')
            axs[idx].set_ylabel('x(t)')
            axs[idx].set_ylim(0, 1)
            axs[idx].legend()
            axs[idx].grid(True)

        time_points = np.linspace(-self.INF, self.INF, 1000)
        y_final = final_output.func(time_points)
        axs[num_components].plot(time_points, y_final, label='Reconstructed Signal')
        axs[num_components].set_title('Reconstructed Signal')
        axs[num_components].set_xlabel('t (Time)')
        axs[num_components].set_ylabel('Amplitude')
        axs[num_components].legend()
        axs[num_components].grid(True)

        for j in range(num_components + 1, len(axs)):
            axs[j].axis('off')

        plt.tight_layout()
        plt.savefig(fileName)
        plt.clf()

    def output_approx(self, signal, delta):
        result = ContinuousSignal(lambda x: 0)
        decomposed = self.linear_combination_of_impulses(signal, delta)
        
        components = []
        shifts = []
        
        for decomposed_signal, coefficient, i in decomposed:
            response = ContinuousSignal(lambda t: np.where(t >= 0, delta, 0))
            shifted_response = response.shift_signal(i)
            component = shifted_response.multiply_const_factor(coefficient)
            components.append(component)
            shifts.append(i)
            result = result.add(component)
        
        self.plot_components_subplots(components, shifts, result, f"{output_dir}/output_approx.png", delta)
        return result

def plot_varying_delta(output_signals, actual_signal, deltas, time_range):
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    axs = axs.flatten()
    time_points = np.linspace(time_range[0], time_range[1], 1000)

    for idx, (output_signal, delta) in enumerate(zip(output_signals, deltas)):
        y_approx = output_signal.func(time_points)
        y_actual = actual_signal.func(time_points)
        axs[idx].plot(time_points, y_approx, label=r'$y_{approx}(t)$', color='blue')
        axs[idx].plot(time_points, y_actual, label=r'$y(t) = (1 - e^{-t})u(t)$', color='orange')
        axs[idx].set_xlabel('t (Time)')
        axs[idx].set_ylabel('x(t)')
        axs[idx].legend()
        axs[idx].grid(True)
        axs[idx].set_title(r'$\Delta = %.2f$' % delta)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/approximate_output_varying_delta.png")
    plt.clf()

def main():
    input_signal = ContinuousSignal(lambda x: np.where(x > 0, np.exp(-x), 0))
    y_actual = ContinuousSignal(y)
    
    y_actual.plot(-3, 3, 'actual signal')
    ContinuousSignal.show(f"{output_dir}/actual_signal.png")
    plt.figure()

    input_signal.plot(-3, 3, "Input Signal")
    ContinuousSignal.show(f"{output_dir}/input_signal.png")
    plt.figure()

    impulse_response = ContinuousSignal(u)
    LTI_system = LTI_Continuous(impulse_response)
    deltas = [0.5, 0.1]

    output_signals = [LTI_system.output_approx(input_signal, delta) for delta in deltas]
    plot_varying_delta(output_signals, y_actual, deltas, time_range=(-3, 3))

main()
