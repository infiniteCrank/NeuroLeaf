import { Matrix } from './Matrix';

export const Activations = {
    relu: (x: number) => Math.max(0, x),
    sigmoid: (x: number) => 1 / (1 + Math.exp(-x)),
    tanh: (x: number) => Math.tanh(x),

    reluMatrix: (m: Matrix) => m.map((x: number) => Math.max(0, x)),
    sigmoidMatrix: (m: Matrix) => m.map((x: number) => 1 / (1 + Math.exp(-x))),
    tanhMatrix: (m: Matrix) => m.map((x: number) => Math.tanh(x))
};
