// ELM.ts - Core ELM logic with TypeScript types

import { Matrix } from './Matrix';
import { Activations } from './Activations';
import { ELMConfig, defaultConfig } from './ELMConfig';
import { TextEncoder } from '../preprocessing/TextEncoder';

export interface ELMModel {
    W: number[][];
    b: number[][];
    beta: number[][];
}

export interface PredictResult {
    label: string;
    prob: number;
}

export class ELM {
    public categories: string[];
    public hiddenUnits: number;
    public maxLen: number;
    public activation: string;
    private encoder: TextEncoder;
    private model: ELMModel | null;

    constructor(config: ELMConfig) {
        const cfg = { ...defaultConfig, ...config };
        this.categories = cfg.categories;
        this.hiddenUnits = cfg.hiddenUnits;
        this.maxLen = cfg.maxLen;
        this.activation = cfg.activation;
        this.encoder = new TextEncoder({ maxLen: this.maxLen });
        this.model = null;
    }

    private oneHot(n: number, index: number): number[] {
        return Array.from({ length: n }, (_, i) => (i === index ? 1 : 0));
    }

    private pseudoInverse(H: number[][], lambda: number = 1e-3): number[][] {
        const Ht = Matrix.transpose(H);
        const HtH = Matrix.multiply(Ht, H);
        const HtH_reg = Matrix.addRegularization(HtH, lambda);
        const HtH_inv = Matrix.inverse(HtH_reg);
        return Matrix.multiply(HtH_inv, Ht);
    }

    private randomMatrix(rows: number, cols: number): number[][] {
        return Array.from({ length: rows }, () =>
            Array.from({ length: cols }, () => Math.random() * 2 - 1)
        );
    }

    public train(): void {
        const X: number[][] = [], Y: number[][] = [];
        this.categories.forEach((cat, i) => {
            const baseVec = this.encoder.normalizeVector(this.encoder.textToVector(cat));
            X.push(baseVec);
            Y.push(this.oneHot(this.categories.length, i));

            const augVec = this.encoder.normalizeVector(this.encoder.textToVector(cat + ' music'));
            X.push(augVec);
            Y.push(this.oneHot(this.categories.length, i));
        });

        const W = this.randomMatrix(this.hiddenUnits, X[0].length);
        const b = this.randomMatrix(this.hiddenUnits, 1);
        const tempH = Matrix.multiply(X, Matrix.transpose(W));
        const activationFn = Activations.get(this.activation);
        const H = Activations.apply(tempH.map(row =>
            row.map((val, j) => val + b[j][0])
        ), activationFn);

        const H_pinv = this.pseudoInverse(H);
        const beta = Matrix.multiply(H_pinv, Y);
        this.model = { W, b, beta };
    }

    public predict(text: string, topK: number = 5): PredictResult[] {
        if (!this.model) throw new Error("Model not trained.");

        const vec = this.encoder.normalizeVector(this.encoder.textToVector(text));
        const { W, b, beta } = this.model;
        const tempH = Matrix.multiply([vec], Matrix.transpose(W));
        const activationFn = Activations.get(this.activation);
        const H = Activations.apply(tempH.map(row =>
            row.map((val, j) => val + b[j][0])
        ), activationFn);

        const rawOutput = Matrix.multiply(H, beta)[0];
        const probs = Activations.softmax(rawOutput);

        return probs
            .map((p, i) => ({ label: this.categories[i], prob: p }))
            .sort((a, b) => b.prob - a.prob)
            .slice(0, topK);
    }
}
