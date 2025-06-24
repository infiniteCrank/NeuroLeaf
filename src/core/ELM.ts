// ELM.ts - Core ELM logic with TypeScript types

import { Matrix } from './Matrix';
import { Activations } from './Activations';
import { ELMConfig, defaultConfig } from './ELMConfig';
import { UniversalEncoder } from '../preprocessing/UniversalEncoder';
import { Augment } from '../utils/Augment';

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
    public charSet: string;
    public useTokenizer: boolean;
    public tokenizerDelimiter?: RegExp;
    public encoder: UniversalEncoder;
    private model: ELMModel | null;
    public metrics?: {
        rmse?: number;
        mae?: number;
        accuracy?: number;
    };
    public verbose?: boolean;
    public savedModelJSON?: string;
    public config: ELMConfig;

    constructor(config: ELMConfig & { charSet?: string; useTokenizer?: boolean; tokenizerDelimiter?: RegExp }) {
        const cfg = { ...defaultConfig, ...config };
        this.categories = cfg.categories;
        this.hiddenUnits = cfg.hiddenUnits;
        this.maxLen = cfg.maxLen;
        this.activation = cfg.activation;
        this.charSet = cfg.charSet ?? 'abcdefghijklmnopqrstuvwxyz';
        this.useTokenizer = cfg.useTokenizer ?? false;
        this.tokenizerDelimiter = cfg.tokenizerDelimiter;
        this.config = cfg;

        this.encoder = new UniversalEncoder({
            charSet: this.charSet,
            maxLen: this.maxLen,
            useTokenizer: this.useTokenizer,
            tokenizerDelimiter: this.tokenizerDelimiter,
            mode: this.useTokenizer ? 'token' : 'char'
        });

        this.model = null;
    }

    public oneHot(n: number, index: number): number[] {
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

    public setCategories(categories: string[]) {
        this.categories = categories;
    }

    public loadModelFromJSON(json: string): void {
        try {
            const parsed: ELMModel = JSON.parse(json);
            this.model = parsed;
            this.savedModelJSON = json;
            if (this.verbose) console.log("✅ Model loaded from JSON");
        } catch (e) {
            console.error("❌ Failed to load model from JSON:", e);
        }
    }

    public train(augmentationOptions?: {
        suffixes?: string[];
        prefixes?: string[];
        includeNoise?: boolean;
    }): void {
        const X: number[][] = [], Y: number[][] = [];

        this.categories.forEach((cat, i) => {
            const variants = Augment.generateVariants(cat, this.charSet, augmentationOptions);
            for (const variant of variants) {
                const vec = this.encoder.normalize(this.encoder.encode(variant));
                X.push(vec);
                Y.push(this.oneHot(this.categories.length, i));
            }
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

        // --- Evaluation and Conditional Save ---
        const predictions = Matrix.multiply(H, beta);
        const results: Record<string, number> = {};
        let allPassed = true;

        if (this.metrics) {
            if (this.metrics.rmse !== undefined) {
                const rmse = this.calculateRMSE(Y, predictions);
                results.rmse = rmse;
                if (rmse > this.metrics.rmse) allPassed = false;
            }

            if (this.metrics.mae !== undefined) {
                const mae = this.calculateMAE(Y, predictions);
                results.mae = mae;
                if (mae > this.metrics.mae) allPassed = false;
            }

            if (this.metrics.accuracy !== undefined) {
                const acc = this.calculateAccuracy(Y, predictions);
                results.accuracy = acc;
                if (acc < this.metrics.accuracy) allPassed = false;
            }

            if (this.verbose) {
                console.log("Evaluation Results:", results);
            }

            if (allPassed) {
                this.savedModelJSON = JSON.stringify(this.model);
                if (this.verbose) console.log("✅ Model saved: All metric thresholds met.");

                if (this.config.exportFileName) {
                    this.saveModelAsJSONFile(this.config.exportFileName);
                }
            } else {
                if (this.verbose) console.log("❌ Model not saved: One or more thresholds not met.");
            }
        } else {
            throw new Error("No metrics defined in config. Please specify at least one metric to evaluate.");
        }
    }

    public saveModelAsJSONFile(filename: string = "elm_model.json"): void {
        if (!this.savedModelJSON) {
            if (this.verbose) console.warn("No model saved — did not meet metric thresholds.");
            return;
        }

        const blob = new Blob([this.savedModelJSON], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        if (this.verbose) console.log(`📦 Model exported as ${filename}`);
    }

    public predict(text: string, topK: number = 5): PredictResult[] {
        if (!this.model) throw new Error("Model not trained.");

        const vec = this.encoder.normalize(this.encoder.encode(text));
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

    public predictFromVector(inputVec: number[][], topK: number = 5): PredictResult[][] {
        if (!this.model) throw new Error("Model not trained.");

        const { W, b, beta } = this.model;
        const tempH = Matrix.multiply(inputVec, Matrix.transpose(W));
        const activationFn = Activations.get(this.activation);
        const H = Activations.apply(tempH.map(row =>
            row.map((val, j) => val + b[j][0])
        ), activationFn);

        return Matrix.multiply(H, beta).map(rawOutput => {
            const probs = Activations.softmax(rawOutput);
            return probs
                .map((p, i) => ({ label: this.categories[i], prob: p }))
                .sort((a, b) => b.prob - a.prob)
                .slice(0, topK);
        });
    }

    private calculateRMSE(Y: number[][], P: number[][]): number {
        const N = Y.length;
        let sum = 0;
        for (let i = 0; i < N; i++) {
            for (let j = 0; j < Y[0].length; j++) {
                const diff = Y[i][j] - P[i][j];
                sum += diff * diff;
            }
        }
        return Math.sqrt(sum / (N * Y[0].length));
    }

    private calculateMAE(Y: number[][], P: number[][]): number {
        const N = Y.length;
        let sum = 0;
        for (let i = 0; i < N; i++) {
            for (let j = 0; j < Y[0].length; j++) {
                sum += Math.abs(Y[i][j] - P[i][j]);
            }
        }
        return sum / (N * Y[0].length);
    }

    private calculateAccuracy(Y: number[][], P: number[][]): number {
        let correct = 0;
        for (let i = 0; i < Y.length; i++) {
            const yMax = Y[i].indexOf(Math.max(...Y[i]));
            const pMax = P[i].indexOf(Math.max(...P[i]));
            if (yMax === pMax) correct++;
        }
        return correct / Y.length;
    }
} 
