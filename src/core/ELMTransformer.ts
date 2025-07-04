// ELMTransformer.ts
import { ELM, PredictResult } from "./ELM";
import { ELMConfig } from "./ELMConfig"
import { UniversalEncoder } from "../preprocessing/UniversalEncoder";

export enum ELMTransformerMode {
    ELM_TO_TRANSFORMER = "ELM_TO_TRANSFORMER",
    TRANSFORMER_TO_ELM = "TRANSFORMER_TO_ELM",
    PARAMETERIZE_ELM = "PARAMETERIZE_ELM",
    ENSEMBLE = "ENSEMBLE",
}

export interface ELMTransformerConfig {
    mode: ELMTransformerMode;
    elmConfig: ELMConfig;
    embedDim: number;
    seqLen: number;
    numHeads: number;
    numLayers: number;
    learningRate?: number;
    dropout?: number;
}

export class ELMTransformer {
    private elm: ELM;
    private encoder: UniversalEncoder;
    private config: ELMTransformerConfig;

    private vocab: string[] = [];
    private wordToIdx: Record<string, number> = {};
    private idxToWord: Record<number, string> = {};
    private embedding: number[][] = [];
    private posEnc: number[][] = [];

    constructor(config: ELMTransformerConfig) {
        this.config = config;
        this.elm = new ELM(config.elmConfig);
        this.encoder = new UniversalEncoder({
            maxLen: config.seqLen,
            charSet: config.elmConfig.charSet,
            mode: "char",
        });
        this.posEnc = this.positionalEncoding(config.seqLen, config.embedDim);
    }

    /**
     * Positional encoding like in Vaswani et al. (2017).
     */
    private positionalEncoding(seqLen: number, embedDim: number): number[][] {
        const encoding: number[][] = [];
        for (let pos = 0; pos < seqLen; pos++) {
            const row: number[] = [];
            for (let i = 0; i < embedDim; i++) {
                const angle = pos / Math.pow(10000, (2 * Math.floor(i / 2)) / embedDim);
                row.push(i % 2 === 0 ? Math.sin(angle) : Math.cos(angle));
            }
            encoding.push(row);
        }
        return encoding;
    }

    /**
     * Softmax over a vector.
     */
    private softmax(x: number[]): number[] {
        const max = Math.max(...x);
        const exps = x.map(v => Math.exp(v - max));
        const sum = exps.reduce((a, b) => a + b, 0);
        return exps.map(e => e / sum);
    }

    /**
     * Dot product of two vectors.
     */
    private dot(a: number[], b: number[]): number {
        return a.reduce((sum, ai, i) => sum + ai * b[i], 0);
    }

    /**
     * Multiply matrix and vector.
     */
    private matVecMul(mat: number[][], vec: number[]): number[] {
        return mat.map(row => this.dot(row, vec));
    }

    /**
     * Multiply two matrices.
     */
    private matMatMul(a: number[][], b: number[][]): number[][] {
        const result: number[][] = [];
        for (let i = 0; i < a.length; i++) {
            const row: number[] = [];
            for (let j = 0; j < b[0].length; j++) {
                let sum = 0;
                for (let k = 0; k < a[0].length; k++) {
                    sum += a[i][k] * b[k][j];
                }
                row.push(sum);
            }
            result.push(row);
        }
        return result;
    }

    /**
     * Transpose a matrix.
     */
    private transpose(mat: number[][]): number[][] {
        return mat[0].map((_, i) => mat.map(row => row[i]));
    }

    /**
     * Layer normalization (mean 0, variance 1).
     */
    private layerNorm(x: number[]): number[] {
        const mean = x.reduce((a, b) => a + b, 0) / x.length;
        const variance = x.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / x.length;
        const std = Math.sqrt(variance + 1e-5);
        return x.map(v => (v - mean) / std);
    }

    /**
     * Add two vectors.
     */
    private addVec(a: number[], b: number[]): number[] {
        return a.map((v, i) => v + b[i]);
    }

    /**
     * Flatten matrix rows into one vector.
     */
    private flatten(mat: number[][]): number[] {
        return mat.reduce((acc, row) => acc.concat(row), []);
    }

    /**
     * Create random matrix.
     */
    private randMatrix(rows: number, cols: number): number[][] {
        return Array.from({ length: rows }, () =>
            Array.from({ length: cols }, () => Math.random() * 2 - 1)
        );
    }

    /**
 * Initialize embedding matrix with random vectors.
 */
    public initEmbedding(vocabSize: number): void {
        this.embedding = this.randMatrix(vocabSize, this.config.embedDim);
    }

    /**
     * Project input matrix through a weight matrix.
     */
    private project(x: number[][], W: number[][]): number[][] {
        return x.map(row => this.matVecMul(W, row));
    }

    /**
     * Compute scaled dot-product attention (with causal mask).
     */
    private scaledDotProduct(Q: number[][], K: number[][]): number[][] {
        const n = Q.length;
        const d = Q[0].length;
        const scores: number[][] = [];

        for (let i = 0; i < n; i++) {
            const row: number[] = [];
            for (let j = 0; j < n; j++) {
                let s = this.dot(Q[i], K[j]) / Math.sqrt(d);
                if (j > i) {
                    s = -1e9; // causal mask
                }
                row.push(s);
            }
            scores.push(row);
        }
        return scores;
    }

    /**
     * Softmax over each row of a 2D matrix.
     */
    private softmax2D(x: number[][]): number[][] {
        return x.map(row => this.softmax(row));
    }

    /**
     * Multi-head attention forward pass.
     */
    private multiHeadAttention(
        x: number[][],
        Wq: number[][][],
        Wk: number[][][],
        Wv: number[][][],
        Wout: number[][]
    ): number[][] {
        const heads: number[][][] = [];

        for (let h = 0; h < this.config.numHeads; h++) {
            const Q = this.project(x, Wq[h]);
            const K = this.project(x, Wk[h]);
            const V = this.project(x, Wv[h]);

            const scores = this.scaledDotProduct(Q, K);
            const weights = this.softmax2D(scores);
            const attention = this.matMatMul(weights, V);
            heads.push(attention);
        }

        // Concatenate heads
        const concat: number[][] = [];
        for (let i = 0; i < x.length; i++) {
            let row: number[] = [];
            for (let h = 0; h < this.config.numHeads; h++) {
                row = row.concat(heads[h][i]);
            }
            concat.push(row);
        }

        // Final output projection
        return this.project(concat, Wout);
    }

}
