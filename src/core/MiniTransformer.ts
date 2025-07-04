// mini_transformer.ts
import * as fs from "fs";
import * as path from "path";

// Helper: random normal
function randn(): number {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

// === Hyperparameters ===
const embedDim = 64;
const seqLen = 16;
const numHeads = 2;
const headDim = embedDim / numHeads;
const learningRate = 0.01;
const epochs = 10;

// === Vocabulary ===
export class Vocab {
    tokens: string[];
    tokenToIdx: Map<string, number>;
    idxToToken: Map<number, string>;

    constructor(tokens: string[]) {
        this.tokens = tokens;
        this.tokenToIdx = new Map(tokens.map((t, i) => [t, i]));
        this.idxToToken = new Map(tokens.map((t, i) => [i, t]));
    }

    encode(text: string): number[] {
        return text.split("").map(c => this.tokenToIdx.get(c) ?? 0);
    }

    decode(indices: number[]): string {
        return indices.map(i => this.idxToToken.get(i) ?? "?").join("");
    }
}

// === Utility functions ===
function randMatrix(rows: number, cols: number): number[][] {
    return Array.from({ length: rows }, () =>
        Array.from({ length: cols }, () => randn())
    );
}

function dot(a: number[], b: number[]): number {
    return a.reduce((sum, ai, i) => sum + ai * b[i], 0);
}

function matVecMul(mat: number[][], vec: number[]): number[] {
    return mat.map(row => dot(row, vec));
}

function softmax(x: number[]): number[] {
    const max = Math.max(...x);
    const exps = x.map(v => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(e => e / sum);
}

function positionalEncoding(seqLen: number, embedDim: number): number[][] {
    const enc: number[][] = [];
    for (let pos = 0; pos < seqLen; pos++) {
        const row: number[] = [];
        for (let i = 0; i < embedDim; i++) {
            const angle = pos / Math.pow(10000, (2 * Math.floor(i / 2)) / embedDim);
            row.push(i % 2 === 0 ? Math.sin(angle) : Math.cos(angle));
        }
        enc.push(row);
    }
    return enc;
}

// === Transformer Layer ===
class TransformerLayer {
    Wq: number[][] = randMatrix(embedDim, headDim);
    Wk: number[][] = randMatrix(embedDim, headDim);
    Wv: number[][] = randMatrix(embedDim, headDim);
    Wout: number[][] = randMatrix(embedDim, embedDim);
}

// === MiniTransformer ===
export class MiniTransformer {
    vocab: Vocab;
    embedding: number[][];
    layers: TransformerLayer[];
    Wo: number[][];
    bOut: number[];
    posEnc: number[][];

    constructor(vocab: Vocab) {
        this.vocab = vocab;
        this.embedding = randMatrix(vocab.tokens.length, embedDim);
        this.layers = [new TransformerLayer(), new TransformerLayer()];
        this.Wo = randMatrix(vocab.tokens.length, embedDim);
        this.bOut = Array(vocab.tokens.length).fill(0);
        this.posEnc = positionalEncoding(seqLen, embedDim);
    }

    // Project input sequence to contextual embedding
    encode(inputIndices: number[]): number[] {
        let x: number[][] = inputIndices.map((idx, i) =>
            this.embedding[idx].map((v, j) => v + this.posEnc[i][j])
        );

        for (const layer of this.layers) {
            x = this.multiHeadAttention(x, layer);
        }

        // Mean pooling
        const mean: number[] = Array(embedDim).fill(0);
        for (const row of x) {
            row.forEach((v, j) => (mean[j] += v));
        }
        return mean.map(v => v / x.length);
    }

    // Multi-head attention
    private multiHeadAttention(x: number[][], layer: TransformerLayer): number[][] {
        const Q = x.map(v => matVecMul(layer.Wq, v));
        const K = x.map(v => matVecMul(layer.Wk, v));
        const V = x.map(v => matVecMul(layer.Wv, v));

        // Compute attention scores
        const scores: number[][] = Q.map((q, i) =>
            K.map(k => dot(q, k) / Math.sqrt(headDim))
        );

        // Softmax over keys
        const weights: number[][] = scores.map(softmax);

        // Weighted sum of V
        const attended = weights.map((row, i) => {
            const out = Array(headDim).fill(0);
            for (let j = 0; j < V.length; j++) {
                for (let k = 0; k < headDim; k++) {
                    out[k] += row[j] * V[j][k];
                }
            }
            return out;
        });

        // Project back to embedDim
        return attended.map(vec => matVecMul(layer.Wout, vec));
    }

    // Predict next token probabilities
    predictNext(inputIndices: number[]): number[] {
        const context = this.encode(inputIndices);
        const logits = this.Wo.map((row, i) => dot(row, context) + this.bOut[i]);
        return softmax(logits);
    }

    // Generate text
    generate(start: string[], tokens: number): string {
        const seq = [...start];
        for (let i = 0; i < tokens; i++) {
            const indices = seq.slice(-seqLen).map(s => this.vocab.tokenToIdx.get(s) ?? 0);
            const probs = this.predictNext(indices);
            const nextIdx = this.sample(probs);
            seq.push(this.vocab.idxToToken.get(nextIdx) ?? "?");
        }
        return seq.join("");
    }

    private sample(probs: number[]): number {
        const r = Math.random();
        let cum = 0;
        for (let i = 0; i < probs.length; i++) {
            cum += probs[i];
            if (r < cum) return i;
        }
        return probs.length - 1;
    }
}
