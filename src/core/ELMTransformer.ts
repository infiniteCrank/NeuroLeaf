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

    /**
 * Feedforward network (2 layers with ReLU).
 */
    private feedForward(
        x: number[],
        W1: number[][],
        b1: number[],
        W2: number[][],
        b2: number[],
        training: boolean
    ): number[] {
        // Layer 1
        const hidden = this.matVecMul(W1, x).map((v, i) => Math.max(0, v + b1[i]));
        // Dropout
        const dropped = hidden.map(h => {
            if (training && this.config.dropout && Math.random() < this.config.dropout) return 0;
            return h;
        });
        // Layer 2
        return this.addVec(this.matVecMul(W2, dropped), b2);
    }

    /**
     * Transformer block with residual connections.
     */
    private transformerBlock(
        x: number[][],
        Wq: number[][][],
        Wk: number[][][],
        Wv: number[][][],
        Wout: number[][],
        Wff1: number[][],
        bff1: number[],
        Wff2: number[][],
        bff2: number[],
        training: boolean
    ): number[][] {
        // Multi-head attention
        const attnOut = this.multiHeadAttention(x, Wq, Wk, Wv, Wout);
        // Add & norm
        const added1 = x.map((row, i) => this.addVec(row, attnOut[i]));
        const norm1 = added1.map(this.layerNorm);

        // Feedforward
        const ffOut = norm1.map(row =>
            this.feedForward(row, Wff1, bff1, Wff2, bff2, training)
        );

        // Add & norm
        const added2 = norm1.map((row, i) => this.addVec(row, ffOut[i]));
        const norm2 = added2.map(this.layerNorm);

        return norm2;
    }
    /**
     * Encodes raw text into numeric sequence vectors.
     */
    private encodeInput(text: string): number[][] {
        const vec = this.encoder.encode(text);
        const split: number[][] = [];
        const dim = this.config.embedDim / this.config.numHeads;
        for (let i = 0; i < this.config.seqLen; i++) {
            // Split into smaller chunks per position
            const start = i * dim;
            const end = start + dim;
            split.push(vec.slice(start, end));
        }
        return split;
    }

    /**
     * Transformer forward pass for a single input.
     */
    private transformerEncode(
        inputSeq: number[][],
        weights: {
            Wq: number[][][],
            Wk: number[][][],
            Wv: number[][][],
            Wout: number[][],
            Wff1: number[][],
            bff1: number[],
            Wff2: number[][],
            bff2: number[]
        },
        training: boolean = false
    ): number[] {
        let x = inputSeq.map((row, i) => this.addVec(row, this.posEnc[i]));
        for (let l = 0; l < this.config.numLayers; l++) {
            x = this.transformerBlock(
                x,
                weights.Wq,
                weights.Wk,
                weights.Wv,
                weights.Wout,
                weights.Wff1,
                weights.bff1,
                weights.Wff2,
                weights.bff2,
                training
            );
        }
        return this.flatten(x);
    }

    /**
     * Predict function combining ELM and Transformer according to mode.
     */
    public predict(text: string, topK = 3): PredictResult[] {
        // Encode
        const inputVec = this.encoder.encode(text);
        const inputSeq = this.encodeInput(text);

        switch (this.config.mode) {
            case ELMTransformerMode.ELM_TO_TRANSFORMER:
                // 1) ELM embedding
                const elmEmbedding = this.elm.getEmbedding([inputVec])[0];
                // 2) Reshape to sequence
                const reshaped: number[][] = [];
                const chunk = Math.floor(elmEmbedding.length / this.config.seqLen);
                for (let i = 0; i < this.config.seqLen; i++) {
                    reshaped.push(elmEmbedding.slice(i * chunk, (i + 1) * chunk));
                }
                // 3) Transformer encode
                const transformerVecA = this.transformerEncode(reshaped, this.initTransformerWeights());
                // 4) Softmax over categories
                return this.vectorToPrediction(transformerVecA, topK);

            case ELMTransformerMode.TRANSFORMER_TO_ELM:
                // 1) Transformer encode
                const transformerVecB = this.transformerEncode(inputSeq, this.initTransformerWeights());
                // 2) ELM prediction
                return this.elm.predictFromVector([transformerVecB], topK)[0];

            case ELMTransformerMode.PARAMETERIZE_ELM:
                // Experimental: transform output modulates ELM weights
                console.warn("[⚠️] PARAMETERIZE_ELM mode is experimental and may not work as expected.");
                const tVec = this.transformerEncode(inputSeq, this.initTransformerWeights());
                const modulated = inputVec.map((v, i) => v * (1 + 0.01 * (tVec[i % tVec.length] || 0)));
                return this.elm.predictFromVector([modulated], topK)[0];

            case ELMTransformerMode.ENSEMBLE:
                // 1) ELM
                const elmPred = this.elm.predict(text, topK);
                // 2) Transformer
                const tVecEnsemble = this.transformerEncode(inputSeq, this.initTransformerWeights());
                const transformerPred = this.vectorToPrediction(tVecEnsemble, topK);
                // 3) Average probabilities
                return elmPred.map((e, i) => ({
                    label: e.label,
                    prob: (e.prob + (transformerPred[i]?.prob || 0)) / 2
                }));

            default:
                throw new Error(`Unknown mode: ${this.config.mode}`);
        }
    }

    /**
     * Converts a vector into predictions over categories.
     */
    private vectorToPrediction(vec: number[], topK: number): PredictResult[] {
        const probs = this.softmax(vec);
        return probs
            .map((p, i) => ({
                label: this.elm.categories[i] || `class_${i}`,
                prob: p
            }))
            .sort((a, b) => b.prob - a.prob)
            .slice(0, topK);
    }

    /**
     * Dummy transformer weights initializer.
     * In a real setup you would want to store and train these.
     */
    private initTransformerWeights() {
        const d = this.config.embedDim;
        return {
            Wq: Array.from({ length: this.config.numHeads }, () => this.randMatrix(d, d / this.config.numHeads)),
            Wk: Array.from({ length: this.config.numHeads }, () => this.randMatrix(d, d / this.config.numHeads)),
            Wv: Array.from({ length: this.config.numHeads }, () => this.randMatrix(d, d / this.config.numHeads)),
            Wout: this.randMatrix(d, d),
            Wff1: this.randMatrix(d, d),
            bff1: Array(d).fill(0),
            Wff2: this.randMatrix(d, d),
            bff2: Array(d).fill(0),
        };
    }

    /**
 * Train the ELM on labeled data.
 * For now, this trains only the ELM part.
 */
    public train(
        trainPairs: { input: string; label: string }[],
        augmentationOptions?: {
            suffixes?: string[];
            prefixes?: string[];
            includeNoise?: boolean;
        }
    ): void {
        // This uses the ELM's built-in training pipeline.
        this.elm.setCategories(
            Array.from(new Set(trainPairs.map(p => p.label)))
        );

        // Auto-generate training matrix
        const X: number[][] = [];
        const Y: number[][] = [];
        for (const { input, label } of trainPairs) {
            const vec = this.encoder.normalize(this.encoder.encode(input));
            const labelIndex = this.elm.categories.indexOf(label);
            X.push(vec);
            Y.push(this.elm.oneHot(this.elm.categories.length, labelIndex));
        }

        this.elm.trainFromData(X, Y);

        console.info("✅ ELM training complete.");
    }

    /**
     * Get an embedding for text.
     * Depending on mode, returns ELM or Transformer embeddings.
     */
    public getEmbedding(text: string): number[] {
        const inputVec = this.encoder.encode(text);
        const inputSeq = this.encodeInput(text);

        switch (this.config.mode) {
            case ELMTransformerMode.ELM_TO_TRANSFORMER:
                const elmEmbedding = this.elm.getEmbedding([inputVec])[0];
                const reshaped: number[][] = [];
                const chunk = Math.floor(elmEmbedding.length / this.config.seqLen);
                for (let i = 0; i < this.config.seqLen; i++) {
                    reshaped.push(elmEmbedding.slice(i * chunk, (i + 1) * chunk));
                }
                return this.transformerEncode(reshaped, this.initTransformerWeights(), false);

            case ELMTransformerMode.TRANSFORMER_TO_ELM:
            case ELMTransformerMode.PARAMETERIZE_ELM:
            case ELMTransformerMode.ENSEMBLE:
                return this.transformerEncode(inputSeq, this.initTransformerWeights(), false);

            default:
                throw new Error(`Unknown mode: ${this.config.mode}`);
        }
    }

    /**
     * Save the ELM model.
     */
    public saveModelAsJSONFile(filename?: string): void {
        this.elm.saveModelAsJSONFile(filename);
    }

    /**
     * Load the ELM model.
     */
    public loadModelFromJSON(json: string): void {
        this.elm.loadModelFromJSON(json);
    }


}
