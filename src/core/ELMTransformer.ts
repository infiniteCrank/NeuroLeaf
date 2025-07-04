// ELMTransformer.ts
import { ELM, PredictResult } from "./ELM";
import { ELMConfig } from "./ELMConfig";
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
    private posEnc: number[][];
    private transformerWeights: ReturnType<ELMTransformer["initTransformerWeights"]>;
    private embedding: number[][] = [];

    constructor(config: ELMTransformerConfig) {
        this.config = config;
        this.elm = new ELM(config.elmConfig);
        this.encoder = new UniversalEncoder({
            maxLen: config.seqLen,
            charSet: config.elmConfig.charSet ?? "abcdefghijklmnopqrstuvwxyz",
            mode: "char",
        });
        this.posEnc = this.positionalEncoding(config.seqLen, config.embedDim);
        this.transformerWeights = this.initTransformerWeights();
    }

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

    private softmax(x: number[]): number[] {
        const max = Math.max(...x);
        const exps = x.map(v => Math.exp(v - max));
        const sum = exps.reduce((a, b) => a + b, 0);
        return exps.map(e => e / sum);
    }

    private dot(a: number[], b: number[]): number {
        return a.reduce((sum, ai, i) => sum + ai * b[i], 0);
    }

    private matVecMul(mat: number[][], vec: number[]): number[] {
        return mat.map(row => this.dot(row, vec));
    }

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

    private transpose(mat: number[][]): number[][] {
        return mat[0].map((_, i) => mat.map(row => row[i]));
    }

    private layerNorm(x: number[]): number[] {
        const mean = x.reduce((a, b) => a + b, 0) / x.length;
        const variance = x.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / x.length;
        const std = Math.sqrt(variance + 1e-5);
        return x.map(v => (v - mean) / std);
    }

    private addVec(a: number[], b: number[]): number[] {
        return a.map((v, i) => v + b[i]);
    }

    private flatten(mat: number[][]): number[] {
        return mat.reduce((acc, row) => acc.concat(row), []);
    }

    private randMatrix(rows: number, cols: number): number[][] {
        return Array.from({ length: rows }, () =>
            Array.from({ length: cols }, () => Math.random() * 2 - 1)
        );
    }

    public initEmbedding(vocabSize: number): void {
        this.embedding = this.randMatrix(vocabSize, this.config.embedDim);
    }

    private project(x: number[][], W: number[][]): number[][] {
        return x.map(row => this.matVecMul(W, row));
    }

    private scaledDotProduct(Q: number[][], K: number[][]): number[][] {
        const n = Q.length;
        const d = Q[0].length;
        const scores: number[][] = [];
        for (let i = 0; i < n; i++) {
            const row: number[] = [];
            for (let j = 0; j < n; j++) {
                let s = this.dot(Q[i], K[j]) / Math.sqrt(d);
                if (j > i) s = -1e9; // causal mask
                row.push(s);
            }
            scores.push(row);
        }
        return scores;
    }

    private softmax2D(x: number[][]): number[][] {
        return x.map(row => this.softmax(row));
    }

    private multiHeadAttention(x: number[][], Wq: number[][][], Wk: number[][][], Wv: number[][][], Wout: number[][]): number[][] {
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

        const concat: number[][] = [];
        for (let i = 0; i < x.length; i++) {
            let row: number[] = [];
            for (let h = 0; h < this.config.numHeads; h++) {
                row = row.concat(heads[h][i]);
            }
            concat.push(row);
        }
        return this.project(concat, Wout);
    }

    private feedForward(x: number[], W1: number[][], b1: number[], W2: number[][], b2: number[], training: boolean): number[] {
        const hidden = this.matVecMul(W1, x).map((v, i) => Math.max(0, v + b1[i]));
        const dropped = hidden.map(h => {
            if (training && this.config.dropout && Math.random() < this.config.dropout) return 0;
            return h;
        });
        return this.addVec(this.matVecMul(W2, dropped), b2);
    }

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
        const attnOut = this.multiHeadAttention(x, Wq, Wk, Wv, Wout);
        const added1 = x.map((row, i) => this.addVec(row, attnOut[i]));
        const norm1 = added1.map(this.layerNorm);

        const ffOut = norm1.map(row =>
            this.feedForward(row, Wff1, bff1, Wff2, bff2, training)
        );

        const added2 = norm1.map((row, i) => this.addVec(row, ffOut[i]));
        const norm2 = added2.map(this.layerNorm);

        return norm2;
    }

    private encodeInput(text: string): number[][] {
        let vec = this.encoder.encode(text);

        // Pad or truncate to embedDim
        if (vec.length < this.config.embedDim) {
            vec = vec.concat(Array(this.config.embedDim - vec.length).fill(0));
        } else if (vec.length > this.config.embedDim) {
            vec = vec.slice(0, this.config.embedDim);
        }

        // Repeat this vector seqLen times
        return Array(this.config.seqLen).fill(vec);
    }

    private transformerEncode(
        inputSeq: number[][],
        weights: ReturnType<ELMTransformer["initTransformerWeights"]>,
        training = false
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

        const flat = this.flatten(x);

        if (flat.some(v => isNaN(v))) {
            console.error(`❌ NaN detected in transformerEncode output.`);
            console.error(`Input sequence:`);
            console.dir(inputSeq, { depth: null });
            console.error(`After positional encoding:`);
            console.dir(x, { depth: null });
            console.error(`Transformer weights:`);
            console.dir(weights, { depth: 1 });
            throw new Error(`NaN detected in transformerEncode output.`);
        }

        return flat;
    }


    public predict(text: string, topK = 3): PredictResult[] {
        const inputVec = this.encoder.encode(text);
        const inputSeq = this.encodeInput(text);

        switch (this.config.mode) {
            case ELMTransformerMode.ELM_TO_TRANSFORMER:
                const elmEmbedding = this.elm.getEmbedding([inputVec])[0];
                const reshaped: number[][] = [];
                const chunk = Math.floor(elmEmbedding.length / this.config.seqLen);
                for (let i = 0; i < this.config.seqLen; i++) {
                    let slice = elmEmbedding.slice(i * chunk, (i + 1) * chunk);

                    // Pad to embedDim
                    if (slice.length < this.config.embedDim) {
                        slice = slice.concat(Array(this.config.embedDim - slice.length).fill(0));
                    } else if (slice.length > this.config.embedDim) {
                        slice = slice.slice(0, this.config.embedDim);
                    }
                    reshaped.push(slice);
                }
                const transformerVecA = this.transformerEncode(reshaped, this.transformerWeights);
                return this.vectorToPrediction(transformerVecA, topK);

            case ELMTransformerMode.TRANSFORMER_TO_ELM:
                const transformerVecB = this.transformerEncode(inputSeq, this.transformerWeights);
                return this.elm.predictFromVector([transformerVecB], topK)[0];

            case ELMTransformerMode.PARAMETERIZE_ELM:
                console.warn("[⚠️] PARAMETERIZE_ELM mode is experimental.");
                const tVec = this.transformerEncode(inputSeq, this.transformerWeights);
                const modulated = inputVec.map((v, i) => v * (1 + 0.01 * (tVec[i % tVec.length] || 0)));
                return this.elm.predictFromVector([modulated], topK)[0];

            case ELMTransformerMode.ENSEMBLE:
                const elmPred = this.elm.predict(text, topK);
                const tVecEnsemble = this.transformerEncode(inputSeq, this.transformerWeights);
                const transformerPred = this.vectorToPrediction(tVecEnsemble, topK);
                return elmPred.map((e, i) => ({
                    label: e.label,
                    prob: (e.prob + (transformerPred[i]?.prob || 0)) / 2
                }));

            default:
                throw new Error(`Unknown mode: ${this.config.mode}`);
        }
    }

    private vectorToPrediction(vec: number[], topK: number): PredictResult[] {
        if (vec.some(v => isNaN(v))) throw new Error(`NaN detected in vectorToPrediction input vector.`);
        const probs = this.softmax(vec);
        return probs
            .map((p, i) => ({
                label: this.elm.categories[i] || `class_${i}`,
                prob: p
            }))
            .sort((a, b) => b.prob - a.prob)
            .slice(0, topK);
    }

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

    public train(trainPairs: { input: string; label: string }[]): void {
        this.elm.setCategories(Array.from(new Set(trainPairs.map(p => p.label))));
        const X = trainPairs.map(p => this.encoder.normalize(this.encoder.encode(p.input)));
        const Y = trainPairs.map(p => this.elm.oneHot(this.elm.categories.length, this.elm.categories.indexOf(p.label)));
        this.elm.trainFromData(X, Y);
        console.info("✅ ELM training complete.");
    }

    public getEmbedding(text: string): number[] {
        const inputVec = this.encoder.encode(text);
        const inputSeq = this.encodeInput(text);
        switch (this.config.mode) {
            case ELMTransformerMode.ELM_TO_TRANSFORMER:
                const elmEmbedding = this.elm.getEmbedding([inputVec])[0];
                const reshaped: number[][] = [];
                const chunk = Math.floor(elmEmbedding.length / this.config.seqLen);
                for (let i = 0; i < this.config.seqLen; i++) {
                    const slice = elmEmbedding.slice(i * chunk, (i + 1) * chunk);
                    reshaped.push(slice.length < chunk
                        ? slice.concat(Array(chunk - slice.length).fill(0))
                        : slice);
                }
                return this.transformerEncode(reshaped, this.transformerWeights);
            default:
                return this.transformerEncode(inputSeq, this.transformerWeights);
        }
    }

    public saveModelAsJSONFile(filename?: string): void {
        this.elm.saveModelAsJSONFile(filename);
    }

    public loadModelFromJSON(json: string): void {
        this.elm.loadModelFromJSON(json);
    }
}
