// MiniTransformer.ts
export class Vocab {
    public tokens: string[];
    public tokenToIdx: Record<string, number>;
    public idxToToken: Record<number, string>;

    constructor(tokens: string[]) {
        this.tokens = tokens;
        this.tokenToIdx = {};
        this.idxToToken = {};
        tokens.forEach((t, i) => {
            this.tokenToIdx[t] = i;
            this.idxToToken[i] = t;
        });
    }

    encode(text: string): number[] {
        return text
            .split("")
            .map(c => this.tokenToIdx[c])
            .filter((idx): idx is number => idx !== undefined);
    }
}

export class MiniTransformer {
    embedDim = 32;
    seqLen = 16;
    embedding: number[][];
    posEnc: number[][];

    constructor(public vocab: Vocab) {
        this.embedding = this.randomMatrix(vocab.tokens.length, this.embedDim);
        this.posEnc = this.positionalEncoding(this.seqLen, this.embedDim);
    }

    randomMatrix(rows: number, cols: number): number[][] {
        return Array.from({ length: rows }, () =>
            Array.from({ length: cols }, () => Math.random() * 2 - 1)
        );
    }

    positionalEncoding(seqLen: number, embedDim: number): number[][] {
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

    encode(inputIndices: number[]): number[] {
        if (!inputIndices || inputIndices.length === 0) {
            // Fallback: zero vector
            return Array(this.embedDim).fill(0);
        }

        // For each token index, get embedding + position encoding
        const x: number[][] = inputIndices.map((idx, i) => {
            if (idx === undefined) {
                // Unknown token fallback embedding
                return Array.from({ length: this.embedDim }, () => 0);
            }
            return this.embedding[idx].map((v, j) => v + this.posEnc[i % this.seqLen][j]);
        });

        // Aggregate into a single embedding (mean)
        return this.meanVecs(x);
    }

    meanVecs(vecs: number[][]): number[] {
        const out = Array(vecs[0].length).fill(0);
        for (const vec of vecs) {
            for (let i = 0; i < vec.length; i++) {
                out[i] += vec[i];
            }
        }
        return out.map(v => v / vecs.length);
    }
}
