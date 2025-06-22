import { Matrix } from './Matrix';
import { Tokenizer } from '../preprocessing/Tokenizer';
import { Activations } from './Activations';

type ELMConfig = {
    charSet: string;
    maxLen: number;
    hiddenUnits: number;
    activation: keyof typeof Activations;
    useTokenizer: boolean;
    tokenizerDelimiter?: RegExp;
    categories?: string[];
    temperature?: number;
};

function normalize(vec: number[]): number[] {
    const norm = Math.sqrt(vec.reduce((acc, v) => acc + v * v, 0)) || 1;
    return vec.map(v => v / norm);
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

    constructor(config: ELMConfig & { charSet?: string; useTokenizer?: boolean; tokenizerDelimiter?: RegExp }) {
        const cfg = { ...defaultConfig, ...config };
        this.categories = cfg.categories;
        this.hiddenUnits = cfg.hiddenUnits;
        this.maxLen = cfg.maxLen;
        this.activation = cfg.activation;
        this.charSet = cfg.charSet ?? 'abcdefghijklmnopqrstuvwxyz';
        this.useTokenizer = cfg.useTokenizer ?? false;
        this.tokenizerDelimiter = cfg.tokenizerDelimiter;

        this.encoder = new UniversalEncoder({
            charSet: this.charSet,
            maxLen: this.maxLen,
            useTokenizer: this.useTokenizer,
            tokenizerDelimiter: this.tokenizerDelimiter,
            mode: this.useTokenizer ? 'token' : 'char'
        });
        type ActivationName = 'relu' | 'sigmoid' | 'tanh';
        type ActivationKey = `${ActivationName}Matrix`;

        this.activator = Activations[`${this.config.activation}Matrix` as ActivationKey];

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

    private oneHot(n: number, index: number): number[] {
        return Array.from({ length: n }, (_, i) => (i === index ? 1 : 0));
    }

    train(trainingData: { text: string; label: string }[]) {
        this.categories = this.config.categories || Array.from(new Set(trainingData.map(d => d.label)));
        const labelToIndex = Object.fromEntries(this.categories.map((label, i) => [label, i]));

        const X = trainingData.map(d => normalize(this.encoder.encode(d.text)));
        const Y = trainingData.map(d => {
            const row = Array(this.categories.length).fill(0);
            row[labelToIndex[d.label]] = 1;
            return row;
        });

        const Xmat = Matrix.from2D(X).transpose();
        const H = this.activator(Matrix.random(this.config.hiddenUnits, Xmat.rows).dot(Xmat));
        this.hiddenMatrix = H;

        const Ymat = Matrix.from2D(Y).transpose();
        const beta = Matrix.inverse(H.transpose().dot(H)).dot(H.transpose()).dot(Ymat);
        this.beta = beta;
    }

    predict(input: string): { label: string; prob: number }[] {
        const x = normalize(this.encoder.encode(input));
        const h = this.activator(this.hiddenMatrix.dot(Matrix.from2D([x]).transpose()));
        const out = this.beta.transpose().dot(h).toArray().map(r => r[0]);

        const temp = this.config.temperature || 1;
        const exps = out.map(v => Math.exp(v / temp));
        const sum = exps.reduce((a, b) => a + b, 0);
        const softmax = exps.map(v => v / sum);

        return this.categories.map((label, i) => ({ label, prob: softmax[i] }))
            .sort((a, b) => b.prob - a.prob);
    }
}
