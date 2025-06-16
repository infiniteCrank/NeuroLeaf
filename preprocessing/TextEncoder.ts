// TextEncoder.ts - Text preprocessing and one-hot encoding for ELM

export interface TextEncoderConfig {
    charSet?: string;
    maxLen?: number;
}

const defaultTextEncoderConfig: Required<TextEncoderConfig> = {
    charSet: 'abcdefghijklmnopqrstuvwxyz',
    maxLen: 15
};

export class TextEncoder {
    private charSet: string;
    private charSize: number;
    private maxLen: number;

    constructor(config: TextEncoderConfig = {}) {
        const cfg = { ...defaultTextEncoderConfig, ...config };
        this.charSet = cfg.charSet;
        this.charSize = cfg.charSet.length;
        this.maxLen = cfg.maxLen;
    }

    public charToOneHot(c: string): number[] {
        const index = this.charSet.indexOf(c.toLowerCase());
        const vec = Array(this.charSize).fill(0);
        if (index !== -1) vec[index] = 1;
        return vec;
    }

    public textToVector(text: string): number[] {
        const cleaned = text.toLowerCase().replace(new RegExp(`[^${this.charSet}]`, 'g'), '').padEnd(this.maxLen, ' ').slice(0, this.maxLen);
        const vec: number[] = [];
        for (let i = 0; i < cleaned.length; i++) {
            vec.push(...this.charToOneHot(cleaned[i]));
        }
        return vec;
    }

    public normalizeVector(v: number[]): number[] {
        const norm = Math.sqrt(v.reduce((sum, x) => sum + x * x, 0));
        return norm > 0 ? v.map(x => x / norm) : v;
    }

    public getVectorSize(): number {
        return this.charSize * this.maxLen;
    }

    public getCharSet(): string {
        return this.charSet;
    }

    public getMaxLen(): number {
        return this.maxLen;
    }
} 
