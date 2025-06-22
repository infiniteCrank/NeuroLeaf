export class Tokenizer {
    private charSet: string[];
    private charToIndex: Record<string, number>;
    private maxLen: number;
    private useTokenizer: boolean;
    private tokenizerDelimiter: RegExp;

    constructor(config: {
        charSet: string;
        maxLen?: number;
        useTokenizer?: boolean;
        tokenizerDelimiter?: RegExp;
    }) {
        this.charSet = config.charSet.split('');
        this.charToIndex = Object.fromEntries(this.charSet.map((c, i) => [c, i]));
        this.maxLen = config.maxLen ?? 30;
        this.useTokenizer = config.useTokenizer ?? false;
        this.tokenizerDelimiter = config.tokenizerDelimiter ?? /\s+/;
    }

    encode(input: string): number[] {
        const vec = new Array(this.charSet.length * this.maxLen).fill(0);

        const tokens = this.useTokenizer
            ? input.split(this.tokenizerDelimiter)
            : input.split('');

        for (let i = 0; i < Math.min(tokens.length, this.maxLen); i++) {
            const token = tokens[i];
            const index = this.charToIndex[token];
            if (index !== undefined) {
                vec[i * this.charSet.length + index] = 1;
            }
        }

        return vec;
    }
}
