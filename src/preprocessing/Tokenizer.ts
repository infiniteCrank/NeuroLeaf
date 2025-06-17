// Tokenizer.ts - Utility for splitting and tokenizing text inputs

export class Tokenizer {
    private delimiter: RegExp;

    constructor(customDelimiter?: RegExp) {
        // Default to splitting on whitespace and punctuation
        this.delimiter = customDelimiter || /[\s,.;!?()\[\]{}"']+/;
    }

    public tokenize(text: string): string[] {
        return text
            .trim()
            .toLowerCase()
            .split(this.delimiter)
            .filter(Boolean); // Remove empty tokens
    }

    public ngrams(tokens: string[], n: number): string[] {
        if (n <= 0 || tokens.length < n) return [];
        const result: string[] = [];
        for (let i = 0; i <= tokens.length - n; i++) {
            result.push(tokens.slice(i, i + n).join(' '));
        }
        return result;
    }
} 
