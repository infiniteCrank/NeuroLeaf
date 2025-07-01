export class TFIDF {
    termFrequency: Record<string, number> = {};
    inverseDocFreq: Record<string, number> = {};
    wordsInDoc: string[] = [];
    processedWords: string[] = [];
    scores: Record<string, number> = {};
    corpus: string = "";

    constructor(corpusDocs: string[]) {
        this.corpus = corpusDocs.join(" ");
        const wordsFinal: string[] = [];
        const re = /[^a-zA-Z0-9]+/g;

        corpusDocs.forEach(doc => {
            const tokens = doc.split(/\s+/);
            tokens.forEach(word => {
                const cleaned = word.replace(re, " ");
                wordsFinal.push(...cleaned.split(/\s+/).filter(Boolean));
            });
        });

        this.wordsInDoc = wordsFinal;
        this.processedWords = TFIDF.processWords(wordsFinal);

        // Compute term frequency
        this.processedWords.forEach(token => {
            this.termFrequency[token] = (this.termFrequency[token] || 0) + 1;
        });

        // Compute inverse document frequency
        for (const term in this.termFrequency) {
            const count = TFIDF.countDocsContainingTerm(corpusDocs, term);
            this.inverseDocFreq[term] = Math.log(corpusDocs.length / (1 + count));
        }
    }

    static countDocsContainingTerm(corpusDocs: string[], term: string): number {
        return corpusDocs.reduce((acc, doc) => (doc.includes(term) ? acc + 1 : acc), 0);
    }

    static processWords(words: string[]): string[] {
        const filtered = TFIDF.removeStopWordsAndStem(words).map(w => TFIDF.lemmatize(w));
        const bigrams = TFIDF.generateNGrams(filtered, 2);
        const trigrams = TFIDF.generateNGrams(filtered, 3);
        return [...filtered, ...bigrams, ...trigrams];
    }

    static removeStopWordsAndStem(words: string[]): string[] {
        const stopWords = new Set([
            "a", "and", "the", "is", "to", "of", "in", "it", "that", "you",
            "this", "for", "on", "are", "with", "as", "be", "by", "at", "from",
            "or", "an", "but", "not", "we"
        ]);
        return words.filter(w => !stopWords.has(w)).map(w => TFIDF.advancedStem(w));
    }

    static advancedStem(word: string): string {
        const programmingKeywords = new Set([
            "func", "package", "import", "interface", "go",
            "goroutine", "channel", "select", "struct",
            "map", "slice", "var", "const", "type",
            "defer", "fallthrough"
        ]);
        if (programmingKeywords.has(word)) return word;
        const suffixes = ["es", "ed", "ing", "s", "ly", "ment", "ness", "ity", "ism", "er"];
        for (const suffix of suffixes) {
            if (word.endsWith(suffix)) {
                if (suffix === "es" && word.length > 2 && word[word.length - 3] === "i") {
                    return word.slice(0, -2);
                }
                return word.slice(0, -suffix.length);
            }
        }
        return word;
    }

    static lemmatize(word: string): string {
        const rules: Record<string, string> = {
            execute: "execute",
            running: "run",
            returns: "return",
            defined: "define",
            compiles: "compile",
            calls: "call",
            creating: "create",
            invoke: "invoke",
            declares: "declare",
            references: "reference",
            implements: "implement",
            utilizes: "utilize",
            tests: "test",
            loops: "loop",
            deletes: "delete",
            functions: "function"
        };
        if (rules[word]) return rules[word];
        if (word.endsWith("ing")) return word.slice(0, -3);
        if (word.endsWith("ed")) return word.slice(0, -2);
        return word;
    }

    static generateNGrams(tokens: string[], n: number): string[] {
        if (tokens.length < n) return [];
        const ngrams: string[] = [];
        for (let i = 0; i <= tokens.length - n; i++) {
            ngrams.push(tokens.slice(i, i + n).join(" "));
        }
        return ngrams;
    }

    calculateScores(): Record<string, number> {
        const totalWords = this.processedWords.length;
        const scores: Record<string, number> = {};
        this.processedWords.forEach(token => {
            const tf = this.termFrequency[token] || 0;
            scores[token] = (tf / totalWords) * (this.inverseDocFreq[token] || 0);
        });
        this.scores = scores;
        return scores;
    }

    extractKeywords(topN: number): Record<string, number> {
        const entries = Object.entries(this.scores).sort((a, b) => b[1] - a[1]);
        return Object.fromEntries(entries.slice(0, topN));
    }

    processedWordsIndex(word: string): number {
        return this.processedWords.indexOf(word);
    }
}
