// ELMConfig.ts - Configuration interface and defaults for ELM-based models

export interface ELMConfig {
    categories: string[];
    hiddenUnits?: number;
    maxLen?: number;
    activation?: string;
    encoder?: any; // Optional: if you're passing a pre-built UniversalEncoder instance

    // Preprocessing options
    charSet?: string;
    useTokenizer?: boolean;
    tokenizerDelimiter?: RegExp;

    //file loading 
    exportFileName?: string
}

export const defaultConfig: Required<Pick<ELMConfig, 'categories' | 'hiddenUnits' | 'maxLen' | 'activation'>> = {
    categories: [],
    hiddenUnits: 120,
    maxLen: 15,
    activation: 'relu'
};
