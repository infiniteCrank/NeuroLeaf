export interface ELMConfig {
    categories: string[];
    hiddenUnits?: number;
    maxLen?: number;
    activation?: string;
}

export const defaultConfig: Required<ELMConfig> = {
    categories: [],
    hiddenUnits: 120,
    maxLen: 15,
    activation: 'relu'
};
