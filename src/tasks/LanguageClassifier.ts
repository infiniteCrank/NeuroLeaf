import { ELM } from '../core/ELM';
import { ELMConfig } from '../core/ELMConfig';
import { IO, LabeledExample } from '../utils/IO';

export class LanguageClassifier {
    private elm: ELM;
    private config: ELMConfig;

    constructor(config: ELMConfig) {
        this.config = config;
        this.elm = new ELM(config);
    }

    loadTrainingData(raw: string, format: 'json' | 'csv' | 'tsv' = 'json'): LabeledExample[] {
        switch (format) {
            case 'csv':
                return IO.importCSV(raw);
            case 'tsv':
                return IO.importTSV(raw);
            case 'json':
            default:
                return IO.importJSON(raw);
        }
    }

    train(data: LabeledExample[]): void {
        const categories = [...new Set(data.map(d => d.label))];
        this.elm.setCategories(categories);
        data.forEach(({ text, label }) => {
            if (!this.trainSamples[label]) this.trainSamples[label] = [];
            this.trainSamples[label].push(text);
        });
        this.elm.train();
    }

    predict(text: string, topK = 3) {
        return this.elm.predict(text, topK);
    }

    private trainSamples: Record<string, string[]> = {};
}
