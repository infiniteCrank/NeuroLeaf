// AutoComplete.ts - High-level autocomplete controller using ELM

import { ELM } from '../core/ELM';
import { bindAutocompleteUI } from '../ui/components/BindUI';
import { EnglishTokenPreset } from '../config/Presets';

export class AutoComplete {

    private model: ELM;

    constructor(categories: string[], options: {
        inputElement: HTMLInputElement;
        outputElement: HTMLElement;
        topK?: number;
        augmentationOptions?: {
            suffixes?: string[];
            prefixes?: string[];
            includeNoise?: boolean;
        };
    }) {
        this.model = new ELM({ ...EnglishTokenPreset, categories });

        // Train the model, safely handling optional augmentationOptions
        this.model.train(options?.augmentationOptions);

        bindAutocompleteUI({
            model: this.model,
            inputElement: options.inputElement,
            outputElement: options.outputElement,
            topK: options.topK
        });
    }

    predict(input: string, topN = 1): { completion: string; prob: number }[] {
        return this.model.predict(input).slice(0, topN).map(p => ({
            completion: p.label,
            prob: p.prob
        }));
    }

    public getModel(): ELM {
        return this.model;
    }

    public loadModelFromJSON(json: string): void {
        this.model.loadModelFromJSON(json);
    }

    public saveModelAsJSONFile(filename?: string): void {
        this.model.saveModelAsJSONFile(filename);
    }

}
