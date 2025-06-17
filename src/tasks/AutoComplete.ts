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
        this.model.train(options.augmentationOptions);

        bindAutocompleteUI({
            model: this.model,
            inputElement: options.inputElement,
            outputElement: options.outputElement,
            topK: options.topK
        });
    }

    public getModel(): ELM {
        return this.model;
    }
}
