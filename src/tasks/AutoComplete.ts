// AutoComplete.ts - High-level autocomplete controller using ELM

import { ELM } from '../core/ELM';
import { bindAutocompleteUI } from '../ui/components/BindUI';
import { EnglishTokenPreset } from '../config/Presets';

export class AutoComplete {

    public elm: ELM;

    constructor(categories: string[], options: {
        inputElement: HTMLInputElement;
        outputElement: HTMLElement;
        topK?: number;
        metrics?: { rmse?: number; mae?: number; accuracy?: number };
        verbose?: boolean;
        exportFileName?: string;
        augmentationOptions?: {
            suffixes?: string[];
            prefixes?: string[];
            includeNoise?: boolean;
        };
    }) {
        this.elm = new ELM({
            ...EnglishTokenPreset,
            categories,
            metrics: options.metrics,
            verbose: options.verbose,
            exportFileName: options.exportFileName
        });

        // Train the model, safely handling optional augmentationOptions
        this.elm.train(options?.augmentationOptions);

        bindAutocompleteUI({
            model: this.elm,
            inputElement: options.inputElement,
            outputElement: options.outputElement,
            topK: options.topK
        });
    }

    predict(input: string, topN = 1): { completion: string; prob: number }[] {
        return this.elm.predict(input).slice(0, topN).map(p => ({
            completion: p.label,
            prob: p.prob
        }));
    }

    public getModel(): ELM {
        return this.elm;
    }

    public loadModelFromJSON(json: string): void {
        this.elm.loadModelFromJSON(json);
    }

    public saveModelAsJSONFile(filename?: string): void {
        this.elm.saveModelAsJSONFile(filename);
    }

}
