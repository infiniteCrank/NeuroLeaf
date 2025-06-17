import { describe, it, expect, vi, beforeEach } from 'vitest';
import { AutoComplete } from '../src/tasks/AutoComplete';
import { ELM } from '../src/core/ELM';
import * as BindUI from '../src/ui/components/BindUI';
import { EnglishTokenPreset } from '../src/config/Presets';

vi.mock('../src/core/ELM', () => {
    return {
        ELM: vi.fn().mockImplementation(() => ({
            train: vi.fn(),
            predict: vi.fn().mockReturnValue([]),
        })),
    };
});

vi.mock('../src/ui/components/BindUI', () => ({
    bindAutocompleteUI: vi.fn()
}));

describe('AutoComplete', () => {
    let inputElement: HTMLInputElement;
    let outputElement: HTMLElement;

    beforeEach(() => {
        inputElement = document.createElement('input');
        outputElement = document.createElement('div');
    });

    it('initializes ELM with given categories and presets', () => {
        const categories = ['rock', 'pop', 'jazz'];
        const auto = new AutoComplete(categories, {
            inputElement,
            outputElement
        });

        expect(ELM).toHaveBeenCalledWith(expect.objectContaining({
            ...EnglishTokenPreset,
            categories
        }));

        const instance = auto.getModel();
        expect(instance.train).toHaveBeenCalled();
        expect(BindUI.bindAutocompleteUI).toHaveBeenCalledWith(expect.objectContaining({
            model: instance,
            inputElement,
            outputElement,
            topK: undefined
        }));
    });

    it('passes topK and augmentation options to model and UI binder', () => {
        const categories = ['a', 'b'];
        const topK = 10;
        const suffixes = ['x', 'y'];
        const auto = new AutoComplete(categories, {
            inputElement,
            outputElement,
            topK,
            augmentationOptions: { suffixes }
        });

        const instance = auto.getModel();
        expect(instance.train).toHaveBeenCalledWith({ suffixes });
        expect(BindUI.bindAutocompleteUI).toHaveBeenCalledWith(expect.objectContaining({
            model: instance,
            inputElement,
            outputElement,
            topK
        }));
    });

    it('getModel returns ELM instance', () => {
        const categories = ['test'];
        const auto = new AutoComplete(categories, {
            inputElement,
            outputElement
        });

        expect(auto.getModel()).toBeDefined();
        expect(typeof auto.getModel().predict).toBe('function');
    });
});
