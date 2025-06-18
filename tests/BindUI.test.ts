// BindUI.test.ts - Unit + Integration + Edge Case tests for BindUI, IntentClassifier, and AutoComplete

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { bindAutocompleteUI } from '../src/ui/components/BindUI';
import { ELM } from '../src/core/ELM';
import { IntentClassifier } from '../src/tasks/IntentClassifier';
import { AutoComplete } from '../src/tasks/AutoComplete';

// Mocked PredictResult
const mockResults = [
    { label: 'rock', prob: 0.9 },
    { label: 'pop', prob: 0.1 }
];

describe('bindAutocompleteUI', () => {
    let input: HTMLInputElement;
    let output: HTMLElement;
    let mockModel: ELM;

    beforeEach(() => {
        input = document.createElement('input');
        output = document.createElement('div');
        mockModel = {
            predict: vi.fn(() => mockResults)
        } as unknown as ELM;
    });

    it('shows prompt when input is empty', () => {
        bindAutocompleteUI({ model: mockModel, inputElement: input, outputElement: output });
        input.value = '';
        input.dispatchEvent(new Event('input'));
        expect(output.innerHTML).toContain('Start typing');
    });

    it('renders predictions when input is non-empty', () => {
        bindAutocompleteUI({ model: mockModel, inputElement: input, outputElement: output });
        input.value = 'ro';
        input.dispatchEvent(new Event('input'));
        expect(output.innerHTML).toContain('rock');
        expect(output.innerHTML).toContain('90.0%');
    });

    it('displays error if model.predict throws', () => {
        mockModel.predict = vi.fn(() => { throw new Error('Prediction failed'); }) as any;
        bindAutocompleteUI({ model: mockModel, inputElement: input, outputElement: output });
        input.value = 'error';
        input.dispatchEvent(new Event('input'));
        expect(output.innerHTML).toContain('Error: Prediction failed');
    });

    it('handles large input gracefully', () => {
        bindAutocompleteUI({ model: mockModel, inputElement: input, outputElement: output });
        input.value = 'a'.repeat(1000);
        input.dispatchEvent(new Event('input'));
        expect(output.innerHTML).toContain('rock');
    });

    it('handles unusual unicode characters', () => {
        bindAutocompleteUI({ model: mockModel, inputElement: input, outputElement: output });
        input.value = 'こんにちは';
        input.dispatchEvent(new Event('input'));
        expect(output.innerHTML).toContain('rock');
    });
});

describe('IntentClassifier Integration', () => {
    it('trains and predicts intents', () => {
        const classifier = new IntentClassifier({
            categories: ['greeting', 'farewell'],
            hiddenUnits: 10,
            maxLen: 10,
            activation: 'relu'
        });

        const samples = [
            { text: 'hello', label: 'greeting' },
            { text: 'hi', label: 'greeting' },
            { text: 'bye', label: 'farewell' }
        ];

        classifier.train(samples);

        const result = classifier.predict('hello', 1);
        expect(result.length).toBe(1);
        const [top] = result;
        expect(['greeting', 'farewell']).toContain(top.label);
        expect(top.prob).toBeGreaterThan(0);
    });

    it('returns empty array if no intent is above threshold', () => {
        const classifier = new IntentClassifier({
            categories: ['greeting'],
            hiddenUnits: 10,
            maxLen: 10,
            activation: 'relu'
        });
        classifier.train([{ text: 'hello', label: 'greeting' }]);

        const result = classifier.predict('unknown', 1, 1.1); // Raise threshold to 1.1 to force empty
        expect(result).toEqual([]);
    });
});

describe('AutoComplete Integration', () => {
    it('binds input and generates suggestions', () => {
        const input = document.createElement('input');
        const output = document.createElement('div');
        document.body.appendChild(input);
        document.body.appendChild(output);

        const ac = new AutoComplete(['rock', 'jazz'], {
            inputElement: input,
            outputElement: output,
            topK: 1
        });

        input.value = 'ro';
        input.dispatchEvent(new Event('input'));
        expect(output.innerHTML).toMatch(/rock/i);
    });

    it('does not fail on empty input field', () => {
        const input = document.createElement('input');
        const output = document.createElement('div');
        document.body.appendChild(input);
        document.body.appendChild(output);

        const ac = new AutoComplete(['classical', 'metal'], {
            inputElement: input,
            outputElement: output,
            topK: 1
        });

        input.value = '';
        input.dispatchEvent(new Event('input'));
        expect(output.innerHTML).toContain('Start typing');
    });
});
