// BindUI.test.ts - Unit tests for BindUI

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { bindAutocompleteUI } from '../src/ui/components/BindUI';
import { ELM } from '../src/core/ELM';

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
});
