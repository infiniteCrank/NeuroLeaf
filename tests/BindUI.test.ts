// BindUI.test.ts - Unit + Integration + Edge Case tests for BindUI, IntentClassifier, AutoComplete, ELMConfig, LanguageClassifier

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { bindAutocompleteUI } from '../src/ui/components/BindUI';
import { ELM } from '../src/core/ELM';
import { IntentClassifier } from '../src/tasks/IntentClassifier';
import { AutoComplete } from '../src/tasks/AutoComplete';
import { LanguageClassifier } from '../src/tasks/LanguageClassifier';
import { EnglishCharPreset, RussianTokenPreset, EmojiHybridPreset } from '../src/config/Presets';
import { IO } from '../src/utils/IO';
import { UniversalEncoder } from '../src/preprocessing/UniversalEncoder';
import { Augment } from '../src/utils/Augment';

// Mocked PredictResult
const mockResults = [
    { label: 'rock', prob: 0.9 },
    { label: 'pop', prob: 0.1 }
];

// BindUI tests

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

// UniversalEncoder Switching

describe('UniversalEncoder Switching', () => {
    it('encodes input differently in char and token mode with consistent vector sizes', () => {
        const text = 'hello world';

        const charEncoder = new UniversalEncoder({
            charSet: 'abcdefghijklmnopqrstuvwxyz ',
            maxLen: 10,
            mode: 'char'
        });

        const tokenEncoder = new UniversalEncoder({
            charSet: 'abcdefghijklmnopqrstuvwxyz ',
            maxLen: 10,
            mode: 'token',
            tokenizerDelimiter: /\s+/
        });

        const charVec = charEncoder.encode(text);
        const tokenVec = tokenEncoder.encode(text);

        expect(charVec.length).toBe(charEncoder.getVectorSize());
        expect(tokenVec.length).toBe(tokenEncoder.getVectorSize());
        expect(charVec).not.toEqual(tokenVec); // Ensure mode has effect
    });
});

// ELMConfig Presets

describe('ELMConfig Preset Configuration', () => {
    it('has correct defaults for EnglishCharPreset', () => {
        expect(EnglishCharPreset.charSet).toContain('a');
        expect(EnglishCharPreset.useTokenizer).toBe(false);
        expect(EnglishCharPreset.activation).toBeDefined();
    });

    it('has correct configuration for RussianTokenPreset', () => {
        expect(RussianTokenPreset.charSet).toContain('а');
        expect(RussianTokenPreset.useTokenizer).toBe(true);
        expect(RussianTokenPreset.tokenizerDelimiter).toBeInstanceOf(RegExp);
    });

    it('has correct configuration for EmojiHybridPreset', () => {
        expect(EmojiHybridPreset.charSet).toContain('🎵');
        expect(EmojiHybridPreset.useTokenizer).toBe(true);
        expect(EmojiHybridPreset.tokenizerDelimiter).toBeInstanceOf(RegExp);
    });
});

// AutoComplete

describe('AutoComplete integration', () => {
    it('initializes and binds autocomplete correctly', () => {
        const input = document.createElement('input');
        const output = document.createElement('div');
        const ac = new AutoComplete(['rock', 'pop'], { inputElement: input, outputElement: output });
        input.value = 'roc';
        input.dispatchEvent(new Event('input'));
        expect(output.innerHTML).toContain('rock');
    });
});

// IO Utilities

describe('IO utilities', () => {
    it('parses JSON correctly', () => {
        const json = '[{"text":"hi","label":"greet"}]';
        const parsed = IO.importJSON(json);
        expect(parsed.length).toBe(1);
        expect(parsed[0].text).toBe('hi');
    });

    it('parses CSV correctly', () => {
        const csv = 'text,label\nhello,English';
        const parsed = IO.importCSV(csv);
        expect(parsed.length).toBe(1);
        expect(parsed[0].label).toBe('English');
    });

    it('parses TSV correctly', () => {
        const tsv = 'text\tlabel\nbonjour\tFrench';
        const parsed = IO.importTSV(tsv);
        expect(parsed.length).toBe(1);
        expect(parsed[0].label).toBe('French');
    });

    it('exports to JSON', () => {
        const data = [{ text: 'hi', label: 'greet' }];
        const json = IO.exportJSON(data);
        expect(json).toContain('hi');
    });

    it('infers schema correctly', () => {
        const csv = 'text,label\nhello,English';
        const schema = IO.inferSchemaFromCSV(csv);
        expect(schema.fields.length).toBeGreaterThan(0);
        expect(Object.keys(schema.suggestedMapping || {})).toContain('text');
    });
});

// Augment

describe('Augment utilities', () => {
    it('adds suffixes', () => {
        const variants = Augment.addSuffix('hi', ['there']);
        expect(variants).toContain('hi there');
    });

    it('adds prefixes', () => {
        const variants = Augment.addPrefix('hi', ['yo']);
        expect(variants).toContain('yo hi');
    });

    it('adds noise', () => {
        const noisy = Augment.addNoise('hi', 'xyz', 1);
        expect(noisy).not.toBe('hi');
    });

    it('mixes text with mixins', () => {
        const mixed = Augment.mix('hi', ['there']);
        expect(mixed).toContain('hi there');
    });

    it('generates multiple variants', () => {
        const all = Augment.generateVariants('hi', 'abc', { suffixes: ['you'], includeNoise: true });
        expect(all.length).toBeGreaterThan(1);
    });
});
