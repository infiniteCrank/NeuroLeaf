// BindUI.test.ts - Unit + Integration + Edge Case tests for BindUI, IntentClassifier, AutoComplete, ELMConfig, LanguageClassifier

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { bindAutocompleteUI } from '../src/ui/components/BindUI';
import { ELM } from '../src/core/ELM';
import { IntentClassifier } from '../src/tasks/IntentClassifier';
import { AutoComplete } from '../src/tasks/AutoComplete';
import { LanguageClassifier } from '../src/tasks/LanguageClassifier';
import { defaultConfig } from '../src/core/ELMConfig';
import { IO } from '../src/utils/IO';
import { UniversalEncoder } from '../src/preprocessing/UniversalEncoder';

// Mocked PredictResult
const mockResults = [
    { label: 'rock', prob: 0.9 },
    { label: 'pop', prob: 0.1 }
];

// ... (existing tests above remain unchanged)

// Additional test: UniversalEncoder Switching

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
