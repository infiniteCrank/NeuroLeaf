import { ELM } from '../core/ELM';
import { ELMConfig } from '../core/ELMConfig';

/**
 * VotingClassifierELM takes predictions from multiple ELMs
 * and learns to choose the most accurate final label.
 */
export class VotingClassifierELM {
    private elm: ELM;

    constructor(config: ELMConfig) {
        this.elm = new ELM({
            ...config,
            useTokenizer: false
        });
    }

    train(predictionsA: string[], predictionsB: string[], trueLabels: string[]): void {
        const inputs = predictionsA.map((labelA, i) => [
            ...this.oneHot(labelA),
            ...this.oneHot(predictionsB[i])
        ]);
        const examples = inputs.map((input, i) => ({ input, label: trueLabels[i] }));
        this.elm.train(examples as any);
    }

    predict(labelA: string, labelB: string): { label: string; prob: number }[] {
        const input = [
            ...this.oneHot(labelA),
            ...this.oneHot(labelB)
        ];
        return this.elm.predict(JSON.stringify(input), 1);
    }

    private oneHot(label: string): number[] {
        const categories = this.elm.categories;
        return categories.map(c => (c === label ? 1 : 0));
    }

}
