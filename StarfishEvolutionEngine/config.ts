// config.ts

export const CONFIG = {
    generations: Infinity,
    queriesPerGeneration: 10,
    recallThresholdRetrain: 0.2,
    recallThresholdRemove: 0.1,
    recallThresholdClone: 0.7,
    hiddenUnitsMutationRange: 16,
    dropoutMutationRange: 0.02,
    vocabSpecialTokens: ["<PAD>", "<UNK>"],
};
