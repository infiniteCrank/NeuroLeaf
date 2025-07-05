// baseline_experiment.ts

import fs from "fs";
import { TFIDFVectorizer } from "../src/ml/TFIDF";
import { KNN, KNNDataPoint } from "../src/ml/KNN";
import { ELM } from "../src/core/ELM";
import { Vocab, MiniTransformer } from "../src/core/MiniTransformer";

interface Module {
    id: string;
    elm?: ELM;
    transformer?: MiniTransformer;
}

function cosine(a: number[], b: number[]): number {
    const dot = a.reduce((sum, v, i) => sum + v * b[i], 0);
    const normA = Math.sqrt(a.reduce((s, x) => s + x * x, 0));
    const normB = Math.sqrt(b.reduce((s, x) => s + x * x, 0));
    return normA && normB ? dot / (normA * normB) : 0;
}

// Load corpus
const rawText = fs.readFileSync("../public/go_textbook.md", "utf8");
const sections = rawText
    .split(/\n(?=#{1,6}\s)/)
    .map((block) => {
        const lines = block.split("\n").filter(Boolean);
        const heading = lines.find((l) => /^#{1,6}\s/.test(l)) || "";
        const content = lines.filter((l) => !/^#{1,6}\s/.test(l)).join(" ").trim();
        return { heading: heading.replace(/^#{1,6}\s/, ""), content };
    })
    .filter((s) => s.content.length > 30);
const texts = sections.map((s) => `${s.heading} ${s.content}`);

// TFIDF
const vectorizer = new TFIDFVectorizer(texts);
const tfidfVectors = texts.map((t) =>
    TFIDFVectorizer.l2normalize(vectorizer.vectorize(t))
);

// Build vocab
const SPECIAL_TOKENS = ["<PAD>", "<UNK>"];
const allChars = new Set<string>(SPECIAL_TOKENS);
texts.forEach((t) => t.toLowerCase().split("").forEach((c) => allChars.add(c)));
const vocab = new Vocab([...allChars]);

// Create module pool
const pool: Module[] = [];
for (let i = 0; i < 5; i++) {
    pool.push({
        id: `ELM_${i}`,
        elm: new ELM({
            categories: [],
            hiddenUnits: 64,
            maxLen: tfidfVectors[0].length,
            activation: "relu",
            log: { modelName: `ELM_${i}`, verbose: false },
            dropout: 0.02,
        }),
    });
}
for (let i = 0; i < 5; i++) {
    pool.push({
        id: `Transformer_${i}`,
        transformer: new MiniTransformer(vocab),
    });
}

// Train ELMs
pool.forEach((mod) => {
    if (mod.elm) {
        mod.elm.trainFromData(tfidfVectors, tfidfVectors);
    }
});

// Prepare CSV output
const csvLines = ["module_id,recall_at_1,mean_cosine_top5"];

// Baseline evaluation query
const query = "How do you declare a map in Go?";
const queryTFIDF = TFIDFVectorizer.l2normalize(vectorizer.vectorize(query));

// Evaluate each module
for (const mod of pool) {
    let embedding: number[] = [];
    if (mod.elm) {
        embedding = mod.elm.computeHiddenLayer([queryTFIDF])[0];
    } else if (mod.transformer) {
        embedding = mod.transformer.encode(
            query
                .toLowerCase()
                .split("")
                .map(
                    (c) =>
                        mod.transformer!.vocab.tokenToIdx[c] ??
                        mod.transformer!.vocab.tokenToIdx["<UNK>"]!
                )
        );
    }

    // Build KNN dataset
    const knnData: KNNDataPoint[] = texts.map((_, i) => ({
        vector: mod.elm
            ? mod.elm!.computeHiddenLayer([tfidfVectors[i]])[0]
            : mod.transformer!.encode(
                texts[i]
                    .toLowerCase()
                    .split("")
                    .map(
                        (c) =>
                            mod.transformer!.vocab.tokenToIdx[c] ??
                            mod.transformer!.vocab.tokenToIdx["<UNK>"]!
                    )
            ),
        label: String(i),
    }));

    const results = KNN.find(embedding, knnData, 5, 5, "cosine");
    const recall1 = results[0].label === "3" ? 1 : 0;
    const meanCosine =
        results.reduce((s, r) => s + r.weight, 0) / results.length;

    csvLines.push(`${mod.id},${recall1},${meanCosine.toFixed(4)}`);
}

// Save metrics
const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
fs.writeFileSync(
    `./baseline_metrics_${timestamp}.csv`,
    csvLines.join("\n"),
    "utf8"
);

console.log(`✅ Baseline evaluation complete.`);
