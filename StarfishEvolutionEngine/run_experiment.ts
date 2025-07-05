// run_experiment.ts

import fs from "fs";
import { ModulePool, Module } from "../src/core/ModulePool";
import { TFIDFVectorizer } from "../src/ml/TFIDF";
import { KNN } from "../src/ml/KNN";
import { ELM } from "../src/core/ELM";
import { MiniTransformer, Vocab } from "../src/core/MiniTransformer";
import { CONFIG } from "./config";
import { QUERIES } from "./queries";

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

// Prepare TFIDF
const vectorizer = new TFIDFVectorizer(texts);
const tfidfVectors = texts.map((t) => TFIDFVectorizer.l2normalize(vectorizer.vectorize(t)));
const knnData = texts.map((t, i) => ({
    vector: tfidfVectors[i],
    label: String(i),
}));

// Initialize or load module pool
const pool = new ModulePool();
if (fs.existsSync("./module_pool.json")) {
    const loaded = JSON.parse(fs.readFileSync("./module_pool.json", "utf8"));
    loaded.forEach((m: Module) => {
        const hydrated: Module = {
            ...m,
            elm: m.elm ? new ELM(m.elm.config) : undefined,
            transformer: m.transformer ? new MiniTransformer(new Vocab(m.transformer.vocab.tokens)) : undefined,
        };
        pool.addModule(hydrated);
    });
    console.log(`✅ Loaded existing module pool.`);
} else {
    console.log(`⚙️ No module_pool.json found. Creating initial random pool...`);
    const allChars = new Set<string>();
    rawText.split("").forEach((c) => allChars.add(c.toLowerCase()));
    const vocab = new Vocab([...CONFIG.vocabSpecialTokens, ...allChars]);

    for (let i = 0; i < 5; i++) {
        pool.addModule({
            id: `elm-${i}`,
            type: "ELM",
            elm: new ELM({
                categories: [],
                hiddenUnits: 64 + Math.floor(Math.random() * 128),
                activation: ["relu", "tanh", "leakyRelu"][i % 3],
                maxLen: 2000,
                log: { modelName: `ELM-${i}`, verbose: false },
                dropout: Math.random() * 0.05,
            }),
            role: "retrieval",
            metrics: { recall1: 0, recall5: 0, mrr: 0, avgLatency: 0 },
            lastEvaluated: Date.now(),
        });
    }

    for (let i = 0; i < 5; i++) {
        pool.addModule({
            id: `transformer-${i}`,
            type: "Transformer",
            transformer: new MiniTransformer(vocab),
            role: "retrieval",
            metrics: { recall1: 0, recall5: 0, mrr: 0, avgLatency: 0 },
            lastEvaluated: Date.now(),
        });
    }

    fs.writeFileSync("./module_pool.json", JSON.stringify(pool.listModules(), null, 2));
    console.log(`✅ Initialized module pool.`);
}

// CSV log
const csvFile = `experiment_log_${new Date().toISOString().replace(/[:.]/g, "-")}.csv`;
fs.writeFileSync(csvFile, "generation,module_id,mean_recall1,mean_latency\n");

let generation = 0;

(async function mainLoop() {
    console.log(`🔄 Starting experiment loop...`);

    while (generation < CONFIG.generations) {
        generation++;
        console.log(`\n✨ Generation ${generation}`);

        const pool = new ModulePool();
        const loaded = JSON.parse(fs.readFileSync("./module_pool.json", "utf8"));
        loaded.forEach((m: Module) => {
            const hydrated: Module = {
                ...m,
                elm: m.elm ? new ELM(m.elm.config) : undefined,
                transformer: m.transformer ? new MiniTransformer(new Vocab(m.transformer.vocab.tokens)) : undefined,
            };
            pool.addModule(hydrated);
        });

        for (const mod of pool.listModules()) {
            let recalls: number[] = [];
            let latencies: number[] = [];

            if (mod.elm && !mod.elm.model) {
                mod.elm.trainFromData(tfidfVectors, tfidfVectors);
            }

            for (const query of QUERIES.slice(0, CONFIG.queriesPerGeneration)) {
                const queryVec = TFIDFVectorizer.l2normalize(vectorizer.vectorize(query));
                const start = performance.now();
                let recall1 = 0;

                if (mod.elm) {
                    const embedding = mod.elm.computeHiddenLayer([queryVec])[0];
                    const results = KNN.find(embedding, knnData, 5, 5, "cosine");
                    recall1 = results[0].label === "3" ? 1 : 0;
                }
                if (mod.transformer) {
                    const embedding = mod.transformer.encode(
                        query.toLowerCase().split("").map((c) => mod.transformer!.vocab.tokenToIdx[c] ?? 1)
                    );
                    const results = KNN.find(embedding, knnData, 5, 5, "cosine");
                    recall1 = results[0].label === "3" ? 1 : 0;
                }

                recalls.push(recall1);
                latencies.push(performance.now() - start);
            }

            const meanRecall = recalls.reduce((a, b) => a + b, 0) / recalls.length;
            const meanLatency = latencies.reduce((a, b) => a + b, 0) / latencies.length;
            mod.metrics.recall1 = meanRecall;
            mod.metrics.avgLatency = meanLatency;

            fs.appendFileSync(csvFile, `${generation},${mod.id},${meanRecall.toFixed(4)},${meanLatency.toFixed(2)}\n`);

            if (meanRecall < CONFIG.recallThresholdRemove) {
                console.log(`❌ Removing ${mod.id} (recall=${meanRecall.toFixed(2)})`);
                pool.removeModule(mod.id);
                continue;
            }
            if (meanRecall < CONFIG.recallThresholdRetrain && mod.elm) {
                console.log(`🔄 Retraining ${mod.id}...`);
                mod.elm.trainFromData(tfidfVectors, tfidfVectors);
            }
            if (meanRecall > CONFIG.recallThresholdClone) {
                const cloned: Module = {
                    ...mod,
                    id: `${mod.id}-clone-${Date.now()}`,
                };
                console.log(`✨ Cloning ${mod.id} as ${cloned.id}`);
                pool.addModule(cloned);
            }
        }

        fs.writeFileSync("./module_pool.json", JSON.stringify(pool.listModules(), null, 2));
        console.log(`✅ Saved generation ${generation}.`);
    }
})();
