import { ModulePool, Module } from "../src/core/ModulePool";
import fs from "fs";
import { TFIDFVectorizer } from "../src/ml/TFIDF";
import { KNN } from "../src/ml/KNN";

const pool = new ModulePool();
const loaded = JSON.parse(fs.readFileSync("./module_pool.json", "utf8"));
loaded.forEach((m: Module) => pool.addModule(m));

const rawText = fs.readFileSync("../public/go_textbook.md", "utf8");
const sections = rawText.split(/\n(?=#{1,6}\s)/).filter((s) => s.length > 30);
const texts = sections.map((s) => s.replace(/^#{1,6}\s/, ""));

const vectorizer = new TFIDFVectorizer(texts);
const tfidfVectors = texts.map(t => TFIDFVectorizer.l2normalize(vectorizer.vectorize(t)));

const knnData = texts.map((t, i) => ({
    vector: tfidfVectors[i],
    label: String(i),
}));

const query = "How do you declare a map in Go?";
const queryVec = TFIDFVectorizer.l2normalize(vectorizer.vectorize(query));

for (const mod of pool.listModules()) {
    const start = performance.now();
    let recall1 = 0;

    if (mod.elm) {
        const embedding = mod.elm.computeHiddenLayer([queryVec])[0];
        const results = KNN.find(embedding, knnData, 5, 5, "cosine");
        recall1 = results[0].label === "3" ? 1 : 0;
    }

    if (
        mod.transformer &&
        mod.transformer.vocab &&
        mod.transformer.vocab.tokenToIdx
    ) {
        const embedding = mod.transformer.encode(
            query
                .toLowerCase()
                .split("")
                .map(c => mod.transformer!.vocab.tokenToIdx[c] ?? 1)
        );

        const results = KNN.find(embedding, knnData, 5, 5, "cosine");
        recall1 = results[0].label === "3" ? 1 : 0;
    }

    const latency = performance.now() - start;
    mod.metrics.recall1 = recall1;
    mod.metrics.avgLatency = latency;
    mod.lastEvaluated = Date.now();
}

fs.writeFileSync("./module_pool.json", JSON.stringify(pool.listModules(), null, 2));
console.log("✅ Evaluation complete.");
