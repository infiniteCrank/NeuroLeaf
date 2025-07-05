// run_experiment.ts
import fs from "fs";
import { ModulePool, Module } from "../src/core/ModulePool";
import { TFIDFVectorizer } from "../src/ml/TFIDF";
import { KNN } from "../src/ml/KNN";
import yargs from "yargs";
import { ELM } from "../src/core/ELM";
import { MiniTransformer } from "../src/core/MiniTransformer";

// CLI options
const argv = yargs
    .option("generations", {
        alias: "g",
        description: "Number of generations to run",
        type: "number",
    })
    .help()
    .alias("help", "h").argv;

const maxGenerations = argv.generations ?? Infinity;

const rawText = fs.readFileSync("./public/go_textbook.md", "utf8");
const sections = rawText.split(/\n(?=#{1,6}\s)/).filter((s) => s.length > 30);
const texts = sections.map((s) => s.replace(/^#{1,6}\s/, ""));

const vectorizer = new TFIDFVectorizer(texts);
const tfidfVectors = texts.map((t) =>
    TFIDFVectorizer.l2normalize(vectorizer.vectorize(t))
);
const knnData = texts.map((t, i) => ({
    vector: tfidfVectors[i],
    label: String(i),
}));

const query = "How do you declare a map in Go?";
const queryVec = TFIDFVectorizer.l2normalize(vectorizer.vectorize(query));

// CSV log
const csvFile = `experiment_log_${new Date().toISOString().replace(/[:.]/g, "-")}.csv`;
fs.writeFileSync(csvFile, "generation,module_id,recall1,latency\n");

let generation = 0;

(async function mainLoop() {
    console.log(`🔄 Starting experiment loop (max generations: ${maxGenerations})`);

    while (generation < maxGenerations) {
        generation++;
        console.log(`\n✨ Generation ${generation}`);

        // Load the current pool
        const pool = new ModulePool();
        const loaded = JSON.parse(fs.readFileSync("./module_pool.json", "utf8"));
        loaded.forEach((m: Module) => pool.addModule(m));

        // Evaluate
        for (const mod of pool.listModules()) {
            const start = performance.now();
            let recall1 = 0;

            if (mod.elm) {
                const embedding = mod.elm.computeHiddenLayer([queryVec])[0];
                const results = KNN.find(embedding, knnData, 5, 5, "cosine");
                recall1 = results[0].label === "3" ? 1 : 0;
            }
            if (mod.transformer) {
                const embedding = mod.transformer.encode(
                    query
                        .toLowerCase()
                        .split("")
                        .map((c) => mod.transformer!.vocab.tokenToIdx[c] ?? 1)
                );
                const results = KNN.find(embedding, knnData, 5, 5, "cosine");
                recall1 = results[0].label === "3" ? 1 : 0;
            }
            const latency = performance.now() - start;
            mod.metrics.recall1 = recall1;
            mod.metrics.avgLatency = latency;
            mod.lastEvaluated = Date.now();

            // Append to CSV
            fs.appendFileSync(
                csvFile,
                `${generation},${mod.id},${recall1},${latency.toFixed(2)}\n`
            );
        }

        // Save evaluated pool
        fs.writeFileSync(
            "./module_pool.json",
            JSON.stringify(pool.listModules(), null, 2)
        );
        console.log(`✅ Evaluation complete.`);

        // Evolve pool
        const evolvedPool = new ModulePool();
        const survivors = pool
            .listModules()
            .sort(
                (a, b) =>
                    (a.metrics.recall1 ?? 0) - (a.metrics.avgLatency ?? 1000) / 5000 -
                    ((b.metrics.recall1 ?? 0) - (b.metrics.avgLatency ?? 1000) / 5000)
            )
            .slice(-5);

        const newModules: Module[] = [];
        for (let i = 0; i < 5; i++) {
            const parent = survivors[i % survivors.length];
            if (parent.elm) {
                const dropout = Math.max(
                    0,
                    Math.min(
                        0.2,
                        (parent.elm.config.dropout ?? 0.02) + (Math.random() * 0.02 - 0.01)
                    )
                );
                const hiddenUnits = Math.max(
                    8,
                    parent.elm.hiddenUnits + Math.floor(Math.random() * 16 - 8)
                );
                const activation = ["relu", "tanh", "leakyRelu"][
                    Math.floor(Math.random() * 3)
                ];
                const child = new ELM({
                    ...parent.elm.config,
                    hiddenUnits,
                    activation,
                    dropout,
                });
                newModules.push({
                    id: `elm-${Date.now()}-${i}`,
                    type: "ELM",
                    elm: child,
                    role: "retrieval",
                    metrics: { recall1: 0, recall5: 0, mrr: 0, avgLatency: 0 },
                    lastEvaluated: Date.now(),
                });
            }
            if (parent.transformer) {
                const embedDim = Math.max(
                    4,
                    parent.transformer.embedDim + Math.floor(Math.random() * 4 - 2)
                );
                const seqLen = Math.max(
                    4,
                    parent.transformer.seqLen + Math.floor(Math.random() * 2 - 1)
                );
                const child = new MiniTransformer(parent.transformer.vocab);
                child.embedDim = embedDim;
                child.seqLen = seqLen;
                child.embedding = child.randomMatrix(
                    parent.transformer.vocab.tokens.length,
                    embedDim
                );
                child.posEnc = child.positionalEncoding(seqLen, embedDim);
                newModules.push({
                    id: `transformer-${Date.now()}-${i}`,
                    type: "Transformer",
                    transformer: child,
                    role: "retrieval",
                    metrics: { recall1: 0, recall5: 0, mrr: 0, avgLatency: 0 },
                    lastEvaluated: Date.now(),
                });
            }
        }

        const nextGenPool = new ModulePool();
        [...survivors, ...newModules].forEach((m) => nextGenPool.addModule(m));
        fs.writeFileSync(
            "./module_pool.json",
            JSON.stringify(nextGenPool.listModules(), null, 2)
        );
        console.log(
            `🌱 Evolved pool: kept ${survivors.length}, added ${newModules.length}`
        );
    }
})();
