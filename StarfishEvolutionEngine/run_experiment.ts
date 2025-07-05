// run_experiment.ts
import fs from "fs";
import { ModulePool, Module } from "../src/core/ModulePool";
import { TFIDFVectorizer } from "../src/ml/TFIDF";
import { KNN } from "../src/ml/KNN";
import yargs from "yargs";
import { ELM } from "../src/core/ELM";
import { Vocab, MiniTransformer } from "../src/core/MiniTransformer";

// Load corpus and parse sections
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

// CLI options
const argv = yargs()
    .option("generations", {
        alias: "g",
        description: "Number of generations to run",
        type: "number",
    })
    .help()
    .alias("help", "h")
    .parseSync();

const maxGenerations = argv.generations ?? Infinity;

// Prepare TFIDF vectors
const vectorizer = new TFIDFVectorizer(texts);
const tfidfVectors = texts.map((t) =>
    TFIDFVectorizer.l2normalize(vectorizer.vectorize(t))
);
const knnData = texts.map((t, i) => ({
    vector: tfidfVectors[i],
    label: String(i),
}));

// Queries to test
const queries = [
    "How do you declare a map in Go?",
    "What is a goroutine?",
    "How does defer work?",
    "How to use channels?",
    "Explain Go interfaces.",
];

// CSV log
const csvFile = `experiment_log_${new Date()
    .toISOString()
    .replace(/[:.]/g, "-")}.csv`;
fs.writeFileSync(csvFile, "generation,module_id,mean_recall1,latency\n");

// Initialize or load module pool
const pool = new ModulePool();

if (fs.existsSync("./module_pool.json")) {
    const loaded = JSON.parse(fs.readFileSync("./module_pool.json", "utf8"));
    loaded.forEach((m: Module) => {
        const hydrated: Module = {
            ...m,
            elm: m.elm ? new ELM(m.elm.config) : undefined,
            transformer: m.transformer
                ? new MiniTransformer(new Vocab(m.transformer.vocab.tokens))
                : undefined,
        };
        pool.addModule(hydrated);
    });
    console.log(`✅ Loaded existing module pool.`);
} else {
    console.log(`⚙️ No module_pool.json found. Creating initial random pool...`);

    const allChars = new Set<string>();
    rawText.split("").forEach((c) => allChars.add(c.toLowerCase()));
    const vocab = new Vocab(["<PAD>", "<UNK>", ...allChars]);

    // 5 ELM modules
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

    // 5 Transformer modules
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

    fs.writeFileSync(
        "./module_pool.json",
        JSON.stringify(pool.listModules(), null, 2)
    );
    console.log(`✅ Initialized module pool with 10 modules.`);
}

let generation = 0;

(async function mainLoop() {
    console.log(`🔄 Starting experiment loop (max generations: ${maxGenerations})`);

    while (generation < maxGenerations) {
        generation++;
        console.log(`\n✨ Generation ${generation}`);

        // Reload module pool fresh each generation
        const pool = new ModulePool();
        const loaded = JSON.parse(fs.readFileSync("./module_pool.json", "utf8"));
        loaded.forEach((m: Module) => {
            const hydrated: Module = {
                ...m,
                elm: m.elm ? new ELM(m.elm.config) : undefined,
                transformer: m.transformer
                    ? new MiniTransformer(new Vocab(m.transformer.vocab.tokens))
                    : undefined,
            };
            pool.addModule(hydrated);
        });

        // Evaluate modules
        for (const mod of pool.listModules()) {
            const start = performance.now();

            // If ELM untrained, train
            if (mod.elm && !mod.elm.model) {
                console.log(`⚙️ Training ${mod.id} before evaluation...`);
                mod.elm.trainFromData(tfidfVectors, tfidfVectors);
            }

            // Evaluate over all queries
            const recalls: number[] = [];

            for (const q of queries) {
                const queryVec = TFIDFVectorizer.l2normalize(vectorizer.vectorize(q));
                let recall = 0;

                if (mod.elm) {
                    const embedding = mod.elm.computeHiddenLayer([queryVec])[0];
                    const results = KNN.find(embedding, knnData, 5, 5, "cosine");
                    recall = results[0].label === "3" ? 1 : 0;
                }

                if (mod.transformer) {
                    const embedding = mod.transformer.encode(
                        q
                            .toLowerCase()
                            .split("")
                            .map((c) => mod.transformer!.vocab.tokenToIdx[c] ?? 1)
                    );
                    const results = KNN.find(embedding, knnData, 5, 5, "cosine");
                    recall = results[0].label === "3" ? 1 : 0;
                }

                recalls.push(recall);
            }

            const meanRecall = recalls.reduce((a, b) => a + b, 0) / recalls.length;
            const latency = performance.now() - start;

            mod.metrics.recall1 = meanRecall;
            mod.metrics.avgLatency = latency;
            mod.lastEvaluated = Date.now();

            fs.appendFileSync(
                csvFile,
                `${generation},${mod.id},${meanRecall.toFixed(3)},${latency.toFixed(2)}\n`
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
                    (a.metrics.recall1 ?? 0) -
                    (a.metrics.avgLatency ?? 1000) / 5000 -
                    ((b.metrics.recall1 ?? 0) -
                        (b.metrics.avgLatency ?? 1000) / 5000)
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
                        (parent.elm.config.dropout ?? 0.02) +
                        (Math.random() * 0.02 - 0.01)
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
